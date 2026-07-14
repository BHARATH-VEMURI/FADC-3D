"""V2 training script — uses UNet3DFADC_V2 (spatial k_att + warm init + temp anneal).

Self-contained — does NOT modify training/train_centralized.py.
Only difference from v1:
  - Imports models.unet_3d_fadc_v2.UNet3DFADC_V2 (instead of UNet3DFADC)
  - Model factory recognizes _v2 model names (unet3d_fadc_bottleneck_v2, etc.)
  - New CLI flags --k_att_temp_start, --k_att_temp_end (default 4.0 → 1.0)
  - Cosine-anneals k_att softmax temperature once per epoch via
    model.set_temperature(t)

Run example:
  python training/train_centralized_v2.py \
      --model unet3d_fadc_bottleneck_v2 \
      --output_dir outputs/fadc_bottleneck_v2_s42 \
      --seed 42 \
      --epochs 100 \
      --k_att_temp_start 4.0 --k_att_temp_end 1.0
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

sys.path.append(str(Path(__file__).parent.parent))
from data.mama_mia_dataset import build_centralized_loaders, DATA_ROOT
from models.unet_3d import UNet3D
from models.unet_3d_fadc_v2 import UNet3DFADC_V2
from fadc_3d_v2.omni_attention_3d_spatial import OmniAttention3DSpatial
from training.losses import DiceCELoss


# ─────────────────────────────────────────────
# 1. ARGS
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    PROJECT_ROOT = str(Path(__file__).parent.parent)
    parser.add_argument("--config",      type=str, default=os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    parser.add_argument("--data_root",   type=str, default=DATA_ROOT)
    parser.add_argument("--output_dir",  type=str, default="outputs/unet3d_fadc_v2")
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--batch_size",  type=int, default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--cache_rate",  type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--persistent_cache_dir",    type=str, default=None)
    parser.add_argument("--preprocessed_cache_dir",  type=str, default=None,
                        help="Path to .npz cache from scripts/preprocess_to_cache.py (fastest)")
    parser.add_argument("--resume",      type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--smoke_test",  action="store_true",
                        help="Run 2 epochs on 4 cases to verify pipeline end-to-end")
    parser.add_argument("--model",       type=str, default="unet3d_fadc_bottleneck_v2",
                        choices=[
                            # v1 names (still usable if user wants apples-to-apples baseline)
                            "unet3d",
                            # v2 FADC variants
                            "unet3d_fadc_v2",
                            "unet3d_fadc_encoder_v2",
                            "unet3d_fadc_bottleneck_v2",
                            "unet3d_fadc_mid_v2",
                            "unet3d_fadc_deep_v2",
                            "unet3d_fadc_deep_lite_v2",
                        ],
                        help="Model architecture to train (v2 variants use UNet3DFADC_V2)")
    parser.add_argument("--patch_size",  type=int, nargs=3, default=None,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (enables cudnn.deterministic)")
    parser.add_argument("--deep_supervision", action="store_true")
    # V2-specific
    parser.add_argument("--k_att_temp_start", type=float, default=4.0,
                        help="Initial softmax temperature for k_att (high=more mixing)")
    parser.add_argument("--k_att_temp_end",   type=float, default=0.8,
                        help="Final softmax temperature for k_att (1.0=normal, <1.0=sharper). "
                             "Default %(default)s. Reverted from the 0.5 endpoint after the "
                             "0.5 + short anneal combination destabilized k_att adaptation at ep30.")
    parser.add_argument("--k_att_anneal_epochs", type=int, default=None,
                        help="Anneal over this many epochs. Default %(default)s means "
                             "int(epochs * 0.6) on this branch so the anneal completes "
                             "at ~60%% of training and the final 40%% trains at t_end.")
    parser.add_argument("--val_every", type=int, default=None,
                        help="Override config.yaml training.val_every (validation cadence in epochs).")
    # Attention diversity aux loss
    parser.add_argument("--attn_diversity_weight", type=float, default=0.005,
                        help="Weight lambda for the c_att/f_att batch-diversity aux loss. "
                             "Default %(default)s (was 0.02 before the ep30 collapse). "
                             "0.0 disables the aux loss entirely.")
    parser.add_argument("--attn_diversity_target", type=float, default=0.03,
                        help="Target batch-std for c_att/f_att. The aux loss is a hinge: "
                             "clamp(target - actual_std, min=0), so once std reaches target "
                             "there is no penalty. 0.03 is 30x the current dead-layer values.")
    parser.add_argument("--attn_diversity_start_epoch", type=int, default=10,
                        help="Epoch (0-indexed) at which to start applying the aux loss. "
                             "Aux is disabled during warm-up before this epoch so seg loss "
                             "can settle. Default %(default)s.")
    parser.add_argument("--attn_diversity_layers", type=str, default="enc3,enc4",
                        help="Comma-separated dotted-path prefixes for layers the aux loss "
                             "should target. Empty string means ALL FADC layers (v2.1 "
                             "behaviour). Default '%(default)s' — apply only to deep encoder "
                             "layers where c_att/f_att collapse is most severe.")
    return parser.parse_args()


_DS_WEIGHTS = [1.0, 0.5, 0.25, 0.125]
_DS_WEIGHTS = [w / sum(_DS_WEIGHTS) for w in _DS_WEIGHTS]
_DS_SCALES  = [1.0, 0.5, 0.25, 0.125]


def deep_supervision_loss(preds_tuple, label, criterion):
    total = 0.0
    main_dice = None
    main_ce   = None
    for i, (p, w, s) in enumerate(zip(preds_tuple, _DS_WEIGHTS, _DS_SCALES)):
        if s == 1.0:
            lbl = label
        else:
            lbl = F.interpolate(label.float(), scale_factor=s, mode="nearest").to(label.dtype)
        sub_total, sub_dice, sub_ce = criterion(p, lbl)
        total = total + w * sub_total
        if i == 0:
            main_dice = sub_dice
            main_ce   = sub_ce
    return total, main_dice, main_ce


def load_config(config_path: str, args) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.epochs      is not None: cfg["training"]["epochs"]      = args.epochs
    if args.batch_size  is not None: cfg["training"]["batch_size"]  = args.batch_size
    if args.lr          is not None: cfg["training"]["lr"]          = args.lr
    if args.cache_rate  is not None: cfg["data"]["cache_rate"]      = args.cache_rate
    if args.num_workers is not None: cfg["data"]["num_workers"]     = args.num_workers
    if args.patch_size  is not None: cfg["data"]["patch_size"]      = args.patch_size

    cfg["model"]["deep_supervision"] = bool(args.deep_supervision)
    # Stamp V2 hyperparameters into cfg so they're saved with the checkpoint
    cfg["model"]["fadc_version"]     = "v2_spatial"
    cfg["training"]["k_att_temp_start"] = args.k_att_temp_start
    cfg["training"]["k_att_temp_end"]   = args.k_att_temp_end
    return cfg


# ─────────────────────────────────────────────
# 2. VALIDATION
# ─────────────────────────────────────────────

def validate(model, val_loader, dice_metric, post_pred, post_label, patch_size, device):
    model.eval()
    dice_metric.reset()
    iou_list  = []
    sens_list = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            with autocast("cuda", enabled=device.type == "cuda"):
                preds = sliding_window_inference(
                    inputs=images, roi_size=patch_size,
                    sw_batch_size=4, predictor=model, overlap=0.0)
            preds_bin  = [post_pred(i)          for i in decollate_batch(preds)]
            labels_bin = [post_label(i.long())  for i in decollate_batch(labels)]
            dice_metric(y_pred=preds_bin, y=labels_bin)
            pred_fg  = preds_bin[0][1].float()
            label_fg = labels_bin[0][1].float()
            tp = (pred_fg * label_fg).sum().item()
            fp = (pred_fg * (1 - label_fg)).sum().item()
            fn = ((1 - pred_fg) * label_fg).sum().item()
            iou_list.append(tp / (tp + fp + fn + 1e-6))
            sens_list.append(tp / (tp + fn + 1e-6))
    mean_dice = dice_metric.aggregate().item()
    mean_iou  = float(np.mean(iou_list))  if iou_list  else 0.0
    mean_sens = float(np.mean(sens_list)) if sens_list else 0.0
    dice_metric.reset()
    return mean_dice, mean_iou, mean_sens


# ─────────────────────────────────────────────
# 3. TEMPERATURE SCHEDULE
# ─────────────────────────────────────────────

def k_att_temperature(epoch: int, anneal_epochs: int, t_start: float, t_end: float) -> float:
    """Cosine anneal from t_start at epoch 0 to t_end at epoch anneal_epochs-1.
    After anneal_epochs, stays at t_end."""
    if anneal_epochs <= 1:
        return t_end
    e = min(max(epoch, 0), anneal_epochs - 1)
    cos_term = 0.5 * (1.0 + math.cos(math.pi * e / (anneal_epochs - 1)))
    return t_end + (t_start - t_end) * cos_term


# ─────────────────────────────────────────────
# 4. MODEL FACTORY
# ─────────────────────────────────────────────

_V2_PLACEMENT = {
    "unet3d_fadc_v2":            "full",
    "unet3d_fadc_encoder_v2":    "encoder",
    "unet3d_fadc_bottleneck_v2": "bottleneck",
    "unet3d_fadc_mid_v2":        "mid",
    "unet3d_fadc_deep_v2":       "deep",
    "unet3d_fadc_deep_lite_v2":  "deep_lite",
}


def build_model(args, model_kwargs, fadc_kwargs, device):
    if args.model in _V2_PLACEMENT:
        placement = _V2_PLACEMENT[args.model]
        return UNet3DFADC_V2(**model_kwargs, fadc_placement=placement, **fadc_kwargs).to(device)
    if args.model == "unet3d":
        if args.deep_supervision:
            raise ValueError("--deep_supervision is only wired for FADC variants.")
        return UNet3D(**model_kwargs).to(device)
    raise ValueError(f"Unknown --model: {args.model}")


# ─────────────────────────────────────────────
# 5. ATTENTION DIVERSITY AUX LOSS
# ─────────────────────────────────────────────
# Diagnostics on Encoder v2/v2.1 show c_att and f_att adaptation collapses
# past enc1 (batch-std < 0.005 at every deeper layer, dropping to 0.0002 at
# enc4.c2 by ep30). k_att is fine because its 3x3x3 spatial trunk bypasses
# the GAP that starves c_att/f_att of per-input signal. This aux loss
# rewards non-trivial batch variation in c/f/s/k attention directly, so
# gradients flow into channel_fc / filter_fc / spatial_fc / kernel_spatial_head
# instead of drifting toward constant outputs.
#
# v2.2 extension: collect s_att and k_att too.
#   c_att (B, C_in,  1, 1, 1)              — hinge on batch-std, avg over C
#   f_att (B, C_out, 1, 1, 1)              — hinge on batch-std, avg over C
#   s_att (B, 1, 1, 1, 1, K, K, K)          — hinge on batch-std, avg over K^3
#   k_att (B, num_branches, D, H, W)        — spatial-mean per (B, branch)
#                                             first (so we reward input
#                                             adaptivity of branch preference,
#                                             not just per-voxel texture),
#                                             then hinge on batch-std over B.
# Total training loss = seg_loss + lambda * aux_loss.

class _AttnStore:
    """Per-batch collector for c/f/s/k attention tensors from every FADC layer."""
    def __init__(self):
        self.enabled = False
        self.c_atts = []
        self.f_atts = []
        self.s_atts = []
        self.k_atts = []

    def clear(self):
        self.c_atts.clear()
        self.f_atts.clear()
        self.s_atts.clear()
        self.k_atts.clear()


def _name_matches_prefixes(name: str, prefixes) -> bool:
    """Return True if `name` equals or is a dotted-descendant of any prefix.
    e.g. name='enc3.conv.conv1.omni_att', prefixes=['enc3','enc4'] -> True.
    prefixes=None or [] treats every layer as a match (backwards compat)."""
    if not prefixes:
        return True
    for p in prefixes:
        p = p.strip()
        if not p:
            continue
        if name == p or name.startswith(p + "."):
            return True
    return False


def register_attention_hooks(model, layer_prefixes=None):
    """Register a forward hook on every OmniAttention3DSpatial that appends
    c_att / f_att to the store when store.enabled is True.

    If `layer_prefixes` is provided (list of dotted-path prefixes such as
    ['enc3','enc4']), only modules whose dotted name matches one of those
    prefixes get a hook. None / empty list = hook every FADC layer.

    Returns (store, handles, hooked_names). Caller must remove handles."""
    store = _AttnStore()
    handles = []
    hooked_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, OmniAttention3DSpatial):
            if not _name_matches_prefixes(name, layer_prefixes):
                continue
            def _hook(_module, _inp, out, _store=store):
                if not _store.enabled:
                    return
                c_att, f_att, s_att, k_att = out
                _store.c_atts.append(c_att)
                _store.f_atts.append(f_att)
                if torch.is_tensor(s_att):
                    _store.s_atts.append(s_att)
                if torch.is_tensor(k_att):
                    _store.k_atts.append(k_att)
            handles.append(mod.register_forward_hook(_hook))
            hooked_names.append(name)
    return store, handles, hooked_names


def _hinge_batch_std(a: torch.Tensor, target_std: float):
    """Hinge on the mean batch-std across all non-batch dims.
    Returns None when there are fewer than 2 samples (std ill-defined)."""
    if a.size(0) < 2:
        return None
    a = a.float()                                              # stable under autocast
    std = a.std(dim=0, unbiased=False)                         # non-batch dims
    return (target_std - std.mean()).clamp(min=0)


def attention_diversity_loss(store, target_std: float, device):
    """Hinge penalty on low batch-std of c / f / s / k attentions.

    c/f/s: raw batch-std averaged over remaining dims.
    k    : spatial-mean per (B, branch) first, THEN batch-std — this targets
           input-adaptivity of branch preference, not per-voxel texture.
    """
    per_layer = []

    # c_att, f_att, s_att — treat identically: hinge on raw batch-std.
    for a in store.c_atts + store.f_atts + store.s_atts:
        term = _hinge_batch_std(a, target_std)
        if term is not None:
            per_layer.append(term)

    # k_att — spatial-mean over (D, H, W) first, then hinge on batch-std.
    for k in store.k_atts:
        if k.size(0) < 2:
            continue
        k_summary = k.float().mean(dim=(2, 3, 4))              # (B, num_branches)
        std_per_branch = k_summary.std(dim=0, unbiased=False)  # (num_branches,)
        per_layer.append((target_std - std_per_branch.mean()).clamp(min=0))

    if not per_layer:
        return torch.zeros((), device=device)
    return torch.stack(per_layer).mean()


# ─────────────────────────────────────────────
# 6. TRAIN
# ─────────────────────────────────────────────

def train(cfg, args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[seed] seeded with {args.seed} (cudnn.deterministic=True)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_size  = tuple(cfg["data"]["patch_size"])
    epochs      = cfg["training"]["epochs"]
    lr          = cfg["training"]["lr"]
    batch_size  = cfg["training"]["batch_size"]
    cache_rate  = cfg["data"]["cache_rate"]
    num_workers = cfg["data"]["num_workers"]
    val_every   = cfg["training"]["val_every"]
    if args.val_every is not None:
        val_every = args.val_every
        print(f"val_every overridden by CLI: {val_every}")

    split_csv = os.path.join(args.data_root, "train_test_splits.csv")
    if not os.path.exists(split_csv):
        split_csv = None
        print("Warning: train_test_splits.csv not found — using all data for train")

    if args.smoke_test:
        print("SMOKE TEST MODE — 4 cases, 2 epochs")
        epochs      = 2
        cache_rate  = 0.0
        num_workers = 0
        val_every   = 1

    train_loader, val_loader = build_centralized_loaders(
        data_root=args.data_root,
        split_csv=split_csv,
        cache_rate=cache_rate,
        num_workers=num_workers,
        batch_size=batch_size,
        max_cases=4 if args.smoke_test else None,
        persistent_cache_dir=args.persistent_cache_dir or "",
        preprocessed_cache_dir=args.preprocessed_cache_dir or "",
        patch_size=patch_size,
        seed=args.seed,
    )

    model_kwargs = dict(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_filters=cfg["model"]["base_filters"],
    )
    fadc_kwargs = dict(deep_supervision=args.deep_supervision)
    model = build_model(args, model_kwargs, fadc_kwargs, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} (v2) | Parameters: {total_params:,}")
    print(f"k_att temperature anneal: {args.k_att_temp_start} -> {args.k_att_temp_end} "
          f"over {args.k_att_anneal_epochs or epochs} epochs")

    criterion = DiceCELoss(
        dice_weight=cfg["training"]["dice_weight"],
        ce_weight=cfg["training"]["ce_weight"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    warmup_epochs = args.warmup_epochs
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)

    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred   = AsDiscrete(argmax=True, to_onehot=2)
    post_label  = AsDiscrete(to_onehot=2)

    start_epoch = 0
    best_dice   = 0.0
    train_log   = []

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        print(f"Resumed from epoch {start_epoch} | best Dice: {best_dice:.4f}")

    # k_att anneal window. If unset, default to ~60% of total epochs so the
    # final 40% trains at t_end rather than a still-annealing softmax.
    if args.k_att_anneal_epochs is not None:
        anneal_epochs = args.k_att_anneal_epochs
    else:
        anneal_epochs = max(1, int(round(epochs * 0.6)))
    print(f"k_att anneal window: {anneal_epochs} epochs "
          f"({args.k_att_temp_start} -> {args.k_att_temp_end}); "
          f"held at {args.k_att_temp_end} for the remaining {epochs - anneal_epochs} epochs")

    # Attention diversity aux loss setup.
    # Parse comma-separated layer-prefix filter — empty string means all.
    layer_prefixes = [p.strip() for p in args.attn_diversity_layers.split(",") if p.strip()]
    attn_store, attn_hook_handles, hooked_names = register_attention_hooks(
        model, layer_prefixes=layer_prefixes or None)
    aux_configured = args.attn_diversity_weight > 0.0
    if aux_configured:
        if hooked_names:
            layer_desc = (", ".join(layer_prefixes) if layer_prefixes else "all")
            print(f"Attention diversity aux loss: lambda={args.attn_diversity_weight}, "
                  f"target_std={args.attn_diversity_target}, "
                  f"start_epoch={args.attn_diversity_start_epoch}, "
                  f"layer_prefixes=[{layer_desc}] "
                  f"({len(hooked_names)} layer(s) hooked: {hooked_names})")
        else:
            print(f"WARNING: layer_prefixes {layer_prefixes} matched 0 FADC layers. "
                  f"Aux loss is configured but will always be 0.")
            aux_configured = False
    else:
        print("Attention diversity aux loss: DISABLED (lambda=0)")

    print(f"\nStarting training: {epochs} epochs | LR {lr} | Batch {batch_size}")
    print("=" * 70)

    aux_was_active_prev = False
    for epoch in range(start_epoch, epochs):
        # V2 — set k_att temperature for this epoch BEFORE training
        if hasattr(model, "set_temperature"):
            t = k_att_temperature(epoch, anneal_epochs,
                                  args.k_att_temp_start, args.k_att_temp_end)
            model.set_temperature(t)
        else:
            t = float("nan")

        # Aux loss gating — only active from --attn_diversity_start_epoch onward.
        aux_active_this_epoch = aux_configured and (epoch >= args.attn_diversity_start_epoch)
        if aux_active_this_epoch and not aux_was_active_prev:
            print(f"  --> Attention diversity aux loss ACTIVATED at epoch {epoch+1}")
        aux_was_active_prev = aux_active_this_epoch

        model.train()
        epoch_loss  = 0.0
        epoch_dice  = 0.0
        epoch_ce    = 0.0
        epoch_aux   = 0.0
        num_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{epochs} T={t:.2f}",
                    leave=False, ncols=110, unit="batch", file=sys.stdout)

        for batch in pbar:
            if isinstance(batch["image"], list):
                imgs = [torch.from_numpy(x.copy()) if isinstance(x, np.ndarray) else x
                        for x in batch["image"]]
                lbls = [torch.from_numpy(x.copy()) if isinstance(x, np.ndarray) else x
                        for x in batch["label"]]
                images = torch.cat(imgs, dim=0).to(device)
                labels = torch.cat(lbls, dim=0).to(device)
            else:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

            optimizer.zero_grad()
            attn_store.clear()
            attn_store.enabled = aux_active_this_epoch
            with autocast("cuda", enabled=device.type == "cuda"):
                preds = model(images)
                if isinstance(preds, tuple):
                    total_loss, dice_loss, ce_loss = deep_supervision_loss(preds, labels, criterion)
                else:
                    total_loss, dice_loss, ce_loss = criterion(preds, labels)

                if aux_active_this_epoch:
                    aux_loss = attention_diversity_loss(
                        attn_store, args.attn_diversity_target, device)
                    total_loss = total_loss + args.attn_diversity_weight * aux_loss
                else:
                    aux_loss = torch.zeros((), device=device)
            attn_store.enabled = False

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += total_loss.item()
            epoch_dice  += dice_loss.item()
            epoch_ce    += ce_loss.item()
            epoch_aux   += float(aux_loss.detach().item())
            num_batches += 1

        pbar.close()
        scheduler.step()

        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        avg_ce   = epoch_ce   / num_batches
        avg_aux  = epoch_aux  / num_batches
        elapsed  = time.time() - t0
        lr_now   = scheduler.get_last_lr()[0]
        mins, secs = divmod(int(elapsed), 60)
        print(f"Epoch {epoch+1:03d}/{epochs} | T={t:.2f} | "
              f"Loss {avg_loss:.4f} | Dice {avg_dice:.4f} | CE {avg_ce:.4f} | "
              f"Aux {avg_aux:.4f} | LR {lr_now:.2e} | {mins}m{secs}s")

        log_entry = {
            "epoch":     epoch + 1,
            "loss":      avg_loss,
            "dice_loss": avg_dice,
            "ce_loss":   avg_ce,
            "aux_loss":  avg_aux,
            "lr":        lr_now,
            "k_att_temperature": t,
            "time_s":    elapsed,
        }

        if (epoch + 1) % val_every == 0:
            val_dice, val_iou, val_sens = validate(
                model, val_loader, dice_metric,
                post_pred, post_label, patch_size, device)
            marker = " <-- BEST" if val_dice > best_dice else ""
            print(f"  Val Dice {val_dice:.4f} | IoU {val_iou:.4f} | Sens {val_sens:.4f}"
                  f"  (best {best_dice:.4f}){marker}")
            log_entry["val_dice"]        = val_dice
            log_entry["val_iou"]         = val_iou
            log_entry["val_sensitivity"] = val_sens

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({
                    "epoch":      epoch,
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "scheduler":  scheduler.state_dict(),
                    "best_dice":  best_dice,
                    "config":     cfg,
                    "model_name": args.model,
                }, output_dir / "best_model.pth")
                print(f"  *** NEW BEST Dice {best_dice:.4f} -> {output_dir}/best_model.pth ***")

        train_log.append(log_entry)

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "best_dice":  best_dice,
                "config":     cfg,
                "model_name": args.model,
            }, output_dir / "latest_checkpoint.pth")

    with open(output_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)
    with open(output_dir / "meta.json", "w") as f:
        json.dump({
            "seed": args.seed,
            "model": args.model,
            "fadc_version": "v2_spatial",
            "k_att_temp_start": args.k_att_temp_start,
            "k_att_temp_end": args.k_att_temp_end,
            "k_att_anneal_epochs": anneal_epochs,
            "attn_diversity_weight": args.attn_diversity_weight,
            "attn_diversity_target": args.attn_diversity_target,
            "attn_diversity_start_epoch": args.attn_diversity_start_epoch,
            "attn_diversity_layers": args.attn_diversity_layers,
            "attn_diversity_hooked_names": hooked_names,
            "best_dice": best_dice,
            "epochs": epochs,
        }, f, indent=2)

    for h in attn_hook_handles:
        h.remove()

    print(f"\nTraining complete. Best Val Dice: {best_dice:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config, args)
    train(cfg, args)
