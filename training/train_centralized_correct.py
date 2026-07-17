"""Training script for the CORRECTED FADC3D encoder-first experiment.

Self-contained. Does not modify training/train_centralized_v2.py or the plain
UNet3D training path — this is a fresh script bound to
`models/unet_3d_fadc_correct.py`.

Design decisions vs the v2 script
---------------------------------
- Attention-diversity aux loss is DEFAULTED OFF (weight=0.0). Segmentation loss
  is the primary objective. The corrected block relies on kernel-side attention
  (baked into W_adaptive) and voxelwise k_att; there is no need to bootstrap
  c/f "batch-std" as a proxy target.
- Sliding-window overlap is CONFIGURABLE via --val_overlap (default 0.5). Prior
  runs used 0.0; the spec requires 0.5 as default for the corrected experiment.
- Validation metrics (Dice / IoU / Sensitivity) are computed per-CASE across
  the whole batch (rather than reading `preds_bin[0][1]` and skipping the rest).
- Checkpoints record the exact model_name / placement / architecture identity,
  and resume refuses to load a checkpoint whose model_name does not match.
- Temperature schedule matches the corrected default: 2.0 -> 1.0 cosine over
  60 epochs. After anneal_epochs, temperature stays at 1.0.

CLI example (mirrors the encoder notebook):

    python training/train_centralized_correct.py \
        --model unet3d_fadc_encoder_correct \
        --output_dir outputs/fadc3d_correct_encoder_s42 \
        --seed 42 \
        --epochs 100 \
        --batch_size 2 \
        --patch_size 128 128 64 \
        --deep_supervision \
        --k_att_temp_start 2.0 --k_att_temp_end 1.0 --k_att_anneal_epochs 60 \
        --val_every 10 --val_overlap 0.5
"""
from __future__ import annotations

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
from models.unet_3d_fadc_correct import (
    build_unet3d_fadc_correct, MODEL_NAMES, EXPECTED_ADAPTIVE_CONV_COUNT,
    UNet3DFADCCorrect,
)
from fadc_3d_correct.adaptive_dilated_conv_3d import AdaptiveDilatedConv3D
from training.losses import DiceCELoss


# ────────────────────────────── ARGS


def parse_args():
    p = argparse.ArgumentParser()
    ROOT = str(Path(__file__).parent.parent)
    p.add_argument("--config", type=str, default=os.path.join(ROOT, "configs", "config.yaml"))
    p.add_argument("--data_root", type=str, default=DATA_ROOT)
    p.add_argument("--output_dir", type=str, default="outputs/unet3d_fadc_correct")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--cache_rate", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--persistent_cache_dir", type=str, default=None)
    p.add_argument("--preprocessed_cache_dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from. Refuses architecture mismatch.")
    p.add_argument("--smoke_test", action="store_true",
                   help="2 epochs on 4 cases — verifies the end-to-end pipeline.")
    p.add_argument("--model", type=str, default="unet3d_fadc_encoder_correct",
                   choices=list(MODEL_NAMES))
    p.add_argument("--patch_size", type=int, nargs=3, default=None,
                   metavar=("X", "Y", "Z"))
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deep_supervision", action="store_true")

    # k_att temperature schedule
    p.add_argument("--k_att_temp_start", type=float, default=2.0,
                   help="Initial k_att softmax temperature. Default 2.0.")
    p.add_argument("--k_att_temp_end", type=float, default=1.0,
                   help="Final k_att softmax temperature. Default 1.0.")
    p.add_argument("--k_att_anneal_epochs", type=int, default=60,
                   help="Cosine anneal window in epochs. Default 60. After that, T stays at t_end.")

    # Segmentation / auxiliary
    p.add_argument("--attn_diversity_weight", type=float, default=0.0,
                   help="Reserved. 0.0 disables the aux loss (default for corrected experiment).")

    # AdaKern / FreqSel toggles (research knobs; keep default off).
    p.add_argument("--use_position_att", action="store_true",
                   help="Enable optional 27-position s_att inside AdaKern3D.")

    # Validation
    p.add_argument("--val_every", type=int, default=None,
                   help="Override config.yaml training.val_every.")
    p.add_argument("--val_overlap", type=float, default=0.5,
                   help="Sliding-window inference overlap for validation. Default 0.5 per spec.")

    return p.parse_args()


# ────────────────────────────── DEEP SUPERVISION

_DS_WEIGHTS = [1.0, 0.5, 0.25, 0.125]
_DS_WEIGHTS = [w / sum(_DS_WEIGHTS) for w in _DS_WEIGHTS]
_DS_SCALES = [1.0, 0.5, 0.25, 0.125]


def deep_supervision_loss(preds_tuple, label, criterion):
    total = 0.0
    main_dice = None
    main_ce = None
    for i, (pred, w, s) in enumerate(zip(preds_tuple, _DS_WEIGHTS, _DS_SCALES)):
        if s == 1.0:
            lbl = label
        else:
            lbl = F.interpolate(label.float(), scale_factor=s, mode="nearest").to(label.dtype)
        sub_total, sub_dice, sub_ce = criterion(pred, lbl)
        total = total + w * sub_total
        if i == 0:
            main_dice = sub_dice
            main_ce = sub_ce
    return total, main_dice, main_ce


# ────────────────────────────── VALIDATION

def _iou_sens_per_case(pred_fg: torch.Tensor, label_fg: torch.Tensor) -> tuple[float, float]:
    """Voxelwise TP/FP/FN over a single case; returns (iou, sensitivity)."""
    tp = (pred_fg * label_fg).sum().item()
    fp = (pred_fg * (1 - label_fg)).sum().item()
    fn = ((1 - pred_fg) * label_fg).sum().item()
    iou = tp / (tp + fp + fn + 1e-6)
    sens = tp / (tp + fn + 1e-6)
    return iou, sens


def validate(model, val_loader, dice_metric, post_pred, post_label,
             patch_size, device, overlap: float = 0.5):
    """Sliding-window validation — Dice, IoU, Sensitivity per CASE (not just batch[0])."""
    model.eval()
    dice_metric.reset()
    iou_list, sens_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            with autocast("cuda", enabled=device.type == "cuda"):
                preds = sliding_window_inference(
                    inputs=images, roi_size=patch_size,
                    sw_batch_size=4, predictor=model,
                    overlap=overlap,
                )
            preds_bin_list = [post_pred(p) for p in decollate_batch(preds)]
            labels_bin_list = [post_label(l.long()) for l in decollate_batch(labels)]
            dice_metric(y_pred=preds_bin_list, y=labels_bin_list)
            # Aggregate IoU / Sensitivity over every case in the batch, not just [0].
            for pb, lb in zip(preds_bin_list, labels_bin_list):
                iou, sens = _iou_sens_per_case(pb[1].float(), lb[1].float())
                iou_list.append(iou)
                sens_list.append(sens)
    mean_dice = dice_metric.aggregate().item()
    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    mean_sens = float(np.mean(sens_list)) if sens_list else 0.0
    dice_metric.reset()
    return mean_dice, mean_iou, mean_sens


# ────────────────────────────── TEMPERATURE

def k_att_temperature(epoch: int, anneal_epochs: int, t_start: float, t_end: float) -> float:
    if anneal_epochs <= 1:
        return t_end
    e = min(max(epoch, 0), anneal_epochs - 1)
    cos_term = 0.5 * (1.0 + math.cos(math.pi * e / (anneal_epochs - 1)))
    return t_end + (t_start - t_end) * cos_term


# ────────────────────────────── CONFIG

def load_config(config_path: str, args) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.cache_rate is not None:
        cfg["data"]["cache_rate"] = args.cache_rate
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.patch_size is not None:
        cfg["data"]["patch_size"] = args.patch_size

    cfg["model"]["deep_supervision"] = bool(args.deep_supervision)
    cfg["model"]["fadc_version"] = "correct"
    cfg["model"]["fadc_placement"] = args.model
    cfg["training"]["k_att_temp_start"] = args.k_att_temp_start
    cfg["training"]["k_att_temp_end"] = args.k_att_temp_end
    cfg["training"]["k_att_anneal_epochs"] = args.k_att_anneal_epochs
    cfg["training"]["val_overlap"] = args.val_overlap
    cfg["fadc_correct"] = {
        "use_position_att": bool(args.use_position_att),
    }
    return cfg


# ────────────────────────────── ARCHITECTURE-IDENTITY GUARDS

def _arch_identity(model_name: str, model_cfg: dict, extras: dict) -> dict:
    """Small dict recording the essential architecture identity of a checkpoint."""
    return {
        "model_name": model_name,
        "in_channels": int(model_cfg["in_channels"]),
        "out_channels": int(model_cfg["out_channels"]),
        "base_filters": int(model_cfg["base_filters"]),
        "deep_supervision": bool(model_cfg.get("deep_supervision", False)),
        "fadc_correct": dict(extras),
    }


def _require_matching_arch(ckpt: dict, current: dict) -> None:
    ckpt_arch = ckpt.get("arch_identity")
    if ckpt_arch is None:
        raise RuntimeError(
            "Refusing to resume: checkpoint has no 'arch_identity'. Almost certainly "
            "from an older run with a different architecture."
        )
    for key in ("model_name", "in_channels", "out_channels", "base_filters", "deep_supervision"):
        if ckpt_arch.get(key) != current.get(key):
            raise RuntimeError(
                f"Refusing to resume: architecture mismatch on '{key}': "
                f"ckpt={ckpt_arch.get(key)}  current={current.get(key)}"
            )
    # fadc_correct is a nested dict; compare keys we actually train with.
    ckpt_fc = ckpt_arch.get("fadc_correct", {})
    cur_fc = current.get("fadc_correct", {})
    if ckpt_fc.get("use_position_att", False) != cur_fc.get("use_position_att", False):
        raise RuntimeError(
            f"Refusing to resume: use_position_att mismatch (ckpt={ckpt_fc.get('use_position_att')} "
            f"vs current={cur_fc.get('use_position_att')})"
        )


# ────────────────────────────── MECHANISM STATISTICS

@torch.no_grad()
def snapshot_mechanism_stats(model: UNet3DFADCCorrect) -> dict:
    """Aggregate cheap per-forward mechanism statistics without retaining graph.

    Uses the diagnostic caches populated on every AdaptiveDilatedConv3D forward:
      last_k_att, last_expected_dilation, last_c_low/f_low/c_high/f_high, last_s_att
    """
    exp_dils, k_ents = [], []
    frac_d1, frac_d2, frac_d3 = [], [], []
    c_low_std, f_low_std, c_high_std, f_high_std = [], [], [], []
    for m in model.modules():
        if not isinstance(m, AdaptiveDilatedConv3D):
            continue
        if m.last_expected_dilation is None:
            continue
        ed = m.last_expected_dilation.detach().float()
        exp_dils.append(ed.mean().item())
        # k_att entropy (bits) averaged per voxel
        p = m.last_k_att.detach().float().clamp(min=1e-12)
        entropy = -(p * p.log()).sum(dim=1)
        k_ents.append(entropy.mean().item())
        # argmax fraction per branch
        arg = p.argmax(dim=1)
        n = arg.numel()
        frac_d1.append((arg == 0).float().sum().item() / n)
        frac_d2.append((arg == 1).float().sum().item() / n)
        frac_d3.append((arg == 2).float().sum().item() / n)
        # per-batch std of c/f attention (input adaptivity proxy)
        if m.last_c_low is not None and m.last_c_low.size(0) > 1:
            c_low_std.append(m.last_c_low.float().std(dim=0).mean().item())
            f_low_std.append(m.last_f_low.float().std(dim=0).mean().item())
            c_high_std.append(m.last_c_high.float().std(dim=0).mean().item())
            f_high_std.append(m.last_f_high.float().std(dim=0).mean().item())

    def _stat(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "expected_dilation_mean": _stat(exp_dils),
        "k_att_entropy_mean": _stat(k_ents),
        "frac_argmax_d1": _stat(frac_d1),
        "frac_argmax_d2": _stat(frac_d2),
        "frac_argmax_d3": _stat(frac_d3),
        "c_low_batch_std": _stat(c_low_std),
        "f_low_batch_std": _stat(f_low_std),
        "c_high_batch_std": _stat(c_high_std),
        "f_high_batch_std": _stat(f_high_std),
    }


# ────────────────────────────── TRAIN

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

    patch_size = tuple(cfg["data"]["patch_size"])
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["lr"]
    batch_size = cfg["training"]["batch_size"]
    cache_rate = cfg["data"]["cache_rate"]
    num_workers = cfg["data"]["num_workers"]
    val_every = cfg["training"]["val_every"]
    if args.val_every is not None:
        val_every = args.val_every

    split_csv = os.path.join(args.data_root, "train_test_splits.csv")
    if not os.path.exists(split_csv):
        split_csv = None
        print("WARNING: train_test_splits.csv not found — using all data for train")

    if args.smoke_test:
        print("SMOKE TEST MODE — 4 cases, 2 epochs")
        epochs = 2
        cache_rate = 0.0
        num_workers = 0
        val_every = 1

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
    adakern_cfg = dict(use_position_att=bool(args.use_position_att))
    model = build_unet3d_fadc_correct(
        model_name=args.model,
        **model_kwargs,
        deep_supervision=args.deep_supervision,
        adakern_cfg=adakern_cfg,
    ).to(device)

    # Enforce placement counts before training starts.
    placement = args.model.split("_")[2]
    n_adapt = model.count_adaptive_convs()
    expected = EXPECTED_ADAPTIVE_CONV_COUNT[placement]
    if n_adapt != expected:
        raise RuntimeError(
            f"Model {args.model} has {n_adapt} corrected adaptive convs, expected {expected}."
        )
    print(f"Model: {args.model}  ({placement})  adaptive_convs={n_adapt}/{expected}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"k_att temperature schedule: {args.k_att_temp_start} -> {args.k_att_temp_end} "
          f"over {args.k_att_anneal_epochs} epochs")
    print(f"val_overlap: {args.val_overlap}")

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
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    start_epoch = 0
    best_dice = 0.0
    train_log = []

    # architecture identity gets saved with every checkpoint
    fadc_extras = {"use_position_att": bool(args.use_position_att)}
    arch_id = _arch_identity(args.model, cfg["model"], fadc_extras)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        _require_matching_arch(ckpt, arch_id)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_dice = float(ckpt.get("best_dice", 0.0))
        print(f"Resumed from epoch {start_epoch} | best Dice: {best_dice:.4f}")

    print(f"\nStarting training: {epochs} epochs | LR {lr} | Batch {batch_size}")
    print("=" * 70)

    for epoch in range(start_epoch, epochs):
        # k_att temperature for this epoch
        t = k_att_temperature(epoch, args.k_att_anneal_epochs,
                              args.k_att_temp_start, args.k_att_temp_end)
        model.set_temperature(t)

        model.train()
        epoch_loss = epoch_dice = epoch_ce = 0.0
        num_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{epochs} T={t:.2f}",
                    leave=False, ncols=110, unit="batch", file=sys.stdout)

        last_stats = None
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
            with autocast("cuda", enabled=device.type == "cuda"):
                preds = model(images)
                if isinstance(preds, tuple):
                    total_loss, dice_loss, ce_loss = deep_supervision_loss(preds, labels, criterion)
                else:
                    total_loss, dice_loss, ce_loss = criterion(preds, labels)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            epoch_dice += dice_loss.item()
            epoch_ce += ce_loss.item()
            num_batches += 1

        # detach the cached diag tensors so nothing keeps a computation graph
        last_stats = snapshot_mechanism_stats(model)

        pbar.close()
        scheduler.step()

        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        avg_ce = epoch_ce / num_batches
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        mins, secs = divmod(int(elapsed), 60)
        print(f"Epoch {epoch+1:03d}/{epochs} | T={t:.2f} | "
              f"Loss {avg_loss:.4f} | Dice {avg_dice:.4f} | CE {avg_ce:.4f} | "
              f"LR {lr_now:.2e} | E[dil]={last_stats['expected_dilation_mean']:.3f} | "
              f"{mins}m{secs}s")

        log_entry = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "dice_loss": avg_dice,
            "ce_loss": avg_ce,
            "lr": lr_now,
            "k_att_temperature": t,
            "time_s": elapsed,
            **{f"mech.{k}": v for k, v in last_stats.items()},
        }

        if (epoch + 1) % val_every == 0:
            val_dice, val_iou, val_sens = validate(
                model, val_loader, dice_metric,
                post_pred, post_label, patch_size, device,
                overlap=args.val_overlap)
            marker = " <-- BEST" if val_dice > best_dice else ""
            print(f"  Val Dice {val_dice:.4f} | IoU {val_iou:.4f} | Sens {val_sens:.4f}"
                  f"  (best {best_dice:.4f}){marker}")
            log_entry["val_dice"] = val_dice
            log_entry["val_iou"] = val_iou
            log_entry["val_sensitivity"] = val_sens

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_dice": best_dice,
                    "config": cfg,
                    "model_name": args.model,
                    "arch_identity": arch_id,
                }, output_dir / "best_model.pth")
                print(f"  *** NEW BEST Dice {best_dice:.4f} -> {output_dir}/best_model.pth ***")

        train_log.append(log_entry)

        # last checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice": best_dice,
            "config": cfg,
            "model_name": args.model,
            "arch_identity": arch_id,
        }, output_dir / "last_checkpoint.pth")

    with open(output_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)
    with open(output_dir / "meta.json", "w") as f:
        json.dump({
            "seed": args.seed,
            "model": args.model,
            "fadc_version": "correct",
            "arch_identity": arch_id,
            "k_att_temp_start": args.k_att_temp_start,
            "k_att_temp_end": args.k_att_temp_end,
            "k_att_anneal_epochs": args.k_att_anneal_epochs,
            "val_overlap": args.val_overlap,
            "best_dice": best_dice,
            "epochs": epochs,
        }, f, indent=2)

    print(f"\nTraining complete. Best Val Dice: {best_dice:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config, args)
    train(cfg, args)
