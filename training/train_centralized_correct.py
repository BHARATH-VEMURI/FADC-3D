"""Training script for the CORRECTED FADC3D encoder-first experiment.

Self-contained. Does NOT modify training/train_centralized_v2.py or the plain
UNet3D training path — this is a fresh script bound to
`models/unet_3d_fadc_correct.py`.

Overview
--------
- Segmentation loss only. Attention-diversity aux is disabled by default.
- Validation is now split into a FAST proxy (overlap 0.0, run frequently) and a
  FORMAL (overlap 0.5, run rarely or in a separate session).
- Only formal validation updates `best_model.pth`. Fast validation writes to
  `best_fast_model.pth` — clearly labelled as a proxy so it is never
  compared with formal Dice.
- Every checkpoint (including `last_checkpoint.pth`) is written ATOMICALLY:
  temp file in the same dir, then `os.replace()`.
- `last_checkpoint.pth` is saved BEFORE validation. If validation is
  interrupted, resume advances to the next epoch and never repeats work.
- Checkpoints carry `epoch`, `model`, `optimizer`, `scheduler`, `scaler`,
  `best_dice`, `best_fast_dice`, `arch_identity`, `config`, and the full
  `train_log` accumulated so far.
- Resume is backward compatible with older checkpoints that lack
  `scaler` / `best_fast_dice` / `train_log`. It prints exactly what it
  restored.

Validation-schedule resolution
------------------------------
- `--fast_val_every N` (default 10)   : fast every N epochs.
- `--fast_val_overlap O` (default 0.0)
- `--fast_val_max_cases K` (default 0=all) : deterministic first-K cases when >0.
- `--formal_val_every N` (default 0=off) : formal every N epochs, all 306 cases.
- `--formal_val_overlap O` (default 0.5)
- LEGACY: `--val_every` / `--val_overlap`. If either is given while the new
  `--formal_val_every` is still at its default, the legacy pair maps to
  FORMAL validation (this preserves prior notebook behaviour, which had
  update-best semantics + overlap=0.5). Prints the resolved schedule.

Deep-supervision inference
--------------------------
During sliding-window validation only the primary full-resolution head is
consumed. The model's DS heads still train in `.train()` and remain in
the checkpoint (strict=True round-trips).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
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


# ═══════════════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════════════

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
    p.add_argument("--k_att_temp_start", type=float, default=2.0)
    p.add_argument("--k_att_temp_end", type=float, default=1.0)
    p.add_argument("--k_att_anneal_epochs", type=int, default=60)

    # Segmentation / auxiliary
    p.add_argument("--attn_diversity_weight", type=float, default=0.0)

    # AdaKern / FreqSel toggles
    p.add_argument("--use_position_att", action="store_true")

    # ---- FAST / FORMAL validation -----------------------------------------
    p.add_argument("--fast_val_every", type=int, default=10,
                   help="Fast/proxy validation cadence in epochs. 0 disables.")
    p.add_argument("--fast_val_overlap", type=float, default=0.0,
                   help="Sliding-window overlap for FAST validation. Default 0.0.")
    p.add_argument("--fast_val_max_cases", type=int, default=0,
                   help="If > 0, use only the first K validation cases (deterministic subset).")
    p.add_argument("--formal_val_every", type=int, default=0,
                   help="Formal validation cadence. 0 disables formal validation during training.")
    p.add_argument("--formal_val_overlap", type=float, default=0.5,
                   help="Sliding-window overlap for FORMAL validation. Default 0.5.")
    p.add_argument("--val_sw_batch_size", type=int, default=4,
                   help="sw_batch_size for sliding-window inference (both fast and formal).")
    p.add_argument("--checkpoint_every", type=int, default=10,
                   help="Extra periodic snapshot cadence (also always saved every epoch).")

    # ---- LEGACY validation args (kept for backward compat) ----------------
    p.add_argument("--val_every", type=int, default=None,
                   help="LEGACY. If given while --formal_val_every is at default (0), "
                        "maps to --formal_val_every (formal semantics with best-update).")
    p.add_argument("--val_overlap", type=float, default=None,
                   help="LEGACY. Overlap for the legacy --val_every path. "
                        "When mapped to formal, overrides --formal_val_overlap.")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# DEEP SUPERVISION (train-time)
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# DEEP-SUPERVISION INFERENCE WRAPPER
# ═══════════════════════════════════════════════════════════════════════

def make_primary_predictor(model: nn.Module):
    """Wrap `model(x)` for sliding-window inference: return only the primary
    full-resolution segmentation head.

    - If the model returns a tuple/list (DS-mode training output), pick out[0].
    - Otherwise return the output unchanged.

    Doesn't disable DS heads or mutate the model — the wrapper just picks the
    right tensor at call time, so `strict=True` reload still works.
    """
    def _predict(x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out
    return _predict


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def _iou_sens_per_case(pred_fg: torch.Tensor, label_fg: torch.Tensor) -> tuple[float, float]:
    tp = (pred_fg * label_fg).sum().item()
    fp = (pred_fg * (1 - label_fg)).sum().item()
    fn = ((1 - pred_fg) * label_fg).sum().item()
    iou = tp / (tp + fp + fn + 1e-6)
    sens = tp / (tp + fn + 1e-6)
    return iou, sens


def limit_val_iterable(val_loader, max_cases: int):
    """Return an iterable over the first `max_cases` batches when max_cases > 0.

    `val_loader` is built with `shuffle=False` and `batch_size=1`, so iterating
    in order and taking the first K gives a deterministic subset that stays
    identical across epochs and resumes.
    """
    if max_cases and max_cases > 0:
        import itertools
        return itertools.islice(val_loader, max_cases)
    return val_loader


def validate(model, val_loader, dice_metric, post_pred, post_label,
             patch_size, device,
             overlap: float, sw_batch_size: int,
             max_cases: int = 0, label: str = "validation") -> tuple[float, float, float, int]:
    """Sliding-window validation with per-case tqdm.

    Returns (mean_dice, mean_iou, mean_sensitivity, n_cases_evaluated).
    """
    model.eval()
    dice_metric.reset()
    iou_list, sens_list = [], []

    total = None
    try:
        total = len(val_loader)
    except TypeError:
        total = None
    if max_cases and max_cases > 0:
        total = min(total or max_cases, max_cases)

    iterable = limit_val_iterable(val_loader, max_cases)

    # Wrap once — sliding-window sees only the primary head under DS.
    predictor = make_primary_predictor(model)

    pbar = tqdm(iterable, total=total, desc=label, unit="case",
                file=sys.stdout, dynamic_ncols=False, ncols=100,
                mininterval=1.0, leave=False)

    n_done = 0
    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            with autocast("cuda", enabled=device.type == "cuda"):
                preds = sliding_window_inference(
                    inputs=images, roi_size=patch_size,
                    sw_batch_size=sw_batch_size, predictor=predictor,
                    overlap=overlap,
                )
            preds_bin_list = [post_pred(p) for p in decollate_batch(preds)]
            labels_bin_list = [post_label(l.long()) for l in decollate_batch(labels)]
            dice_metric(y_pred=preds_bin_list, y=labels_bin_list)
            for pb, lb in zip(preds_bin_list, labels_bin_list):
                iou, sens = _iou_sens_per_case(pb[1].float(), lb[1].float())
                iou_list.append(iou)
                sens_list.append(sens)
                n_done += 1
    pbar.close()
    mean_dice = dice_metric.aggregate().item() if n_done else 0.0
    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    mean_sens = float(np.mean(sens_list)) if sens_list else 0.0
    dice_metric.reset()
    return mean_dice, mean_iou, mean_sens, n_done


# ═══════════════════════════════════════════════════════════════════════
# TEMPERATURE
# ═══════════════════════════════════════════════════════════════════════

def k_att_temperature(epoch: int, anneal_epochs: int, t_start: float, t_end: float) -> float:
    if anneal_epochs <= 1:
        return t_end
    e = min(max(epoch, 0), anneal_epochs - 1)
    cos_term = 0.5 * (1.0 + math.cos(math.pi * e / (anneal_epochs - 1)))
    return t_end + (t_start - t_end) * cos_term


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

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
    cfg["fadc_correct"] = {
        "use_position_att": bool(args.use_position_att),
    }
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# ARCHITECTURE-IDENTITY GUARD
# ═══════════════════════════════════════════════════════════════════════

def _arch_identity(model_name: str, model_cfg: dict, extras: dict) -> dict:
    """Small dict recording the architecture identity of a checkpoint.

    Deliberately excludes patch_size and validation settings: those do NOT
    change parameter shapes, so a strict-load can safely restore weights
    regardless.
    """
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
    ckpt_fc = ckpt_arch.get("fadc_correct", {})
    cur_fc = current.get("fadc_correct", {})
    if ckpt_fc.get("use_position_att", False) != cur_fc.get("use_position_att", False):
        raise RuntimeError(
            f"Refusing to resume: use_position_att mismatch (ckpt={ckpt_fc.get('use_position_att')} "
            f"vs current={cur_fc.get('use_position_att')})"
        )


# ═══════════════════════════════════════════════════════════════════════
# MECHANISM STATS (unchanged)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def snapshot_mechanism_stats(model: UNet3DFADCCorrect) -> dict:
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
        p = m.last_k_att.detach().float().clamp(min=1e-12)
        entropy = -(p * p.log()).sum(dim=1)
        k_ents.append(entropy.mean().item())
        arg = p.argmax(dim=1)
        n = arg.numel()
        frac_d1.append((arg == 0).float().sum().item() / n)
        frac_d2.append((arg == 1).float().sum().item() / n)
        frac_d3.append((arg == 2).float().sum().item() / n)
        if m.last_c_low is not None and m.last_c_low.size(0) > 1:
            c_low_std.append(m.last_c_low.float().std(dim=0).mean().item())
            f_low_std.append(m.last_f_low.float().std(dim=0).mean().item())
            c_high_std.append(m.last_c_high.float().std(dim=0).mean().item())
            f_high_std.append(m.last_f_high.float().std(dim=0).mean().item())

    def _stat(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "expected_dilation_mean": _stat(exp_dils),
        "k_att_entropy_mean":     _stat(k_ents),
        "frac_argmax_d1":         _stat(frac_d1),
        "frac_argmax_d2":         _stat(frac_d2),
        "frac_argmax_d3":         _stat(frac_d3),
        "c_low_batch_std":        _stat(c_low_std),
        "f_low_batch_std":        _stat(f_low_std),
        "c_high_batch_std":       _stat(c_high_std),
        "f_high_batch_std":       _stat(f_high_std),
    }


# ═══════════════════════════════════════════════════════════════════════
# ATOMIC CHECKPOINT WRITE
# ═══════════════════════════════════════════════════════════════════════

def atomic_torch_save(obj, dest: os.PathLike) -> None:
    """Write `obj` via torch.save to a temp file in the same directory, then
    os.replace() into `dest`. Never leaves a partially-written destination.
    """
    dest = str(dest)
    dest_dir = os.path.dirname(dest) or "."
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_" + os.path.basename(dest) + ".",
        suffix=".pth", dir=dest_dir,
    )
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, dest)   # atomic on POSIX; best-effort on Windows
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            raise


def atomic_json_write(obj, dest: os.PathLike) -> None:
    dest = str(dest)
    dest_dir = os.path.dirname(dest) or "."
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_" + os.path.basename(dest) + ".",
        suffix=".json", dir=dest_dir,
    )
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp_path, dest)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            raise


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION SCHEDULE RESOLUTION
# ═══════════════════════════════════════════════════════════════════════

def resolve_val_schedule(args) -> dict:
    """Resolve the fast/formal validation schedule from args, applying the
    legacy-CLI mapping. Returns a plain dict; prints the resolved schedule.

    Rule:
      1. Start from the new args (fast_val_*, formal_val_*).
      2. If `--val_every` was passed AND `--formal_val_every` is still at its
         default 0, promote the legacy pair to FORMAL:
            formal_val_every = args.val_every
            formal_val_overlap = args.val_overlap or 0.5
      3. If both legacy and new-formal are given, refuse ambiguously.
    """
    fast_every  = int(args.fast_val_every)
    fast_over   = float(args.fast_val_overlap)
    fast_max    = int(args.fast_val_max_cases)
    form_every  = int(args.formal_val_every)
    form_over   = float(args.formal_val_overlap)

    legacy_every = args.val_every
    legacy_over  = args.val_overlap
    if legacy_every is not None:
        if form_every != 0:
            raise SystemExit(
                "Both --val_every (legacy) and --formal_val_every were specified; "
                "this is ambiguous. Pass only one."
            )
        form_every = int(legacy_every)
        if legacy_over is not None:
            form_over = float(legacy_over)
        # If someone still relied on the old default overlap 0.5, form_over
        # remains at the fresh default 0.5 — matches previous behaviour.

    schedule = {
        "fast_val_every":     fast_every,
        "fast_val_overlap":   fast_over,
        "fast_val_max_cases": fast_max,
        "formal_val_every":   form_every,
        "formal_val_overlap": form_over,
        "val_sw_batch_size":  int(args.val_sw_batch_size),
        "checkpoint_every":   int(args.checkpoint_every),
    }
    print("Resolved validation schedule:")
    print(f"  FAST   every {fast_every} ep  overlap {fast_over}  "
          f"max_cases {'ALL' if fast_max == 0 else fast_max}  (proxy — updates best_fast_model.pth only)")
    print(f"  FORMAL every {form_every} ep  overlap {form_over}  "
          f"max_cases ALL  (updates best_model.pth)")
    print(f"  sw_batch_size {schedule['val_sw_batch_size']}   "
          f"checkpoint_every {schedule['checkpoint_every']}")
    if fast_every <= 0 and form_every <= 0:
        print("  NOTE: both fast and formal validation are DISABLED — training only.")
    return schedule


# ═══════════════════════════════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════════════════════════════

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

    schedule = resolve_val_schedule(args)

    split_csv = os.path.join(args.data_root, "train_test_splits.csv")
    if not os.path.exists(split_csv):
        split_csv = None
        print("WARNING: train_test_splits.csv not found — using all data for train")

    if args.smoke_test:
        print("SMOKE TEST MODE — 4 cases, 2 epochs")
        epochs = 2
        cache_rate = 0.0
        num_workers = 0
        # Force a fast val each epoch in smoke, formal off.
        schedule["fast_val_every"] = 1
        schedule["fast_val_max_cases"] = 0
        schedule["formal_val_every"] = 0
        schedule["checkpoint_every"] = 1

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
    best_dice = 0.0       # FORMAL only
    best_fast_dice = 0.0  # PROXY only
    train_log: list = []

    fadc_extras = {"use_position_att": bool(args.use_position_att)}
    arch_id = _arch_identity(args.model, cfg["model"], fadc_extras)

    # ─────────────────────────── RESUME
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        _require_matching_arch(ckpt, arch_id)
        restored = []
        model.load_state_dict(ckpt["model"], strict=True)
        restored.append("model (strict)")
        optimizer.load_state_dict(ckpt["optimizer"])
        restored.append("optimizer")
        scheduler.load_state_dict(ckpt["scheduler"])
        restored.append("scheduler")
        if "scaler" in ckpt and isinstance(ckpt["scaler"], dict):
            try:
                scaler.load_state_dict(ckpt["scaler"])
                restored.append("scaler")
            except Exception as e:
                print(f"  WARN: scaler state present but load failed ({e}); using fresh scaler.")
        else:
            restored.append("scaler=(legacy ckpt; fresh)")
        start_epoch = int(ckpt["epoch"]) + 1
        best_dice = float(ckpt.get("best_dice", 0.0))
        best_fast_dice = float(ckpt.get("best_fast_dice", 0.0))
        if "train_log" in ckpt and isinstance(ckpt["train_log"], list):
            train_log = list(ckpt["train_log"])
            restored.append(f"train_log ({len(train_log)} entries)")
        else:
            restored.append("train_log=(legacy ckpt; empty)")
        print(f"Resumed from checkpoint: {args.resume}")
        print(f"  start_epoch      = {start_epoch}")
        print(f"  best_dice(formal)= {best_dice:.4f}")
        print(f"  best_fast_dice   = {best_fast_dice:.4f}")
        print(f"  restored         : {', '.join(restored)}")

    print(f"\nStarting training: {epochs} epochs | LR {lr} | Batch {batch_size}")
    print("=" * 70)

    # ──────────────── helpers bound to this training run
    def build_ckpt_dict(epoch_completed: int) -> dict:
        return {
            "epoch":          int(epoch_completed),
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "scaler":         scaler.state_dict(),
            "best_dice":      float(best_dice),
            "best_fast_dice": float(best_fast_dice),
            "config":         cfg,
            "model_name":     args.model,
            "arch_identity":  arch_id,
            "schedule":       schedule,
            "train_log":      list(train_log),
        }

    # ──────────────── training loop
    for epoch in range(start_epoch, epochs):
        # k_att temperature — set BEFORE training for this epoch.
        t = k_att_temperature(epoch, args.k_att_anneal_epochs,
                              args.k_att_temp_start, args.k_att_temp_end)
        model.set_temperature(t)

        model.train()
        epoch_loss = epoch_dice = epoch_ce = 0.0
        num_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{epochs} T={t:.2f}",
                    leave=False, ncols=110, unit="batch", file=sys.stdout,
                    mininterval=1.0)

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
            epoch_ce   += ce_loss.item()
            num_batches += 1

        last_stats = snapshot_mechanism_stats(model)
        pbar.close()
        scheduler.step()

        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        avg_ce   = epoch_ce   / num_batches
        elapsed  = time.time() - t0
        lr_now   = scheduler.get_last_lr()[0]
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
        train_log.append(log_entry)

        # ─── PRE-VALIDATION CHECKPOINT (contains this epoch's completed training) ───
        atomic_torch_save(build_ckpt_dict(epoch_completed=epoch), output_dir / "last_checkpoint.pth")
        atomic_json_write(train_log, output_dir / "train_log.json")

        # ─── FAST (proxy) VALIDATION ───
        if schedule["fast_val_every"] > 0 and (epoch + 1) % schedule["fast_val_every"] == 0:
            print("\n---- FAST/PROXY VALIDATION - not the formal final metric ----")
            fdice, fiou, fsens, n = validate(
                model, val_loader, dice_metric, post_pred, post_label,
                patch_size, device,
                overlap=schedule["fast_val_overlap"],
                sw_batch_size=schedule["val_sw_batch_size"],
                max_cases=schedule["fast_val_max_cases"],
                label=f"FAST val (overlap={schedule['fast_val_overlap']}, n={{}})".format(
                    'ALL' if schedule['fast_val_max_cases']==0 else schedule['fast_val_max_cases']),
            )
            marker = " <-- NEW BEST FAST (proxy)" if fdice > best_fast_dice else ""
            print(f"  FAST  Dice {fdice:.4f} | IoU {fiou:.4f} | Sens {fsens:.4f} "
                  f"| n={n}  (best_fast {best_fast_dice:.4f}){marker}")
            log_entry["fast_val_dice"] = fdice
            log_entry["fast_val_iou"] = fiou
            log_entry["fast_val_sensitivity"] = fsens
            log_entry["fast_val_n_cases"] = n
            log_entry["fast_val_overlap"] = schedule["fast_val_overlap"]
            if fdice > best_fast_dice:
                best_fast_dice = fdice
                atomic_torch_save(build_ckpt_dict(epoch_completed=epoch),
                                  output_dir / "best_fast_model.pth")
                print(f"  proxy checkpoint -> {output_dir}/best_fast_model.pth")

        # ─── FORMAL VALIDATION ───
        if schedule["formal_val_every"] > 0 and (epoch + 1) % schedule["formal_val_every"] == 0:
            print("\n==== FORMAL FULL VALIDATION ====")
            fmt_dice, fmt_iou, fmt_sens, n = validate(
                model, val_loader, dice_metric, post_pred, post_label,
                patch_size, device,
                overlap=schedule["formal_val_overlap"],
                sw_batch_size=schedule["val_sw_batch_size"],
                max_cases=0,      # formal always uses all
                label=f"FORMAL val (overlap={schedule['formal_val_overlap']})",
            )
            marker = " <-- NEW BEST FORMAL" if fmt_dice > best_dice else ""
            print(f"  FORMAL Dice {fmt_dice:.4f} | IoU {fmt_iou:.4f} | Sens {fmt_sens:.4f} "
                  f"| n={n}  (best_formal {best_dice:.4f}){marker}")
            log_entry["formal_val_dice"] = fmt_dice
            log_entry["formal_val_iou"] = fmt_iou
            log_entry["formal_val_sensitivity"] = fmt_sens
            log_entry["formal_val_n_cases"] = n
            log_entry["formal_val_overlap"] = schedule["formal_val_overlap"]
            if fmt_dice > best_dice:
                best_dice = fmt_dice
                atomic_torch_save(build_ckpt_dict(epoch_completed=epoch),
                                  output_dir / "best_model.pth")
                print(f"  formal best -> {output_dir}/best_model.pth")

        # ─── POST-VALIDATION CHECKPOINT ───
        # Rewrite last_checkpoint with the validation-augmented log entry.
        atomic_torch_save(build_ckpt_dict(epoch_completed=epoch), output_dir / "last_checkpoint.pth")
        atomic_json_write(train_log, output_dir / "train_log.json")

        # Periodic named snapshot (in addition to last_checkpoint).
        if schedule["checkpoint_every"] > 0 and (epoch + 1) % schedule["checkpoint_every"] == 0:
            atomic_torch_save(build_ckpt_dict(epoch_completed=epoch),
                              output_dir / f"checkpoint_epoch{epoch+1:03d}.pth")

    # end of training
    atomic_json_write({
        "seed": args.seed,
        "model": args.model,
        "fadc_version": "correct",
        "arch_identity": arch_id,
        "k_att_temp_start": args.k_att_temp_start,
        "k_att_temp_end": args.k_att_temp_end,
        "k_att_anneal_epochs": args.k_att_anneal_epochs,
        "schedule": schedule,
        "best_dice_formal": best_dice,
        "best_fast_dice": best_fast_dice,
        "epochs": epochs,
    }, output_dir / "meta.json")

    print(f"\nTraining complete. Best FORMAL Val Dice: {best_dice:.4f}   "
          f"Best FAST/proxy Dice: {best_fast_dice:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config, args)
    train(cfg, args)
