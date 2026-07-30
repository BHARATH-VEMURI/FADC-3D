"""Training script for the CORRECTED FADC3D encoder-first experiment.

Self-contained. Does NOT modify training/train_centralized_v2.py or the plain
UNet3D training path — this is a fresh script bound to
`models/unet_3d_fadc_correct.py`.

Overview
--------
- Segmentation loss only. Attention-diversity aux is disabled by default.
- One validation protocol: FORMAL. Every --val_every epochs, runs
  sliding-window inference over ALL validation cases (306 in production) at
  --val_overlap (default 0.5). No proxy / subset / low-overlap variant.
- Only formal validation updates `best_model.pth`.
- Every checkpoint (including `last_checkpoint.pth`) is written ATOMICALLY:
  temp file in the same dir, then `os.replace()`.
- `last_checkpoint.pth` is saved BEFORE validation. If validation is
  interrupted, resume advances to the next epoch and never repeats work.
  After validation, `last_checkpoint.pth` is rewritten with the newly
  computed validation metrics folded into the log entry.
- Checkpoints carry `epoch`, `model`, `optimizer`, `scheduler`, `scaler`,
  `best_dice`, `arch_identity`, `config`, and the full `train_log`
  accumulated so far.
- Resume is backward compatible with older checkpoints that lack
  `scaler` / `train_log`, and silently ignores any legacy `best_fast_dice`
  field.

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
from models.unet_3d_fadc_continuous import (
    build_unet3d_fadc_continuous,
    MODEL_NAMES as CONTINUOUS_MODEL_NAMES,
    EXPECTED_ADAPTIVE_BLOCK_COUNT as CONTINUOUS_EXPECTED_ADAPTIVE_BLOCK_COUNT,
)
from fadc_3d_correct.adaptive_dilated_conv_3d import AdaptiveDilatedConv3D
from fadc_3d_continuous import CONTINUOUS_ADADR3D_META
from training.losses import DiceCELoss
from training.patch_validation import validate_patch_size


# ═══════════════════════════════════════════════════════════════════════
# MODEL-KIND DISPATCH
#
# The trainer natively handles both the discrete package
# (`models/unet_3d_fadc_correct.py`) and the continuous package
# (`models/unet_3d_fadc_continuous.py`). Checkpoints carry `arch_kind`
# so a strict resume cannot cross the two implementations.
# ═══════════════════════════════════════════════════════════════════════

ALL_MODEL_NAMES = tuple(list(MODEL_NAMES) + list(CONTINUOUS_MODEL_NAMES))


def _model_kind(model_name: str) -> str:
    if model_name in CONTINUOUS_MODEL_NAMES:
        return "continuous"
    if model_name in MODEL_NAMES:
        return "discrete"
    raise ValueError(f"Unknown model {model_name!r}; expected one of {ALL_MODEL_NAMES}")


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
    p.add_argument("--split_manifest", type=str, default=None,
                   help="Path to a CSV manifest produced by training/split_manifest.py. "
                        "When set, patient partitioning comes from the manifest and the "
                        "old preprocessed_cache_dir/{train,val} subdir enumeration is bypassed. "
                        "--preprocessed_cache_dir is still required as the base for "
                        "relative_npz_path resolution.")
    p.add_argument("--split_partition_train", type=str, default="train",
                   help="Which manifest partition supplies the training loader. Default 'train'.")
    p.add_argument("--split_partition_val", type=str, default="val",
                   help="Which manifest partition supplies the validation loader. Default 'val'.")
    p.add_argument("--smoke_test", action="store_true",
                   help="2 epochs on 4 cases — verifies the end-to-end pipeline.")
    p.add_argument("--model", type=str, default="unet3d_fadc_encoder_correct",
                   choices=list(ALL_MODEL_NAMES))
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

    # ---- FORMAL validation (single protocol) ------------------------------
    p.add_argument("--val_every", type=int, default=20,
                   help="Formal validation cadence in epochs. Uses ALL "
                        "validation cases at --val_overlap.")
    p.add_argument("--val_overlap", type=float, default=0.5,
                   help="Sliding-window overlap for formal validation. Default 0.5.")
    p.add_argument("--val_sw_batch_size", type=int, default=4,
                   help="sw_batch_size for sliding-window inference.")
    p.add_argument("--checkpoint_every", type=int, default=10,
                   help="Extra periodic named-snapshot cadence "
                        "(last_checkpoint.pth is always saved every epoch regardless).")

    # ---- Gradient accumulation ------------------------------------------
    # Default 1 keeps every historical experiment (including all discrete
    # notebooks) bit-for-bit identical: a single microbatch per optimizer
    # step. Values >1 divide the loss by grad_accum_steps for backward,
    # accumulate gradients across grad_accum_steps microbatches, then take
    # one optimizer step. Effective batch = --batch_size * --grad_accum_steps.
    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help="Number of microbatches whose gradients accumulate "
                        "before one optimizer step. Default 1 preserves the "
                        "legacy per-microbatch step behavior.")

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


def validate(model, val_loader, dice_metric, post_pred, post_label,
             patch_size, device,
             overlap: float, sw_batch_size: int) -> tuple[float, float, float, int]:
    """Sliding-window formal validation over the ENTIRE val loader.

    Iterates every case in `val_loader` — no subset, no proxy. Returns
    (mean_dice, mean_iou, mean_sensitivity, n_cases_evaluated).
    """
    model.eval()
    dice_metric.reset()
    iou_list, sens_list = [], []

    try:
        total = len(val_loader)
    except TypeError:
        total = None

    predictor = make_primary_predictor(model)

    pbar = tqdm(val_loader, total=total,
                desc="Formal validation", unit="case",
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

def _arch_identity(model_name: str, model_cfg: dict, extras: dict,
                   arch_kind: str = "discrete") -> dict:
    """Small dict recording the architecture identity of a checkpoint.

    Deliberately excludes patch_size and validation settings: those do NOT
    change parameter shapes, so a strict-load can safely restore weights
    regardless.

    `arch_kind` is the discrete-vs-continuous discriminator. Ckpts written
    before this field existed are all discrete, so a missing value on the
    ckpt side is treated as "discrete" for backward-compat.
    """
    ident = {
        "model_name": model_name,
        "in_channels": int(model_cfg["in_channels"]),
        "out_channels": int(model_cfg["out_channels"]),
        "base_filters": int(model_cfg["base_filters"]),
        "deep_supervision": bool(model_cfg.get("deep_supervision", False)),
        "arch_kind": str(arch_kind),
    }
    if arch_kind == "continuous":
        ident["fadc_continuous"] = {k: v for k, v in CONTINUOUS_ADADR3D_META.items()}
        ident["fadc_correct"] = {}
    else:
        ident["fadc_correct"] = dict(extras)
    return ident


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
    # arch_kind: discrete vs continuous. Legacy ckpts predate this field —
    # they are all discrete, so a missing value is normalised.
    ckpt_kind = ckpt_arch.get("arch_kind", "discrete")
    cur_kind = current.get("arch_kind", "discrete")
    if ckpt_kind != cur_kind:
        raise RuntimeError(
            f"Refusing to resume: arch_kind mismatch (ckpt={ckpt_kind!r} "
            f"vs current={cur_kind!r}). Discrete and continuous FADC3D "
            f"share model names but not parameter shapes."
        )
    # Only check use_position_att for the discrete arch — continuous has no
    # such knob.
    if cur_kind == "discrete":
        ckpt_fc = ckpt_arch.get("fadc_correct", {}) or {}
        cur_fc = current.get("fadc_correct", {}) or {}
        if ckpt_fc.get("use_position_att", False) != cur_fc.get("use_position_att", False):
            raise RuntimeError(
                f"Refusing to resume: use_position_att mismatch (ckpt={ckpt_fc.get('use_position_att')} "
                f"vs current={cur_fc.get('use_position_att')})"
            )


def _require_matching_train(ckpt: dict, current: dict) -> None:
    """Refuse to resume across a change in the training-run identity.

    A checkpoint carries the (physical_batch_size, grad_accum_steps,
    effective_batch_size) triple under 'train_identity'. Resuming under
    different values changes the optimisation trajectory, so we hard-fail
    rather than silently continue. Legacy checkpoints (predating this
    field) are treated as physical_batch_size=<their cfg batch> with
    grad_accum_steps=1 so backward-compat is preserved.
    """
    ckpt_train = ckpt.get("train_identity")
    if ckpt_train is None:
        # Legacy path: infer from cfg.training.batch_size if available;
        # accumulation is always 1 on legacy since the flag did not exist.
        legacy_bs = int(
            (ckpt.get("config") or {}).get("training", {}).get("batch_size", 0)
        ) or None
        ckpt_train = {
            "physical_batch_size":  legacy_bs,
            "grad_accum_steps":     1,
            "effective_batch_size": legacy_bs,
        }
    for key in ("physical_batch_size", "grad_accum_steps", "effective_batch_size"):
        cv = ckpt_train.get(key)
        nv = (current or {}).get(key)
        # Skip if either side is None (legacy or a run that didn't record it).
        if cv is None or nv is None:
            continue
        if int(cv) != int(nv):
            raise RuntimeError(
                f"Refusing to resume: train identity mismatch on '{key}': "
                f"ckpt={cv}  current={nv}"
            )


def _require_matching_split(ckpt: dict, current: dict) -> None:
    """Refuse to resume across a manifest change.

    Compares (split_manifest_sha256, split_partition_train, split_partition_val,
    split_seed, split_kind) on the checkpoint against the current run. A
    checkpoint that trained on no manifest cannot be resumed under a manifest,
    and vice versa. Purely additive — legacy resumes that predate the manifest
    field skip silently because both sides carry None.

    `split_kind` is the discriminator between the historical seed-42 70/10/20
    manifests (kind='seed_split' or None on legacy) and the default-cache
    snapshot (kind='default_cache'). A 70/10/20 checkpoint cannot be
    resumed under a default-cache run — the training set differs by ~150
    patients and validation Dice would be measured on a leaked-in cohort.
    """
    ckpt_split = ckpt.get("split_identity", {}) or {}
    cur_split = current or {}
    # If both are wholly None (legacy on both sides), let resume proceed.
    if not any(ckpt_split.values()) and not any(cur_split.values()):
        return
    for key in ("split_manifest_sha256", "split_partition_train",
                "split_partition_val", "split_seed", "split_kind"):
        if ckpt_split.get(key) != cur_split.get(key):
            raise RuntimeError(
                f"Refusing to resume: split identity mismatch on '{key}': "
                f"ckpt={ckpt_split.get(key)}  current={cur_split.get(key)}"
            )


# ═══════════════════════════════════════════════════════════════════════
# MECHANISM STATS — kind-aware
#
# Discrete FADC has voxelwise k_att over three dilation branches (D=1,2,3)
# and channel/filter low/high attention batch-stds. Continuous FADC has
# per-voxel effective dilation D(p) = 1 + s(p) and a per-position mask
# m(p, q) for q in {-1,0,1}^3. Different mechanisms, different summaries.
# The trainer dispatches on model kind and never prints NaNs for the
# other side's stats.
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


@torch.no_grad()
def snapshot_continuous_mechanism_stats(model: nn.Module) -> dict:
    """Continuous-FADC mechanism snapshot.

    Iterates every `ContinuousDilatedConv3D` in the model and folds their
    per-voxel `last_effective_dilation` and per-voxel per-position `last_mask`
    caches into scalar summaries. Values are averaged across modules so a
    single dict describes the whole encoder.

    Returned keys (all floats):
        eff_dilation_mean   : mean of (1 + s(p)) across every voxel/module
        eff_dilation_std    : std across voxels within a module, averaged over modules
        eff_dilation_min    : min effective dilation observed
        eff_dilation_max    : max effective dilation observed
        mask_mean           : mean of sigmoid mask values in [0, 1]
        mask_std            : std of mask values (voxelwise across positions)
        mask_frac_near_half : fraction of mask values within |m - 0.5| < 0.05
                              (unmoved-from-init proxy)
    """
    from fadc_3d_continuous.continuous_dilated_conv_3d import ContinuousDilatedConv3D

    means, stds, mins, maxs = [], [], [], []
    m_means, m_stds, m_frac05 = [], [], []
    for module in model.modules():
        if not isinstance(module, ContinuousDilatedConv3D):
            continue
        ed = module.adadr.last_effective_dilation
        mk = module.adadr.last_mask
        if ed is None or mk is None:
            continue
        ed_f = ed.detach().float()
        means.append(ed_f.mean().item())
        # std per module — measured within the (B, D, H, W) tensor for that module.
        stds.append(ed_f.std().item() if ed_f.numel() > 1 else 0.0)
        mins.append(ed_f.min().item())
        maxs.append(ed_f.max().item())

        mk_f = mk.detach().float()
        m_means.append(mk_f.mean().item())
        m_stds.append(mk_f.std().item() if mk_f.numel() > 1 else 0.0)
        near_half = ((mk_f - 0.5).abs() < 0.05).float().mean().item()
        m_frac05.append(near_half)

    def _stat(xs, agg=float("nan")):
        return float(np.mean(xs)) if xs else float(agg)

    return {
        "eff_dilation_mean":   _stat(means),
        "eff_dilation_std":    _stat(stds),
        "eff_dilation_min":    _stat(mins),
        "eff_dilation_max":    _stat(maxs),
        "mask_mean":           _stat(m_means),
        "mask_std":            _stat(m_stds),
        "mask_frac_near_half": _stat(m_frac05),
    }


def _format_mech_line(kind: str, stats: dict) -> str:
    """Compact one-line mechanism summary for the per-epoch printout.

    Discrete: E[dil]=<...>  (matches historical format).
    Continuous: <D>=<...> mask_mu=<...> — no E[dil]=NaN clutter.
    Returns an empty string if the stats dict is empty (all NaN)."""
    if kind == "continuous":
        m = stats.get("eff_dilation_mean")
        s = stats.get("eff_dilation_std")
        mm = stats.get("mask_mean")
        if m is None or (isinstance(m, float) and (m != m)):   # NaN check
            return ""
        return (f"<D>={m:.3f}±{s:.3f} "
                f"[{stats['eff_dilation_min']:.2f},{stats['eff_dilation_max']:.2f}] "
                f"mask_mu={mm:.3f}")
    else:
        m = stats.get("expected_dilation_mean")
        if m is None or (isinstance(m, float) and (m != m)):
            return ""
        return f"E[dil]={m:.3f}"


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
# TRAIN
# ═══════════════════════════════════════════════════════════════════════

def _validate_grad_accum_steps(raw) -> int:
    """Fail-fast validator for --grad_accum_steps. Called at the very top
    of train() so an invalid value cannot silently pass loader construction
    and only surface deep inside the training loop.
    """
    if raw is None:
        raw = 1
    try:
        v = int(raw)
    except (TypeError, ValueError):
        raise SystemExit(
            f"--grad_accum_steps must be an integer >= 1 (got {raw!r})"
        )
    if v < 1:
        raise SystemExit(
            f"--grad_accum_steps must be >= 1 (got {v}). A zero or negative "
            "value would produce a division-by-zero on the loss/step or a "
            "negative-magnitude update; refusing to run."
        )
    return v


def train(cfg, args):
    # Fail-fast on --grad_accum_steps BEFORE any I/O (loader construction,
    # dataset scan, model build). Repeat the validation later when the
    # local variable is initialised — this early call is purely a guard.
    _validate_grad_accum_steps(getattr(args, "grad_accum_steps", 1))

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

    patch_size = validate_patch_size(cfg["data"]["patch_size"], name="patch_size")
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["lr"]
    batch_size = cfg["training"]["batch_size"]
    cache_rate = cfg["data"]["cache_rate"]
    num_workers = cfg["data"]["num_workers"]

    val_config = {
        "val_every":         int(args.val_every),
        "val_overlap":       float(args.val_overlap),
        "val_sw_batch_size": int(args.val_sw_batch_size),
        "checkpoint_every":  int(args.checkpoint_every),
    }
    print("Validation schedule (formal only):")
    print(f"  every {val_config['val_every']} ep  overlap {val_config['val_overlap']}  "
          f"all cases  sw_batch_size {val_config['val_sw_batch_size']}  "
          f"(updates best_model.pth)")
    print(f"  checkpoint_every {val_config['checkpoint_every']} ep "
          f"(last_checkpoint.pth is written every epoch regardless)")
    if val_config['val_every'] <= 0:
        print("  NOTE: validation is DISABLED — training only.")

    split_csv = os.path.join(args.data_root, "train_test_splits.csv")
    if not os.path.exists(split_csv):
        split_csv = None
        print("WARNING: train_test_splits.csv not found — using all data for train")

    if args.smoke_test:
        print("SMOKE TEST MODE — 4 cases, 2 epochs")
        epochs = 2
        cache_rate = 0.0
        num_workers = 0
        # Force formal validation each epoch in smoke.
        val_config["val_every"] = 1
        val_config["checkpoint_every"] = 1

    # Manifest short-circuits the legacy split_csv path. Refuse to fall
    # back to train_test_splits.csv when a manifest is explicitly requested.
    split_manifest_arg = args.split_manifest or ""
    if split_manifest_arg:
        if not os.path.exists(split_manifest_arg):
            raise SystemExit(f"--split_manifest not found: {split_manifest_arg}")
        split_csv = None  # never mix manifest + legacy split_csv

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
        split_manifest=split_manifest_arg,
        split_train=args.split_partition_train,
        split_val=args.split_partition_val,
    )

    model_kwargs = dict(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_filters=cfg["model"]["base_filters"],
    )
    kind = _model_kind(args.model)
    if kind == "continuous":
        if args.deep_supervision:
            raise SystemExit(
                "--deep_supervision is incompatible with the continuous FADC3D "
                "encoder (contract: DS off — see models/unet_3d_fadc_continuous.py)."
            )
        model = build_unet3d_fadc_continuous(
            model_name=args.model,
            **model_kwargs,
            deep_supervision=False,
        ).to(device)
        n_adapt = model.count_adaptive_blocks()
        expected = CONTINUOUS_EXPECTED_ADAPTIVE_BLOCK_COUNT["encoder"]
        if n_adapt != expected:
            raise RuntimeError(
                f"Model {args.model} has {n_adapt} continuous adaptive blocks, expected {expected}."
            )
        print(f"Model: {args.model}  (continuous encoder)  adaptive_blocks={n_adapt}/{expected}")
    else:
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
    _t_msg = (f"k_att temperature schedule: {args.k_att_temp_start} -> "
              f"{args.k_att_temp_end} over {args.k_att_anneal_epochs} epochs")
    if kind == "continuous":
        _t_msg += "  (inert — continuous model has no k_att temperature)"
    print(_t_msg)

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
    train_log: list = []

    fadc_extras = {"use_position_att": bool(args.use_position_att)}
    arch_id = _arch_identity(args.model, cfg["model"], fadc_extras, arch_kind=kind)

    # Training-run identity. Legacy defaults keep every prior discrete run's
    # ckpt loadable: --grad_accum_steps defaults to 1, so physical ==
    # effective. Strict validation happens once at the top of train() via
    # _validate_grad_accum_steps; this re-call is a single source of truth
    # for the local variable.
    grad_accum_steps = _validate_grad_accum_steps(
        getattr(args, "grad_accum_steps", 1)
    )
    physical_batch_size = int(batch_size)
    effective_batch_size = physical_batch_size * grad_accum_steps
    train_id = {
        "physical_batch_size":  physical_batch_size,
        "grad_accum_steps":     grad_accum_steps,
        "effective_batch_size": effective_batch_size,
    }
    # Mirror the same fields into cfg so any downstream serialisation
    # (train_log, formal_val json) sees them too.
    cfg["training"]["physical_batch_size"]  = physical_batch_size
    cfg["training"]["grad_accum_steps"]     = grad_accum_steps
    cfg["training"]["effective_batch_size"] = effective_batch_size
    print(f"batch identity : physical={physical_batch_size} "
          f"accum={grad_accum_steps} effective={effective_batch_size}")

    # Manifest identity: burn the CSV SHA256, split partitions and (if we
    # can derive it) ratios/seed/kind into every checkpoint. Downstream
    # resumes and final-test evaluators refuse mismatched manifests. All
    # fields None-safe when no manifest is used, so legacy training paths
    # are unaffected.
    split_identity: dict = {
        "split_manifest_path":   os.path.abspath(split_manifest_arg) if split_manifest_arg else None,
        "split_manifest_sha256": None,
        "split_partition_train": args.split_partition_train if split_manifest_arg else None,
        "split_partition_val":   args.split_partition_val if split_manifest_arg else None,
        "split_seed":            None,
        "split_ratios":          None,
        "split_kind":            None,   # 'default_cache' | 'seed_split' | None (legacy)
    }
    if split_manifest_arg:
        from training.split_manifest import manifest_sha256
        split_identity["split_manifest_sha256"] = manifest_sha256(split_manifest_arg)
        # Best-effort read of the companion meta.json (same basename with
        # _metadata.json suffix); tolerate absence.
        _meta_guess = os.path.splitext(split_manifest_arg)[0] + "_metadata.json"
        if os.path.exists(_meta_guess):
            try:
                with open(_meta_guess, "r", encoding="utf-8") as _f:
                    _meta = json.load(_f)
                split_identity["split_seed"]   = _meta.get("seed")
                split_identity["split_ratios"] = _meta.get("ratios")
                # split_kind is present on default-cache snapshots and any
                # future manifest dialects. Historical seed-42 metadata files
                # do not include it — normalise those to 'seed_split' so a
                # default-cache resume vs seed-split resume is rejected by
                # _require_matching_split.
                split_identity["split_kind"] = _meta.get("split_kind") or "seed_split"
            except Exception:
                pass  # metadata is a convenience; the checksum is authoritative.
    print(f"split_identity : {split_identity}")

    # ─────────────────────────── RESUME
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        _require_matching_arch(ckpt, arch_id)
        _require_matching_split(ckpt, split_identity)
        _require_matching_train(ckpt, train_id)
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
        completed_epoch = int(ckpt["epoch"])
        start_epoch = completed_epoch + 1
        best_dice = float(ckpt.get("best_dice", 0.0))
        if "train_log" in ckpt and isinstance(ckpt["train_log"], list):
            train_log = list(ckpt["train_log"])
            restored.append(f"train_log ({len(train_log)} entries)")
        else:
            restored.append("train_log=(legacy ckpt; empty)")
        if "best_fast_dice" in ckpt:
            # Legacy field from the deprecated fast-validation system — ignored.
            restored.append("best_fast_dice=(legacy; ignored)")

        # Temperature the next epoch will train at (mirrors the top of the loop).
        t_next = k_att_temperature(start_epoch, args.k_att_anneal_epochs,
                                   args.k_att_temp_start, args.k_att_temp_end)
        lr_now = scheduler.get_last_lr()[0]
        print(f"Resumed from checkpoint: {args.resume}")
        print(f"  completed epoch  = {completed_epoch}")
        print(f"  next epoch       = {start_epoch}")
        print(f"  best_dice(formal)= {best_dice:.4f}")
        print(f"  scheduler LR now = {lr_now:.2e}")
        print(f"  k_att T at next  = {t_next:.4f}")
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
            "config":         cfg,
            "model_name":     args.model,
            "arch_identity":  arch_id,
            "split_identity": dict(split_identity),
            "train_identity": dict(train_id),
            "val_config":     val_config,
            "train_log":      list(train_log),
        }

    # ──────────────── training loop
    for epoch in range(start_epoch, epochs):
        # k_att temperature — set BEFORE training for this epoch.
        # Continuous model has no k_att branch; the schedule is logged but
        # not applied.
        t = k_att_temperature(epoch, args.k_att_anneal_epochs,
                              args.k_att_temp_start, args.k_att_temp_end)
        if hasattr(model, "set_temperature"):
            model.set_temperature(t)

        model.train()
        epoch_loss = epoch_dice = epoch_ce = 0.0
        num_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{epochs} T={t:.2f}",
                    leave=False, ncols=110, unit="batch", file=sys.stdout,
                    mininterval=1.0)

        # Gradient accumulation state — reset at the start of each epoch.
        # grad_accum_steps == 1 collapses this to the historical single-step
        # path: zero_grad -> forward -> backward -> step -> zero_grad.
        optimizer.zero_grad(set_to_none=True)
        microbatches_in_group = 0
        optimizer_steps_this_epoch = 0

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

            with autocast("cuda", enabled=device.type == "cuda"):
                preds = model(images)
                if isinstance(preds, tuple):
                    total_loss, dice_loss, ce_loss = deep_supervision_loss(preds, labels, criterion)
                else:
                    total_loss, dice_loss, ce_loss = criterion(preds, labels)

            # Divide ONLY for backward — the gradient accumulated across
            # grad_accum_steps microbatches averages out to what a single
            # bigger-batch backward would have produced. Logging below uses
            # the UNSCALED loss so per-batch metrics stay comparable across
            # runs with different accumulation values.
            loss_for_backward = total_loss / grad_accum_steps if grad_accum_steps > 1 else total_loss
            scaler.scale(loss_for_backward).backward()
            microbatches_in_group += 1

            # Optimizer step only at an accumulation boundary.
            if microbatches_in_group == grad_accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                microbatches_in_group = 0
                optimizer_steps_this_epoch += 1

            epoch_loss += total_loss.item()
            epoch_dice += dice_loss.item()
            epoch_ce   += ce_loss.item()
            num_batches += 1

        # Final partial accumulation group — the loader emitted a count of
        # microbatches that isn't a multiple of grad_accum_steps. We divided
        # each microbatch's loss by grad_accum_steps for backward, so the
        # accumulated gradients over m < grad_accum_steps microbatches have
        # magnitude (m / grad_accum_steps) * mean_grad instead of the
        # mean_grad we want. Multiply the unscaled grads by
        # (grad_accum_steps / m) BEFORE clipping so the resulting step is
        # numerically equivalent to averaging exactly the m microbatches
        # actually present. The order matters: unscale first (undoes AMP
        # scaling), then rescale (undoes the /grad_accum_steps bias for a
        # partial group), then clip, then step + update + zero_grad.
        if microbatches_in_group > 0:
            scaler.unscale_(optimizer)
            if grad_accum_steps > 1 and microbatches_in_group < grad_accum_steps:
                correction = grad_accum_steps / microbatches_in_group
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            microbatches_in_group = 0
            optimizer_steps_this_epoch += 1

        if kind == "continuous":
            last_stats = snapshot_continuous_mechanism_stats(model)
        else:
            last_stats = snapshot_mechanism_stats(model)
        pbar.close()
        scheduler.step()

        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        avg_ce   = epoch_ce   / num_batches
        elapsed  = time.time() - t0
        lr_now   = scheduler.get_last_lr()[0]
        mins, secs = divmod(int(elapsed), 60)
        mech_line = _format_mech_line(kind, last_stats)
        # Continuous has no k_att temperature — drop the 'T=' segment in
        # the epoch line to avoid a meaningless value.
        _t_seg = f" T={t:.2f} |" if kind != "continuous" else ""
        _mech_seg = f" {mech_line} |" if mech_line else ""
        print(f"Epoch {epoch+1:03d}/{epochs} |{_t_seg} "
              f"Loss {avg_loss:.4f} | Dice {avg_dice:.4f} | CE {avg_ce:.4f} | "
              f"LR {lr_now:.2e} |{_mech_seg} {mins}m{secs}s")

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

        # ─── FORMAL VALIDATION ───
        if val_config["val_every"] > 0 and (epoch + 1) % val_config["val_every"] == 0:
            try:
                n_val = len(val_loader)
            except TypeError:
                n_val = "?"
            print(f"\n==== FORMAL FULL VALIDATION: {n_val} cases, "
                  f"overlap={val_config['val_overlap']} ====")
            v_dice, v_iou, v_sens, n = validate(
                model, val_loader, dice_metric, post_pred, post_label,
                patch_size, device,
                overlap=val_config["val_overlap"],
                sw_batch_size=val_config["val_sw_batch_size"],
            )
            marker = " <-- NEW BEST" if v_dice > best_dice else ""
            print(f"  FORMAL Dice {v_dice:.4f} | IoU {v_iou:.4f} | Sens {v_sens:.4f} "
                  f"| n={n}  (best {best_dice:.4f}){marker}")
            log_entry["val_dice"] = v_dice
            log_entry["val_iou"] = v_iou
            log_entry["val_sensitivity"] = v_sens
            log_entry["val_n_cases"] = n
            log_entry["val_overlap"] = val_config["val_overlap"]
            if v_dice > best_dice:
                best_dice = v_dice
                atomic_torch_save(build_ckpt_dict(epoch_completed=epoch),
                                  output_dir / "best_model.pth")
                print(f"  best -> {output_dir}/best_model.pth")

        # ─── POST-VALIDATION CHECKPOINT ───
        # Rewrite last_checkpoint with the validation-augmented log entry.
        atomic_torch_save(build_ckpt_dict(epoch_completed=epoch), output_dir / "last_checkpoint.pth")
        atomic_json_write(train_log, output_dir / "train_log.json")

        # Periodic named snapshot (in addition to last_checkpoint).
        if val_config["checkpoint_every"] > 0 and (epoch + 1) % val_config["checkpoint_every"] == 0:
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
        "val_config": val_config,
        "best_dice": best_dice,
        "epochs": epochs,
    }, output_dir / "meta.json")

    print(f"\nTraining complete. Best FORMAL Val Dice: {best_dice:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config, args)
    train(cfg, args)
