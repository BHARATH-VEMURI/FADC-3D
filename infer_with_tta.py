"""
Test-time augmentation + threshold sweep inference for FADC variants.

Stack:
  - Sliding-window inference at overlap=0.5 (default) with Gaussian patch fusion
  - 8-flip test-time augmentation (averages softmax probabilities across flips)
  - Threshold sweep over a configurable list of probability cutoffs
  - Reports per-threshold mean Dice / IoU / Sensitivity over the val set
  - Picks the threshold that maximizes mean Dice and saves a JSON summary

Typical lift vs the training-time validation pass (overlap=0.0, no TTA, threshold=0.5):
  overlap=0.5         : +0.005 to +0.015 Dice
  8-flip TTA          : +0.005 to +0.010 Dice
  threshold sweep     : +0.005 to +0.010 Dice
Stacked, realistic: 0.6851 -> 0.70-0.715 on the FADC-Bottleneck headline.

Usage (local or Kaggle):
    python infer_with_tta.py \
        --ckpt   /path/to/best_model.pth \
        --model  unet3d_fadc_bottleneck \
        --data_root /path/to/dataset_root_with_split_csv \
        --preprocessed_cache_dir /path/to/2ch_npz_cache \
        --output_dir /path/to/results
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from monai.inferers import sliding_window_inference

sys.path.append(str(Path(__file__).parent))
from data.mama_mia_dataset import build_centralized_loaders
from models.unet_3d import UNet3D
from models.unet_3d_fadc import UNet3DFADC


# ─────────────────────────────────────────────
# 1. MODEL BUILD
# ─────────────────────────────────────────────

_FADC_PLACEMENT = {
    "unet3d_fadc":              "full",
    "unet3d_fadc_encoder":      "encoder",
    "unet3d_fadc_bottleneck":   "bottleneck",
    "unet3d_fadc_mid":          "mid",
    "unet3d_fadc_deep":         "deep",
    "unet3d_fadc_deep_lite":    "deep_lite",
}


def build_model(model_name: str, cfg: dict, device: torch.device):
    kwargs = dict(
        in_channels  = cfg["model"]["in_channels"],
        out_channels = cfg["model"]["out_channels"],
        base_filters = cfg["model"]["base_filters"],
    )
    ds_flag = cfg["model"].get("deep_supervision", False)
    if model_name in _FADC_PLACEMENT:
        model = UNet3DFADC(**kwargs,
                           fadc_placement=_FADC_PLACEMENT[model_name],
                           deep_supervision=ds_flag).to(device)
    elif model_name == "unet3d":
        model = UNet3D(**kwargs).to(device)
    else:
        raise ValueError(f"Unknown --model: {model_name}")
    model.eval()
    return model


# ─────────────────────────────────────────────
# 2. TTA — 8 FLIP COMBINATIONS
# ─────────────────────────────────────────────

# Spatial axes for a (B, C, D, H, W) tensor are 2/3/4.
# 2^3 = 8 combinations including identity.
_FLIP_AXES_LIST = [
    (),
    (2,), (3,), (4,),
    (2, 3), (2, 4), (3, 4),
    (2, 3, 4),
]


def tta_predict(model, image, patch_size, sw_batch_size, overlap, device, use_tta=True):
    """Run sliding-window inference, optionally averaged across 8 flips.

    Returns averaged softmax probabilities, shape (B, C, D, H, W).
    """
    axes_list = _FLIP_AXES_LIST if use_tta else [()]
    accum = None
    for axes in axes_list:
        x = torch.flip(image, dims=list(axes)) if axes else image
        with autocast("cuda", enabled=device.type == "cuda"):
            logits = sliding_window_inference(
                inputs=x,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
            )
        prob = torch.softmax(logits, dim=1).float()
        if axes:
            prob = torch.flip(prob, dims=list(axes))
        accum = prob if accum is None else accum + prob
    return accum / len(axes_list)


# ─────────────────────────────────────────────
# 3. PER-CASE METRICS (THRESHOLD SWEEP)
# ─────────────────────────────────────────────

def case_metrics(prob_fg: torch.Tensor, label_fg: torch.Tensor, thresholds, smooth: float = 1e-6):
    """For one case, return {threshold: (dice, iou, sens)}.

    prob_fg:  (D, H, W) float tumor-class probability
    label_fg: (D, H, W) float binary ground truth
    """
    out = {}
    label_flat = label_fg.flatten()
    label_sum  = label_flat.sum().item()
    for t in thresholds:
        pred = (prob_fg >= t).float().flatten()
        tp   = (pred * label_flat).sum().item()
        fp   = (pred * (1.0 - label_flat)).sum().item()
        fn   = ((1.0 - pred) * label_flat).sum().item()
        dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
        iou  = (tp + smooth) / (tp + fp + fn + smooth)
        sens = (tp + smooth) / (tp + fn + smooth) if (tp + fn) > 0 else float("nan")
        out[float(t)] = (dice, iou, sens)
    return out


# ─────────────────────────────────────────────
# 4. ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True, help="Path to best_model.pth")
    p.add_argument("--model",      required=True,
                   choices=["unet3d"] + list(_FADC_PLACEMENT.keys()),
                   help="Architecture (must match the checkpoint)")
    p.add_argument("--data_root",              required=True)
    p.add_argument("--preprocessed_cache_dir", required=True)
    p.add_argument("--output_dir",             required=True)
    p.add_argument("--overlap",       type=float, default=0.5)
    p.add_argument("--sw_batch_size", type=int,   default=4)
    p.add_argument("--thresholds",    type=float, nargs="+",
                   default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    p.add_argument("--no_tta",     action="store_true",
                   help="Disable 8-flip TTA (overlap=0.5 + threshold sweep only)")
    p.add_argument("--limit",      type=int, default=None,
                   help="Run on the first N val cases only (quick smoke test)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device           : {device}")
    if device.type != "cuda":
        print("WARNING: running on CPU. TTA inference on 306 cases will take many hours.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ──
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg  = ckpt["config"]
    patch_size = tuple(cfg["data"]["patch_size"])
    print(f"Checkpoint       : {args.ckpt}")
    print(f"  best_dice (train val): {ckpt.get('best_dice', '?')}")
    print(f"  epoch                : {ckpt.get('epoch', '?')}")
    print(f"Patch size       : {patch_size}")
    print(f"Model            : {args.model}")
    print(f"Overlap          : {args.overlap}")
    print(f"TTA              : {'OFF (--no_tta)' if args.no_tta else 'ON (8 flips)'}")
    print(f"Thresholds       : {args.thresholds}")

    model = build_model(args.model, cfg, device)
    model.load_state_dict(ckpt["model"])

    # ── Build val loader (cache-mode, batch=1) ──
    split_csv = os.path.join(args.data_root, "train_test_splits.csv")
    if not os.path.exists(split_csv):
        split_csv = None
        print("WARNING: train_test_splits.csv not found at data_root — using all cases")

    _, val_loader = build_centralized_loaders(
        data_root              = args.data_root,
        split_csv              = split_csv,
        cache_rate             = 0.0,
        num_workers            = 0,
        batch_size             = 1,
        preprocessed_cache_dir = args.preprocessed_cache_dir,
        patch_size             = patch_size,
    )

    # ── Inference loop ──
    thresholds = [float(t) for t in args.thresholds]
    results = {t: {"dice": [], "iou": [], "sens": []} for t in thresholds}

    n_cases = 0
    t_start = time.time()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if args.limit is not None and i >= args.limit:
                break
            images = batch["image"].to(device)
            labels = batch["label"].to(device)  # (1, 1, D, H, W)

            t0 = time.time()
            prob = tta_predict(
                model, images, patch_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.overlap,
                device=device,
                use_tta=not args.no_tta,
            )
            prob_fg  = prob[0, 1]            # (D, H, W) tumor probability
            label_fg = labels[0, 0].float()  # (D, H, W) binary mask

            metrics = case_metrics(prob_fg, label_fg, thresholds)
            for t, (dice, iou, sens) in metrics.items():
                results[t]["dice"].append(dice)
                results[t]["iou"].append(iou)
                if not (sens != sens):  # filter NaN
                    results[t]["sens"].append(sens)
            n_cases += 1

            elapsed = time.time() - t0
            print(f"  [{i+1:03d}] dice@0.50 = {metrics[0.50][0]:.4f}  "
                  f"({elapsed:.1f}s)", flush=True)

    total_time = time.time() - t_start

    # ── Aggregate ──
    summary = {}
    for t in thresholds:
        summary[t] = {
            "dice_mean": float(np.mean(results[t]["dice"])) if results[t]["dice"] else float("nan"),
            "iou_mean":  float(np.mean(results[t]["iou"]))  if results[t]["iou"]  else float("nan"),
            "sens_mean": float(np.mean(results[t]["sens"])) if results[t]["sens"] else float("nan"),
            "n_cases":   len(results[t]["dice"]),
        }
    best_t = max(summary.keys(), key=lambda t: summary[t]["dice_mean"])

    # ── Print table ──
    print("\n" + "=" * 72)
    print(f"FINAL RESULTS — {args.model}  (overlap={args.overlap}, "
          f"TTA={'on' if not args.no_tta else 'off'}, "
          f"n_cases={n_cases}, total_time={total_time/60:.1f} min)")
    print("=" * 72)
    print(f"{'Threshold':<12}{'Dice':<12}{'IoU':<12}{'Sensitivity':<14}{'':<8}")
    print("-" * 72)
    for t in sorted(summary.keys()):
        s = summary[t]
        marker = " <-- BEST" if t == best_t else ""
        print(f"{t:<12.2f}{s['dice_mean']:<12.4f}{s['iou_mean']:<12.4f}"
              f"{s['sens_mean']:<14.4f}{marker}")
    print("=" * 72)
    print(f"Best Dice  : {summary[best_t]['dice_mean']:.4f}  @ threshold {best_t:.2f}")
    print(f"IoU        : {summary[best_t]['iou_mean']:.4f}")
    print(f"Sensitivity: {summary[best_t]['sens_mean']:.4f}")
    train_val_dice = ckpt.get("best_dice", None)
    if train_val_dice is not None:
        delta = summary[best_t]["dice_mean"] - float(train_val_dice)
        print(f"Lift vs training-val Dice ({float(train_val_dice):.4f}): {delta:+.4f}")
    print("=" * 72)

    # ── Save JSON ──
    payload = {
        "ckpt":         str(args.ckpt),
        "model":        args.model,
        "overlap":      args.overlap,
        "tta":          not args.no_tta,
        "patch_size":   list(patch_size),
        "thresholds":   thresholds,
        "n_cases":      n_cases,
        "total_minutes": round(total_time / 60, 2),
        "per_threshold": {f"{t:.2f}": v for t, v in summary.items()},
        "best_threshold": best_t,
        "best_dice":      summary[best_t]["dice_mean"],
        "best_iou":       summary[best_t]["iou_mean"],
        "best_sens":      summary[best_t]["sens_mean"],
        "train_val_best_dice": float(train_val_dice) if train_val_dice is not None else None,
    }
    out_json = out_dir / "tta_results.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
