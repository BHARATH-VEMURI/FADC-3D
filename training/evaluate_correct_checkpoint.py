"""Standalone formal evaluator for corrected FADC3D checkpoints.

Reads a checkpoint saved by training/train_centralized_correct.py, rebuilds
the exact architecture from `arch_identity`, strict-loads the weights, and
runs sliding-window inference over ALL 306 validation cases at the given
overlap (default 0.5). Writes a JSON report.

Never mutates the checkpoint. Never updates best_model.pth. Purely read-only.

Handles deep-supervision models by wrapping the predictor to consume the
primary head only (delegates to train_centralized_correct.make_primary_predictor).

CLI example:

    python training/evaluate_correct_checkpoint.py \\
        --checkpoint outputs/fadc3d_correct_encoder_s42/best_model.pth \\
        --preprocessed_cache_dir /kaggle/input/.../val_cache \\
        --overlap 0.5 --sw_batch_size 4 \\
        --out results/formal_eval_best.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

sys.path.append(str(Path(__file__).parent.parent))
from data.mama_mia_dataset import build_centralized_loaders, DATA_ROOT
from models.unet_3d_fadc_correct import build_unet3d_fadc_correct
from training.train_centralized_correct import (
    make_primary_predictor,
    _iou_sens_per_case,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a corrected-FADC checkpoint (.pth).")
    p.add_argument("--data_root", type=str, default=DATA_ROOT)
    p.add_argument("--preprocessed_cache_dir", type=str, default=None,
                   help="If set, drives the val loader from a preprocessed .npz cache. "
                        "This is the Kaggle path.")
    p.add_argument("--patch_size", type=int, nargs=3, default=None,
                   help="Override sliding-window patch size. Defaults to the "
                        "checkpoint's config patch_size.")
    p.add_argument("--overlap", type=float, default=0.5,
                   help="Sliding-window inference overlap. Default 0.5 (formal).")
    p.add_argument("--sw_batch_size", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out", type=str, default=None,
                   help="Destination for the results JSON. Defaults to "
                        "<checkpoint_dir>/formal_eval_<epoch>_<overlap>.json.")
    return p.parse_args()


def _git_commit_hash(repo_root: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device       : {device}")
    print(f"checkpoint   : {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch = ckpt.get("arch_identity")
    if arch is None:
        raise SystemExit("Refusing to evaluate: checkpoint has no arch_identity.")
    cfg = ckpt.get("config", {})
    model_name = arch["model_name"]
    epoch = int(ckpt.get("epoch", -1))
    ckpt_best = float(ckpt.get("best_dice", 0.0))
    print(f"epoch        : {epoch}")
    print(f"model_name   : {model_name}")
    print(f"arch_identity: {arch}")
    print(f"ckpt best_dice(formal, self-reported): {ckpt_best:.4f}")

    # Rebuild the exact model.
    adakern_cfg = {"use_position_att": bool(arch.get("fadc_correct", {}).get("use_position_att", False))}
    model = build_unet3d_fadc_correct(
        model_name=model_name,
        in_channels=int(arch["in_channels"]),
        out_channels=int(arch["out_channels"]),
        base_filters=int(arch["base_filters"]),
        deep_supervision=bool(arch["deep_supervision"]),
        adakern_cfg=adakern_cfg,
    ).to(device).eval()
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    if missing or unexpected:
        raise SystemExit(f"strict load failed: missing={missing} unexpected={unexpected}")

    # Patch size — prefer CLI override, else ckpt config.
    if args.patch_size is not None:
        patch_size = tuple(args.patch_size)
    else:
        patch_size = tuple(cfg.get("data", {}).get("patch_size", [128, 128, 64]))
    print(f"patch_size   : {patch_size}")
    print(f"overlap      : {args.overlap}   sw_batch_size : {args.sw_batch_size}")

    # Build the val loader.
    split_csv = os.path.join(args.data_root, "train_test_splits.csv")
    if not os.path.exists(split_csv):
        split_csv = None
    _, val_loader = build_centralized_loaders(
        data_root=args.data_root,
        split_csv=split_csv,
        cache_rate=0.0,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        preprocessed_cache_dir=args.preprocessed_cache_dir or "",
        patch_size=patch_size,
        seed=0,
    )

    # Metrics.
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    predictor = make_primary_predictor(model)

    try:
        total = len(val_loader)
    except TypeError:
        total = None

    print("\n==== FORMAL FULL VALIDATION (standalone, read-only) ====")
    t0 = time.time()
    iou_list, sens_list, per_case = [], [], []
    with torch.no_grad():
        pbar = tqdm(val_loader, total=total, desc="FORMAL eval",
                    unit="case", file=sys.stdout, ncols=100,
                    dynamic_ncols=False, mininterval=1.0)
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            with autocast("cuda", enabled=device.type == "cuda"):
                preds = sliding_window_inference(
                    inputs=images, roi_size=patch_size,
                    sw_batch_size=args.sw_batch_size, predictor=predictor,
                    overlap=args.overlap,
                )
            preds_bin_list = [post_pred(p) for p in decollate_batch(preds)]
            labels_bin_list = [post_label(l.long()) for l in decollate_batch(labels)]
            dice_metric(y_pred=preds_bin_list, y=labels_bin_list)
            for pb, lb in zip(preds_bin_list, labels_bin_list):
                iou, sens = _iou_sens_per_case(pb[1].float(), lb[1].float())
                iou_list.append(iou)
                sens_list.append(sens)
                per_case.append({"iou": iou, "sensitivity": sens})
        pbar.close()

    mean_dice = dice_metric.aggregate().item() if iou_list else 0.0
    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    mean_sens = float(np.mean(sens_list)) if sens_list else 0.0
    elapsed = time.time() - t0

    print(f"\nFORMAL Dice        : {mean_dice:.4f}")
    print(f"FORMAL IoU         : {mean_iou:.4f}")
    print(f"FORMAL Sensitivity : {mean_sens:.4f}")
    print(f"n_cases            : {len(iou_list)}")
    print(f"elapsed            : {elapsed/60:.1f} min")

    # Destination.
    if args.out:
        out_path = Path(args.out)
    else:
        ck_dir = Path(args.checkpoint).parent
        out_path = ck_dir / f"formal_eval_ep{epoch:03d}_ov{args.overlap:.2f}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = str(Path(__file__).parent.parent)
    result = {
        "checkpoint":          str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch":    epoch,
        "checkpoint_best_dice_selfreported": ckpt_best,
        "model_name":          model_name,
        "arch_identity":       arch,
        "overlap":             args.overlap,
        "sw_batch_size":       args.sw_batch_size,
        "patch_size":          list(patch_size),
        "n_cases":             len(iou_list),
        "dice":                mean_dice,
        "iou":                 mean_iou,
        "sensitivity":         mean_sens,
        "elapsed_seconds":     elapsed,
        "per_case_metrics":    per_case,
        "git_commit":          _git_commit_hash(repo_root),
        "evaluated_at":        time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nresults written : {out_path}")


if __name__ == "__main__":
    main()
