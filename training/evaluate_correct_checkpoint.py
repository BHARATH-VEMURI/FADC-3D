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
from models.unet_3d_fadc_correct import (
    build_unet3d_fadc_correct, MODEL_NAMES as DISCRETE_MODEL_NAMES,
)
from models.unet_3d_fadc_continuous import (
    build_unet3d_fadc_continuous, MODEL_NAMES as CONTINUOUS_MODEL_NAMES,
)
from training.train_centralized_correct import (
    make_primary_predictor,
    _iou_sens_per_case,
)


def _build_model_from_arch(arch: dict, device: torch.device) -> torch.nn.Module:
    """Rebuild the exact model captured by `arch_identity`.

    Dispatch on `arch_kind` — 'continuous' → continuous package,
    everything else (including legacy ckpts with no arch_kind field) →
    discrete package.
    """
    kind = arch.get("arch_kind", "discrete")
    name = arch["model_name"]
    in_ch = int(arch["in_channels"])
    out_ch = int(arch["out_channels"])
    base = int(arch["base_filters"])
    ds = bool(arch["deep_supervision"])
    if kind == "continuous":
        if name not in CONTINUOUS_MODEL_NAMES:
            raise SystemExit(
                f"Refusing to evaluate: arch_kind=continuous but model_name "
                f"{name!r} is not registered as a continuous model."
            )
        if ds:
            raise SystemExit(
                "Refusing to evaluate: continuous checkpoint has deep_supervision=True, "
                "which the continuous encoder does not support."
            )
        return build_unet3d_fadc_continuous(
            model_name=name, in_channels=in_ch, out_channels=out_ch,
            base_filters=base, deep_supervision=False,
        ).to(device).eval()
    # discrete (legacy or explicit)
    if name not in DISCRETE_MODEL_NAMES:
        raise SystemExit(
            f"Refusing to evaluate: arch_kind={kind!r} but model_name "
            f"{name!r} is not registered as a discrete model."
        )
    adakern_cfg = {
        "use_position_att": bool(arch.get("fadc_correct", {}).get("use_position_att", False))
    }
    return build_unet3d_fadc_correct(
        model_name=name, in_channels=in_ch, out_channels=out_ch,
        base_filters=base, deep_supervision=ds, adakern_cfg=adakern_cfg,
    ).to(device).eval()


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
    p.add_argument("--split_manifest", type=str, default=None,
                   help="Path to the split-manifest CSV. When set, the evaluator "
                        "loads from --split_partition (default 'test') instead of "
                        "enumerating <preprocessed_cache_dir>/val.")
    p.add_argument("--split_partition", type=str, default="test",
                   help="Manifest partition to evaluate: train|val|test. Default 'test' "
                        "(the locked final-test partition).")
    p.add_argument("--require_manifest_checksum", type=str, default=None,
                   help="If set, refuse to evaluate unless the checkpoint's stored "
                        "split_manifest_sha256 equals this hex digest. Enforces "
                        "that the checkpoint was trained on the same split.")
    p.add_argument("--per_collection", action="store_true",
                   help="Also emit per-collection Dice/IoU/Sensitivity in the "
                        "results JSON (needed for the final-test summary).")
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

    # Rebuild the exact model (discrete or continuous).
    model = _build_model_from_arch(arch, device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    if missing or unexpected:
        raise SystemExit(f"strict load failed: missing={missing} unexpected={unexpected}")

    # Patch size — prefer CLI override, else ckpt config.
    if args.patch_size is not None:
        _raw_patch = tuple(args.patch_size)
        _patch_source = "--patch_size"
    else:
        _raw_patch = tuple(cfg.get("data", {}).get("patch_size", [128, 128, 64]))
        _patch_source = "checkpoint cfg[\"data\"][\"patch_size\"]"
    from training.patch_validation import validate_patch_size as _vps
    patch_size = _vps(_raw_patch, name=f"sliding-window patch_size (source={_patch_source})")
    print(f"patch_size   : {patch_size}")
    print(f"overlap      : {args.overlap}   sw_batch_size : {args.sw_batch_size}")

    # Manifest guard: when the caller passes a required checksum, refuse to
    # evaluate a checkpoint that was trained on a different manifest. Cheap,
    # catches an entire class of "wrong split" bugs at the final-test step.
    if args.require_manifest_checksum:
        ckpt_sig = (ckpt.get("split_identity") or {}).get("split_manifest_sha256")
        if ckpt_sig != args.require_manifest_checksum:
            raise SystemExit(
                f"Refusing to evaluate: checkpoint's split_manifest_sha256 "
                f"({ckpt_sig!r}) does not match --require_manifest_checksum "
                f"({args.require_manifest_checksum!r})."
            )

    # Build the loader for the requested partition. Manifest short-circuits
    # the legacy train_test_splits.csv path.
    if args.split_manifest:
        if not os.path.exists(args.split_manifest):
            raise SystemExit(f"--split_manifest not found: {args.split_manifest}")
        # build_centralized_loaders returns (train_loader, val_loader). We
        # only want the requested partition, so map it into the val slot and
        # discard train. Set split_partition_train to something that WILL
        # exist to avoid a KeyError; the loader is thrown away.
        _dummy_partition = args.split_partition
        _, val_loader = build_centralized_loaders(
            data_root=args.data_root,
            split_csv=None,
            cache_rate=0.0,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            preprocessed_cache_dir=args.preprocessed_cache_dir or "",
            patch_size=patch_size,
            seed=0,
            split_manifest=args.split_manifest,
            split_train=_dummy_partition,   # not iterated; loader discarded
            split_val=args.split_partition,
        )
        print(f"Evaluating manifest partition : {args.split_partition!r}")
    else:
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
    # Per-case Dice — needed for median/std and per-collection breakdown.
    # DiceMetric.aggregate() returns the batch mean; we recompute per case
    # using a fresh metric instance per iteration so we get the raw list.
    per_case_dice: list[float] = []
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
            _one = DiceMetric(include_background=False, reduction="mean")
            _one(y_pred=preds_bin_list, y=labels_bin_list)
            _one_val = _one.aggregate().item()
            per_case_dice.append(_one_val)
            # Best-effort patient_id / collection extraction. MONAI decollate
            # returns meta only if attached upstream; the manifest loader
            # doesn't attach it, so we recover from the batch payload when
            # available.
            pid = None
            coll = None
            if "patient_id" in batch:
                v = batch["patient_id"]
                pid = v[0] if isinstance(v, (list, tuple)) else v
            if "collection" in batch:
                v = batch["collection"]
                coll = v[0] if isinstance(v, (list, tuple)) else v
            for pb, lb in zip(preds_bin_list, labels_bin_list):
                iou, sens = _iou_sens_per_case(pb[1].float(), lb[1].float())
                iou_list.append(iou)
                sens_list.append(sens)
                per_case.append({"dice": _one_val, "iou": iou,
                                 "sensitivity": sens,
                                 "patient_id": pid, "collection": coll})
        pbar.close()

    mean_dice = dice_metric.aggregate().item() if iou_list else 0.0
    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    mean_sens = float(np.mean(sens_list)) if sens_list else 0.0
    elapsed = time.time() - t0

    # Optional per-collection metrics — only meaningful when the manifest
    # attaches collection info per case. Emitted whenever --per_collection
    # is set, even if some entries are None (surfaced as an 'unknown' bucket).
    per_collection: dict = {}
    if args.per_collection:
        buckets: dict = {}
        for entry in per_case:
            key = entry.get("collection") or "unknown"
            buckets.setdefault(key, {"dice": [], "iou": [], "sens": []})
            buckets[key]["dice"].append(entry["dice"])
            buckets[key]["iou"].append(entry["iou"])
            buckets[key]["sens"].append(entry["sensitivity"])
        for key, d in buckets.items():
            per_collection[key] = {
                "n":           len(d["dice"]),
                "dice_mean":   float(np.mean(d["dice"])),
                "dice_median": float(np.median(d["dice"])),
                "dice_std":    float(np.std(d["dice"], ddof=1)) if len(d["dice"]) > 1 else 0.0,
                "iou_mean":    float(np.mean(d["iou"])),
                "sens_mean":   float(np.mean(d["sens"])),
            }

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
    # Per-case Dice stats — used for the final-test report's median/std/CI.
    if per_case_dice:
        _pcd = np.asarray(per_case_dice, dtype=np.float64)
        dice_median = float(np.median(_pcd))
        dice_std = float(np.std(_pcd, ddof=1)) if len(_pcd) > 1 else 0.0
        # 95% CI via Student-t; falls back to +/- 1.96*sem if scipy missing.
        try:
            from scipy import stats as _sst
            sem = dice_std / np.sqrt(len(_pcd))
            tcrit = _sst.t.ppf(0.975, df=len(_pcd) - 1) if len(_pcd) > 1 else 0.0
            dice_ci95 = [float(mean_dice - tcrit * sem), float(mean_dice + tcrit * sem)]
        except Exception:
            sem = dice_std / (len(_pcd) ** 0.5) if len(_pcd) else 0.0
            dice_ci95 = [float(mean_dice - 1.96 * sem), float(mean_dice + 1.96 * sem)]
    else:
        dice_median = 0.0
        dice_std = 0.0
        dice_ci95 = [0.0, 0.0]

    result = {
        "checkpoint":          str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch":    epoch,
        "checkpoint_best_dice_selfreported": ckpt_best,
        "model_name":          model_name,
        "arch_identity":       arch,
        "split_identity":      ckpt.get("split_identity"),
        "split_partition_evaluated": args.split_partition if args.split_manifest else None,
        "overlap":             args.overlap,
        "sw_batch_size":       args.sw_batch_size,
        "patch_size":          list(patch_size),
        "n_cases":             len(iou_list),
        "dice":                mean_dice,
        "dice_median":         dice_median,
        "dice_std":            dice_std,
        "dice_ci95":           dice_ci95,
        "iou":                 mean_iou,
        "sensitivity":         mean_sens,
        "per_collection":      per_collection if args.per_collection else None,
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
