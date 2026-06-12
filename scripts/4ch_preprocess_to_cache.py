"""
4-channel preprocessing for MAMA-MIA — builds a fresh cache separate from the 2ch one.

Channels per case (universal across all 1506 patients):
  0: pre-contrast       (_0000)
  1: post-contrast #1   (_0001)   -- peak enhancement reference
  2: post-contrast #2   (_0002)   -- mid post-contrast
  3: subtraction        (post1 - pre)  -- explicit enhancement map

Why these 4: per the phase audit (scripts/audit_mama_mia_phases.py), only
_0000/_0001/_0002 are universal. _0003 is missing in 84% of ISPY1 and 91% of
NACT, which would gut the federated multi-center story if dropped.
Subtraction is derivable from universal phases and matches the MAMA-MIA paper's
input philosophy (they used subtraction as one of their augmentation phases).

Normalization order: each phase is percentile-clipped to [0,1] independently
FIRST, then subtraction is computed as (post1_norm - pre_norm) -> range [-1, +1].

Output: <out_dir>/{train,val}/{patient_id}.npz with
        image: (4, 128, 128, 64) float16
        label: (1, 128, 128, 64) uint8

Does NOT touch the existing 2ch cache at C:\\Users\\bhara\\Desktop\\mama_mia_cache.

Usage:
    python scripts/4ch_preprocess_to_cache.py \\
        --data_root "C:/Users/bhara/Desktop/MAMA_MIA_COMPLETE" \\
        --out_dir   "C:/Users/bhara/Desktop/4ch_mama_mia_cache" \\
        --n_jobs 4

Estimated time: ~3-5 h with 4 workers.
Estimated size: ~40-42 GB compressed.
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))


PATCH_SIZE = (128, 128, 64)
COLLECTIONS = ["DUKE", "ISPY1", "ISPY2", "NACT"]


# ── Discovery ────────────────────────────────────────────────────────────────

def discover_cases_4ch(data_root: str, split_csv: str = None, split: str = "train") -> list[dict]:
    """Return cases that have all THREE universal phases (_0000, _0001, _0002).
    Subtraction (channel 3) is derived, not loaded.
    """
    import pandas as pd
    data_root = Path(data_root)
    images_dir = data_root / "images"
    seg_dir    = data_root / "segmentations" / "expert"

    split_ids = None
    if split_csv and Path(split_csv).exists():
        df = pd.read_csv(split_csv)
        col = "train_split" if split == "train" else "test_split"
        split_ids = set(df[col].dropna().astype(str).str.lower())

    cases = []
    for patient_folder in sorted(images_dir.iterdir()):
        if not patient_folder.is_dir():
            continue
        name = patient_folder.name
        collection = name.split("_")[0]
        patient_id = name.lower()
        if collection not in COLLECTIONS:
            continue
        if split_ids is not None and patient_id not in split_ids:
            continue

        pre_path   = patient_folder / f"{patient_id}_0000.nii.gz"
        post1_path = patient_folder / f"{patient_id}_0001.nii.gz"
        post2_path = patient_folder / f"{patient_id}_0002.nii.gz"
        if not pre_path.exists():   pre_path   = patient_folder / f"{patient_id}_0000.nii"
        if not post1_path.exists(): post1_path = patient_folder / f"{patient_id}_0001.nii"
        if not post2_path.exists(): post2_path = patient_folder / f"{patient_id}_0002.nii"

        seg_path = seg_dir / f"{patient_id}.nii.gz"
        if not seg_path.exists():
            seg_path = seg_dir / f"{patient_id}.nii"

        if not (pre_path.exists() and post1_path.exists()
                and post2_path.exists() and seg_path.exists()):
            continue

        cases.append({
            "image_pre":   str(pre_path),
            "image_post1": str(post1_path),
            "image_post2": str(post2_path),
            "label":       str(seg_path),
            "patient_id":  patient_id,
            "collection":  collection,
        })
    return cases


# ── Worker ───────────────────────────────────────────────────────────────────

def _is_valid_npz(path: str) -> bool:
    try:
        d = np.load(path)
        img_shape = d["image"].shape
        lbl_shape = d["label"].shape
        return img_shape[0] == 4 and lbl_shape[0] == 1
    except Exception:
        return False


def _process_one(task: tuple) -> tuple[str, str]:
    patient_id, image_pre, image_post1, image_post2, label_path, out_path, verify = task

    if Path(out_path).exists():
        if not verify or _is_valid_npz(out_path):
            return patient_id, "skipped"
        Path(out_path).unlink(missing_ok=True)

    try:
        from monai.transforms import (
            Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
            Spacingd, ScaleIntensityRangePercentilesd, CropForegroundd,
            SpatialPadd, EnsureTyped,
        )

        img_keys = ["image_pre", "image_post1", "image_post2"]
        all_keys = img_keys + ["label"]

        # Pipeline: load → orient → resample → normalize per phase → crop
        # Subtraction + concat happen AFTER, by hand (clean dimension control).
        transform = Compose([
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            Orientationd(keys=all_keys, axcodes="RAS"),
            Spacingd(
                keys=all_keys,
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "bilinear", "nearest"),
            ),
            ScaleIntensityRangePercentilesd(
                keys=img_keys, lower=1, upper=99,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=all_keys, source_key="image_post1"),
            EnsureTyped(keys=all_keys),
        ])

        data = transform({
            "image_pre":   image_pre,
            "image_post1": image_post1,
            "image_post2": image_post2,
            "label":       label_path,
        })

        # Each phase is (1, H, W, D), all in [0, 1].
        pre   = data["image_pre"]
        post1 = data["image_post1"]
        post2 = data["image_post2"]
        label = data["label"]

        # Subtraction = post1 - pre  (in normalized space)
        # Range: roughly [-1, +1]. Negative values = pre brighter than post (rare).
        subtraction = post1 - pre

        # Stack to 4-channel image: (4, H, W, D)
        image = torch.cat([pre, post1, post2, subtraction], dim=0)

        # Pad to PATCH_SIZE if any spatial dim is below
        pad = SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE)
        padded = pad({"image": image, "label": label})
        image = padded["image"]
        label = padded["label"]

        image_arr = image.numpy().astype(np.float16)
        label_arr = label.numpy().astype(np.uint8)

        np.savez_compressed(out_path, image=image_arr, label=label_arr)
        return patient_id, "done"

    except Exception as e:
        return patient_id, f"ERROR: {e}"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default=r"C:\Users\bhara\Desktop\MAMA_MIA_COMPLETE")
    parser.add_argument("--out_dir",   type=str,
                        default=r"C:\Users\bhara\Desktop\4ch_mama_mia_cache")
    parser.add_argument("--split_csv", type=str, default=None,
                        help="Path to train_test_splits.csv (auto-detected if omitted)")
    parser.add_argument("--n_jobs",    type=int, default=4,
                        help="Parallel workers")
    parser.add_argument("--verify",    action="store_true",
                        help="Check existing .npz files and reprocess any that are corrupted")
    args = parser.parse_args()

    data_root = args.data_root
    out_dir   = Path(args.out_dir)

    split_csv = args.split_csv
    if split_csv is None:
        auto = os.path.join(data_root, "train_test_splits.csv")
        split_csv = auto if os.path.exists(auto) else None

    train_out = out_dir / "train"
    val_out   = out_dir / "val"
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    train_cases = discover_cases_4ch(data_root, split_csv, split="train")
    val_cases   = discover_cases_4ch(data_root, split_csv, split="test")
    all_cases   = [(c, train_out) for c in train_cases] + \
                  [(c, val_out)   for c in val_cases]

    print(f"Train: {len(train_cases)} | Val: {len(val_cases)} | Total: {len(all_cases)}")
    print(f"Output dir: {out_dir}")
    print(f"Workers   : {args.n_jobs}")
    if args.verify:
        print("Mode: VERIFY — existing files will be checked and corrupted ones reprocessed")
    print()

    tasks = [
        (c["patient_id"], c["image_pre"], c["image_post1"], c["image_post2"], c["label"],
         str(dst / f"{c['patient_id']}.npz"), args.verify)
        for c, dst in all_cases
    ]

    done = skipped = errors = 0
    error_list = []

    if args.n_jobs == 1:
        for task in tqdm(tasks, desc="Preprocessing", unit="case"):
            pid, status = _process_one(task)
            if status == "done":      done    += 1
            elif status == "skipped": skipped += 1
            else:                     errors  += 1; error_list.append((pid, status))
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            futures = {pool.submit(_process_one, t): t[0] for t in tasks}
            with tqdm(total=len(tasks), desc="Preprocessing", unit="case") as pbar:
                for future in as_completed(futures):
                    pid, status = future.result()
                    if status == "done":      done    += 1
                    elif status == "skipped": skipped += 1
                    else:                     errors  += 1; error_list.append((pid, status))
                    pbar.update(1)
                    pbar.set_postfix(done=done, skip=skipped, err=errors)

    print(f"\nDone: {done} | Skipped: {skipped} | Errors: {errors}")
    if error_list:
        print("Failed cases:")
        for pid, msg in error_list[:30]:
            print(f"  {pid}: {msg}")
        if len(error_list) > 30:
            print(f"  ... and {len(error_list) - 30} more")

    total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*.npz"))
    print(f"\nTotal 4ch cache size: {total_bytes / 1e9:.2f} GB")
    print(f"\nNext step: upload '{out_dir}' to Kaggle account 'vbk1999'")
    print("           using scripts/4ch_upload_to_kaggle.py")


if __name__ == "__main__":
    main()
