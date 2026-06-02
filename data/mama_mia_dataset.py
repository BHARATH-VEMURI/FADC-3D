import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from monai.data import CacheDataset, PersistentDataset, DataLoader as MonaiLoader
from monai.data.utils import list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ConcatItemsd,
    DeleteItemsd,
    EnsureTyped,
    ToTensord,
)


DATA_ROOT = r"C:\Users\bhara\Desktop\MAMA_MIA_COMPLETE"


def _safe_collate(batch):
    """numpy→tensor conversion before list_data_collate.

    PersistentDataset loads cached items as numpy arrays. PyTorch's default_collate
    (called inside list_data_collate) can't handle numpy, so we convert first.
    """
    def convert(item):
        if isinstance(item, dict):
            return {k: convert(v) for k, v in item.items()}
        if isinstance(item, list):
            return [convert(i) for i in item]
        if isinstance(item, np.ndarray):
            return torch.as_tensor(item.copy())
        return item

    return list_data_collate([convert(b) for b in batch])

COLLECTIONS = ["DUKE", "ISPY1", "ISPY2", "NACT"]

# FL client assignment — one collection per client
CLIENT_MAP = {
    "DUKE":  0,
    "ISPY1": 1,
    "ISPY2": 2,
    "NACT":  3,
}


# ─────────────────────────────────────────────
# 1. FILE DISCOVERY
# ─────────────────────────────────────────────

def discover_cases(data_root: str, split_csv: str = None, split: str = "train") -> list[dict]:
    """
    Scan the images folder and match each patient to their expert segmentation.
    Returns a list of dicts: {image, label, patient_id, collection, client_id}

    If split_csv is given, filter to only train or test cases.
    """
    data_root = Path(data_root)
    images_dir = data_root / "images"
    seg_dir    = data_root / "segmentations" / "expert"

    # Load official train/test split if provided
    # CSV format: two columns — train_split, test_split — each row has one train and one test patient ID
    split_ids = None
    if split_csv:
        df = pd.read_csv(split_csv)
        col = "train_split" if split == "train" else "test_split"
        split_ids = set(df[col].dropna().astype(str).str.lower())

    cases = []
    for patient_folder in sorted(images_dir.iterdir()):
        if not patient_folder.is_dir():
            continue

        folder_name = patient_folder.name          # e.g. DUKE_001
        collection  = folder_name.split("_")[0]    # e.g. DUKE
        patient_id  = folder_name.lower()          # e.g. duke_001

        if collection not in COLLECTIONS:
            continue

        # Filter by split
        if split_ids is not None and patient_id not in split_ids:
            continue

        # Two channels: pre-contrast (_0000) + post-contrast phase 1 (_0001).
        # The pre→post subtraction is the actual DCE-MRI enhancement signal —
        # giving the model both phases lets it learn the optimal combination.
        pre_path = patient_folder / f"{patient_id}_0000.nii.gz"
        if not pre_path.exists():
            pre_path = patient_folder / f"{patient_id}_0000.nii"

        post_path = patient_folder / f"{patient_id}_0001.nii.gz"
        if not post_path.exists():
            post_path = patient_folder / f"{patient_id}_0001.nii"

        seg_path = seg_dir / f"{patient_id}.nii.gz"
        if not seg_path.exists():
            seg_path = seg_dir / f"{patient_id}.nii"

        if not pre_path.exists() or not post_path.exists() or not seg_path.exists():
            continue

        cases.append({
            "image_pre":   str(pre_path),
            "image_post":  str(post_path),
            "label":       str(seg_path),
            "patient_id":  patient_id,
            "collection":  collection,
            "client_id":   CLIENT_MAP[collection],
        })

    return cases


def discover_cases_from_cache(cache_dir: str, split_csv: str = None, split: str = "train") -> list[dict]:
    """
    List .npz files in cache_dir and filter by the official split CSV if provided.

    Used when training from the preprocessed cache — does NOT require the raw
    NIfTI dataset to be mounted. Each .npz already contains the 2-channel image
    and label, so we only need patient_id, collection, and client_id per case.
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return []

    split_ids = None
    if split_csv and os.path.exists(split_csv):
        df = pd.read_csv(split_csv)
        col = "train_split" if split == "train" else "test_split"
        split_ids = set(df[col].dropna().astype(str).str.lower())

    cases = []
    for npz_path in sorted(cache_path.glob("*.npz")):
        patient_id = npz_path.stem.lower()           # e.g. duke_001
        collection = patient_id.split("_")[0].upper()  # e.g. DUKE

        if collection not in COLLECTIONS:
            continue
        if split_ids is not None and patient_id not in split_ids:
            continue

        cases.append({
            "patient_id": patient_id,
            "collection": collection,
            "client_id":  CLIENT_MAP[collection],
        })
    return cases


# ─────────────────────────────────────────────
# 2. TRANSFORMS
# ─────────────────────────────────────────────

DEFAULT_PATCH_SIZE = (128, 128, 64)   # used when no override is provided
POS_NEG_RATIO = 1                     # 1:1 — half patches centered on tumor, half random


def get_train_transforms(patch_size=None):
    ps = tuple(patch_size) if patch_size is not None else DEFAULT_PATCH_SIZE
    img_keys = ["image_pre", "image_post"]
    all_keys = img_keys + ["label"]
    return Compose([
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(
            keys=all_keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        # Per-channel intensity normalization — pre and post have different ranges
        ScaleIntensityRangePercentilesd(
            keys=img_keys,
            lower=1, upper=99,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        # Use post-contrast for foreground crop (brighter, more signal)
        CropForegroundd(keys=all_keys, source_key="image_post"),
        # Stack pre + post into a single 2-channel "image"
        ConcatItemsd(keys=img_keys, name="image", dim=0),
        DeleteItemsd(keys=img_keys),

        SpatialPadd(keys=["image", "label"], spatial_size=ps),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=ps,
            pos=POS_NEG_RATIO,
            neg=POS_NEG_RATIO,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),

        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),

        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms():
    img_keys = ["image_pre", "image_post"]
    all_keys = img_keys + ["label"]
    return Compose([
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(
            keys=all_keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        ScaleIntensityRangePercentilesd(
            keys=img_keys,
            lower=1, upper=99,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=all_keys, source_key="image_post"),
        ConcatItemsd(keys=img_keys, name="image", dim=0),
        DeleteItemsd(keys=img_keys),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_rand_train_transforms(patch_size=None):
    """Random-only transforms applied at training time to preprocessed volumes."""
    ps = tuple(patch_size) if patch_size is not None else DEFAULT_PATCH_SIZE
    return Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=ps,
            pos=POS_NEG_RATIO,
            neg=POS_NEG_RATIO,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])


# ─────────────────────────────────────────────
# 3b. PREPROCESSED CACHE DATASET
# ─────────────────────────────────────────────

class PreprocessedDataset(Dataset):
    """
    Loads preprocessed .npz files (float16 image, uint8 label) produced by
    scripts/preprocess_to_cache.py and applies random augmentations on the fly.

    Each __getitem__ returns a list of 2 patch dicts (from RandCrop num_samples=2)
    — same structure CacheDataset returns, so _safe_collate works unchanged.
    """

    def __init__(self, cache_dir: str, cases: list[dict], is_train: bool = True, patch_size=None):
        self.cache_dir  = Path(cache_dir)
        self.cases      = cases
        self.is_train   = is_train
        self.patch_size = patch_size
        self.transform  = get_rand_train_transforms(patch_size) if is_train else None

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        patient_id = self.cases[idx]["patient_id"]
        npz_path   = self.cache_dir / f"{patient_id}.npz"

        try:
            data_npz = np.load(npz_path)
            item = {
                "image": torch.from_numpy(data_npz["image"].astype(np.float32)),
                "label": torch.from_numpy(data_npz["label"].astype(np.float32)),
            }
            if self.transform is not None:
                item = self.transform(item)
            return item
        except Exception as e:
            # Cases discovered from the cache don't carry raw NIfTI paths, so we can't
            # fall back here. Fail loudly — a corrupted .npz means the cache needs re-upload.
            case = self.cases[idx]
            if "image_pre" not in case or "image_post" not in case:
                raise RuntimeError(
                    f"Failed to load cached .npz for {patient_id} at {npz_path} ({e}). "
                    f"The cache file is missing or corrupted — re-upload the preprocessed cache."
                ) from e
            import warnings
            warnings.warn(f"Corrupted cache for {patient_id}, falling back to NIfTI load")
            fallback = get_train_transforms(self.patch_size) if self.is_train else get_val_transforms()
            result = fallback(case)
            if isinstance(result, list):
                return [{"image": p["image"], "label": p["label"]} for p in result]
            return {"image": result["image"], "label": result["label"]}


# ─────────────────────────────────────────────
# 3. DATASET BUILDERS
# ─────────────────────────────────────────────

def _seed_worker(worker_id):
    """DataLoader worker_init_fn for reproducibility. Seeds numpy + python random
    in each worker from the torch seed PyTorch already derived for that worker."""
    import random as _random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    _random.seed(worker_seed)


def build_centralized_loaders(
    data_root: str = DATA_ROOT,
    split_csv: str = None,
    cache_rate: float = 0.0,
    num_workers: int = 2,
    batch_size: int = 2,
    max_cases: int = None,
    persistent_cache_dir: str = "",
    preprocessed_cache_dir: str = "",
    patch_size=None,
    seed: int = None,
):
    """
    Returns (train_loader, val_loader) using all collections combined.

    preprocessed_cache_dir: path to .npz cache built by scripts/preprocess_to_cache.py.
      Fastest option — skips all NIfTI loading and resampling on Kaggle.
      Expected layout: <cache_dir>/train/{patient_id}.npz, <cache_dir>/val/{patient_id}.npz
    persistent_cache_dir: MONAI PersistentDataset cache (preprocess once to disk per session).
    cache_rate=0.0: no caching — slowest but works anywhere.
    """
    if preprocessed_cache_dir:
        # Cache path: list patient_ids from .npz files — raw NIfTI dataset not needed.
        train_cache = os.path.join(preprocessed_cache_dir, "train")
        val_cache   = os.path.join(preprocessed_cache_dir, "val")
        train_cases = discover_cases_from_cache(train_cache, split_csv, split="train")
        val_cases   = discover_cases_from_cache(val_cache,   split_csv, split="test")
        if max_cases is not None:
            train_cases = train_cases[:max_cases]
            val_cases   = val_cases[:max_cases]
        print(f"Train cases: {len(train_cases)} | Val cases: {len(val_cases)}")
        print(f"Using preprocessed .npz cache: {preprocessed_cache_dir}")
        train_ds = PreprocessedDataset(train_cache, train_cases, is_train=True,  patch_size=patch_size)
        val_ds   = PreprocessedDataset(val_cache,   val_cases,   is_train=False, patch_size=patch_size)
    else:
        # Raw NIfTI path: needs the raw dataset with both _0000 and _0001 phases.
        train_cases = discover_cases(data_root, split_csv, split="train")
        val_cases   = discover_cases(data_root, split_csv, split="test")
        if max_cases is not None:
            train_cases = train_cases[:max_cases]
            val_cases   = val_cases[:max_cases]
        print(f"Train cases: {len(train_cases)} | Val cases: {len(val_cases)}")

        if persistent_cache_dir:
            train_cache = os.path.join(persistent_cache_dir, "train")
            val_cache   = os.path.join(persistent_cache_dir, "val")
            os.makedirs(train_cache, exist_ok=True)
            os.makedirs(val_cache,   exist_ok=True)
            print(f"Using PersistentDataset cache: {persistent_cache_dir}")
            train_ds = PersistentDataset(
                data=train_cases,
                transform=get_train_transforms(patch_size),
                cache_dir=train_cache,
            )
            val_ds = PersistentDataset(
                data=val_cases,
                transform=get_val_transforms(),
                cache_dir=val_cache,
            )
        else:
            train_ds = CacheDataset(
                data=train_cases,
                transform=get_train_transforms(patch_size),
                cache_rate=cache_rate,
                num_workers=num_workers,
            )
            val_ds = CacheDataset(
                data=val_cases,
                transform=get_val_transforms(),
                cache_rate=cache_rate,
                num_workers=num_workers,
            )

    loader_kwargs = dict(collate_fn=_safe_collate, pin_memory=True,
                         persistent_workers=num_workers > 0)
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
        loader_kwargs["worker_init_fn"] = _seed_worker
        loader_kwargs["generator"] = gen

    train_loader = MonaiLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, **loader_kwargs)
    val_loader   = MonaiLoader(val_ds,   batch_size=1,          shuffle=False,
                               num_workers=num_workers, **loader_kwargs)

    return train_loader, val_loader


def build_client_loaders(
    client_id: int,
    data_root: str = DATA_ROOT,
    split_csv: str = None,
    cache_rate: float = 0.0,
    num_workers: int = 2,
    batch_size: int = 2,
):
    """
    Returns (train_loader, val_loader) for a single FL client (one collection).
    """
    collection = [k for k, v in CLIENT_MAP.items() if v == client_id][0]

    train_cases = [c for c in discover_cases(data_root, split_csv, split="train")
                   if c["collection"] == collection]
    val_cases   = [c for c in discover_cases(data_root, split_csv, split="test")
                   if c["collection"] == collection]

    print(f"Client {client_id} ({collection}) — Train: {len(train_cases)} | Val: {len(val_cases)}")

    train_ds = CacheDataset(data=train_cases, transform=get_train_transforms(),
                            cache_rate=cache_rate, num_workers=num_workers)
    val_ds   = CacheDataset(data=val_cases,   transform=get_val_transforms(),
                            cache_rate=cache_rate, num_workers=num_workers)

    train_loader = MonaiLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = MonaiLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# ─────────────────────────────────────────────
# 4. SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 50)
    print("Discovering cases...")
    cases = discover_cases(DATA_ROOT)
    print(f"Total cases found: {len(cases)}")

    by_collection = {}
    for c in cases:
        by_collection.setdefault(c["collection"], 0)
        by_collection[c["collection"]] += 1
    for col, count in sorted(by_collection.items()):
        print(f"  {col}: {count} cases")

    print("\nTesting transforms on 1 case (CPU, no cache)...")
    from monai.data import CacheDataset
    from monai.data import DataLoader as MonaiLoader

    test_cases = cases[:2]
    ds = CacheDataset(data=test_cases, transform=get_train_transforms(),
                      cache_rate=0.0, num_workers=0)

    t0 = time.time()
    sample = ds[0]
    elapsed = time.time() - t0

    print(f"  Time to load + preprocess 1 case: {elapsed:.1f}s")
    print(f"  Patches returned: {len(sample)}")
    for i, patch in enumerate(sample):
        print(f"  Patch {i} — image: {patch['image'].shape} | label: {patch['label'].shape} "
              f"| range: [{patch['image'].min():.3f}, {patch['image'].max():.3f}] "
              f"| tumor voxels: {int(patch['label'].sum())}")
    print("\nSmoke test passed.")
