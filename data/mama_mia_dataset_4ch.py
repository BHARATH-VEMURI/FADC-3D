"""
4-channel MAMA-MIA dataset module — sibling of mama_mia_dataset.py.

Channels in the cache:
  0: pre-contrast (_0000)
  1: post-contrast #1 (_0001)
  2: post-contrast #2 (_0002)
  3: subtraction (post1_norm - pre_norm)

Built-in phase augmentation (PhaseAugment4Ch):
  During training, with probability p_apply, randomly permutes channels [1, 2, 3]
  (the three post-like channels). Channel 0 (pre) is never moved.
  This mimics the MAMA-MIA paper's single-channel phase substitution adapted
  for a 4-channel input — the model learns the post-like channels are
  interchangeable in role, making it robust to which post-phase appears where.

  Validation: no permutation; natural order [pre, post1, post2, sub] always.

Does NOT touch data/mama_mia_dataset.py. Both modules can coexist.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset

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
    EnsureTyped,
    MapTransform,
)


# ── Constants ────────────────────────────────────────────────────────────────

DATA_ROOT = r"C:\Users\bhara\Desktop\MAMA_MIA_COMPLETE"
DEFAULT_PATCH_SIZE = (128, 128, 64)
POS_NEG_RATIO = 1
COLLECTIONS = ["DUKE", "ISPY1", "ISPY2", "NACT"]
CLIENT_MAP = {"DUKE": 0, "ISPY1": 1, "ISPY2": 2, "NACT": 3}

IN_CHANNELS_4CH = 4


# ── Collate ──────────────────────────────────────────────────────────────────

def _safe_collate(batch):
    """numpy→tensor conversion before list_data_collate (same as 2ch module)."""
    def convert(item):
        if isinstance(item, dict):
            return {k: convert(v) for k, v in item.items()}
        if isinstance(item, list):
            return [convert(i) for i in item]
        if isinstance(item, np.ndarray):
            return torch.as_tensor(item.copy())
        return item
    return list_data_collate([convert(b) for b in batch])


# ── Phase augmentation ───────────────────────────────────────────────────────

class PhaseAugment4Ch(MapTransform):
    """Randomly permute channels [1, 2, 3] of a 4-channel image.

    Inspired by the MAMA-MIA paper's single-channel phase substitution
    augmentation. Here adapted to 4-channel input: with probability `p_apply`,
    we shuffle the three post-like channels among positions 1, 2, 3. Channel 0
    (pre-contrast) is fixed.

    With p_apply=0.5 (default), half the training batches see the natural
    channel order [pre, post1, post2, sub] — same as validation — and the
    other half see one of the 5 non-natural permutations of {post1, post2, sub}.

    Compatible with the dict-style transforms used elsewhere. Operates in-place.
    """

    def __init__(self, keys=("image",), p_apply: float = 0.5, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.p_apply = p_apply

    def __call__(self, data):
        d = dict(data)
        if np.random.rand() > self.p_apply:
            return d
        for key in self.key_iterator(d):
            image = d[key]
            # image is (4, H, W, D); torch or numpy
            if isinstance(image, np.ndarray):
                perm = np.random.permutation([1, 2, 3])
                new_image = image.copy()
                new_image[1] = image[perm[0]]
                new_image[2] = image[perm[1]]
                new_image[3] = image[perm[2]]
            else:
                perm = np.random.permutation([1, 2, 3])
                new_image = image.clone()
                new_image[1] = image[perm[0]]
                new_image[2] = image[perm[1]]
                new_image[3] = image[perm[2]]
            d[key] = new_image
        return d


# ── Discovery ────────────────────────────────────────────────────────────────

def discover_cases_4ch(data_root: str, split_csv: str = None, split: str = "train") -> list[dict]:
    """Find cases with all THREE universal phases. Subtraction is derived later.

    Returns dicts with image_pre / image_post1 / image_post2 / label paths +
    patient_id, collection, client_id.
    """
    data_root = Path(data_root)
    images_dir = data_root / "images"
    seg_dir    = data_root / "segmentations" / "expert"

    split_ids = None
    if split_csv and os.path.exists(split_csv):
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
            "client_id":   CLIENT_MAP[collection],
        })
    return cases


def discover_cases_from_cache_4ch(cache_dir: str, split_csv: str = None, split: str = "train") -> list[dict]:
    """List .npz files in 4ch cache_dir, filter by split CSV if provided."""
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
        patient_id = npz_path.stem.lower()
        collection = patient_id.split("_")[0].upper()
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


# ── Transforms ───────────────────────────────────────────────────────────────

def get_rand_train_transforms_4ch(patch_size=None, phase_aug: bool = True, p_phase_aug: float = 0.5):
    """Training transforms applied to preprocessed 4-channel volumes.

    Order:
      1. RandCropByPosNegLabel — patch sampling biased toward tumor
      2. RandFlip × 3 axes — geometric robustness
      3. RandRotate90 — geometric robustness
      4. RandScaleIntensity / RandShiftIntensity — intensity robustness
      5. PhaseAugment4Ch (optional) — channel permutation for phase-order invariance
      6. EnsureType

    Phase aug runs LAST among augmentations so the geometric/intensity ops see
    the natural channel order (consistent supervision signal).
    """
    ps = tuple(patch_size) if patch_size is not None else DEFAULT_PATCH_SIZE
    transforms = [
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
    ]
    if phase_aug:
        transforms.append(PhaseAugment4Ch(keys=["image"], p_apply=p_phase_aug))
    transforms.append(EnsureTyped(keys=["image", "label"]))
    return Compose(transforms)


def get_val_transforms_4ch():
    """No phase aug at validation — natural channel order [pre, post1, post2, sub]."""
    return Compose([EnsureTyped(keys=["image", "label"])])


# ── Dataset ──────────────────────────────────────────────────────────────────

class PreprocessedDataset4Ch(Dataset):
    """Loads preprocessed 4ch .npz files and applies random augmentations."""

    def __init__(self, cache_dir: str, cases: list[dict], is_train: bool = True,
                 patch_size=None, phase_aug: bool = True, p_phase_aug: float = 0.5):
        self.cache_dir  = Path(cache_dir)
        self.cases      = cases
        self.is_train   = is_train
        self.patch_size = patch_size
        if is_train:
            self.transform = get_rand_train_transforms_4ch(
                patch_size, phase_aug=phase_aug, p_phase_aug=p_phase_aug)
        else:
            self.transform = get_val_transforms_4ch()

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
            assert item["image"].shape[0] == IN_CHANNELS_4CH, (
                f"Expected 4-channel image, got {item['image'].shape} for {patient_id}")
            item = self.transform(item)
            return item
        except Exception as e:
            raise RuntimeError(
                f"Failed to load 4ch .npz for {patient_id} at {npz_path} ({e}). "
                f"Re-upload the cache or re-run scripts/4ch_preprocess_to_cache.py."
            ) from e


# ── Loaders ──────────────────────────────────────────────────────────────────

def _seed_worker(worker_id):
    import random as _random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    _random.seed(worker_seed)


def build_centralized_loaders_4ch(
    preprocessed_cache_dir: str,
    split_csv: str = None,
    num_workers: int = 2,
    batch_size: int = 2,
    max_cases: int = None,
    patch_size=None,
    seed: int = None,
    phase_aug: bool = True,
    p_phase_aug: float = 0.5,
):
    """Returns (train_loader, val_loader) for 4ch preprocessed cache."""
    train_cache = os.path.join(preprocessed_cache_dir, "train")
    val_cache   = os.path.join(preprocessed_cache_dir, "val")
    train_cases = discover_cases_from_cache_4ch(train_cache, split_csv, split="train")
    val_cases   = discover_cases_from_cache_4ch(val_cache,   split_csv, split="test")
    if max_cases is not None:
        train_cases = train_cases[:max_cases]
        val_cases   = val_cases[:max_cases]
    print(f"Train cases: {len(train_cases)} | Val cases: {len(val_cases)}")
    print(f"Using 4ch preprocessed cache: {preprocessed_cache_dir}")
    print(f"Phase augmentation: {'ON (p=%.2f)' % p_phase_aug if phase_aug else 'OFF'}")

    train_ds = PreprocessedDataset4Ch(
        train_cache, train_cases, is_train=True, patch_size=patch_size,
        phase_aug=phase_aug, p_phase_aug=p_phase_aug,
    )
    val_ds = PreprocessedDataset4Ch(
        val_cache, val_cases, is_train=False, patch_size=patch_size,
        phase_aug=False,  # never aug at val
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


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    cache_dir = r"C:\Users\bhara\Desktop\4ch_mama_mia_cache"
    if not os.path.exists(cache_dir):
        print(f"4ch cache not found at {cache_dir}.")
        print("Run scripts/4ch_preprocess_to_cache.py first.")
        sys.exit(0)

    train_cases = discover_cases_from_cache_4ch(os.path.join(cache_dir, "train"))
    val_cases   = discover_cases_from_cache_4ch(os.path.join(cache_dir, "val"))
    print(f"Train: {len(train_cases)} | Val: {len(val_cases)}")
    if not train_cases:
        sys.exit(0)

    ds = PreprocessedDataset4Ch(
        os.path.join(cache_dir, "train"), train_cases[:2],
        is_train=True, phase_aug=True, p_phase_aug=1.0,
    )
    t0 = time.time()
    sample = ds[0]
    elapsed = time.time() - t0

    item = sample[0] if isinstance(sample, list) else sample
    print(f"Time: {elapsed:.2f}s")
    print(f"Image: {item['image'].shape} | range: [{item['image'].min():.3f}, {item['image'].max():.3f}]")
    print(f"Label: {item['label'].shape} | tumor voxels: {int(item['label'].sum())}")
    print(f"Channel min/max:")
    for c in range(item['image'].shape[0]):
        print(f"  ch{c}: [{item['image'][c].min():.3f}, {item['image'][c].max():.3f}]")
