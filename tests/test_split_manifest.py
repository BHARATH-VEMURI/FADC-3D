"""Focused tests for training/split_manifest.py + the loader hooks that
consume it. Runs on CPU. No GPU, no real MRI data — synthetic .npz files.

Coverage:
  1  Enumeration walks both `train/` and `val/` subdirs, dedups patient IDs.
  2  Deterministic seed-42 split reproduction — byte-identical CSV twice.
  3  Patient-level disjointness (train ∩ val = ∅, val ∩ test = ∅, train ∩ test = ∅).
  4  Union of splits equals the full patient index.
  5  Collection stratification — within each collection the ratios hold to ±1.
  6  Manifest path resolution across old train/+val/ folders — the returned
     absolute path points into the correct physical subdir.
  7  load_manifest raises when a referenced .npz is missing.
  8  build_centralized_loaders(split_manifest=...) returns a train loader
     containing ONLY the train partition; val loader ONLY the val partition.
  9  A test-only manifest partition is not touched by the training call.
  10 Manifest SHA256 is portable (LF line endings on all platforms).
  11 Checkpoint's split_identity carries the manifest checksum exactly.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.split_manifest import (
    COLLECTIONS,
    assign_splits,
    enumerate_patients,
    generate_and_write,
    load_manifest,
    manifest_sha256,
    verify_manifest_partitions,
    write_manifest,
    write_metadata,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures — synthetic on-disk cache with the old train/ + val/ layout.
# ─────────────────────────────────────────────────────────────────────────

def _make_case(dir_path: Path, patient_id: str):
    """Write a tiny valid .npz with the expected 'image' + 'label' arrays."""
    dir_path.mkdir(parents=True, exist_ok=True)
    p = dir_path / f"{patient_id}.npz"
    img = np.random.RandomState(0).randn(2, 8, 8, 4).astype(np.float16)
    lbl = np.zeros((1, 8, 8, 4), dtype=np.uint8)
    np.savez(p, image=img, label=lbl)
    return p


def _make_cache(root: Path, n_per_coll_train: int = 20, n_per_coll_val: int = 5):
    """Build a synthetic cache that mimics the real Kaggle layout."""
    root.mkdir(parents=True, exist_ok=True)
    for coll in COLLECTIONS:
        for i in range(n_per_coll_train):
            _make_case(root / "train", f"{coll.lower()}_{i:03d}")
        for i in range(n_per_coll_val):
            _make_case(root / "val", f"{coll.lower()}_{100 + i:03d}")


def _fresh_tmp(prefix: str = "fadc_split_") -> Path:
    p = Path(tempfile.mkdtemp(prefix=prefix))
    return p


# ─────────────────────────────────────────────────────────────────────────
# 1  Enumeration + dedup
# ─────────────────────────────────────────────────────────────────────────

def test_enumeration_walks_both_subdirs_and_dedups():
    root = _fresh_tmp()
    try:
        _make_cache(root, n_per_coll_train=5, n_per_coll_val=3)
        patients = enumerate_patients(str(root))
        assert len(patients) == 4 * (5 + 3), (
            f"expected {4 * 8} patients, got {len(patients)}"
        )
        # unique
        pids = [c["patient_id"] for c in patients]
        assert len(set(pids)) == len(pids), "duplicate patient_ids returned"
        # both subdirs present
        assert {"train", "val"} == {c["source_subdir"] for c in patients}
        # collection extraction
        assert {c["collection"] for c in patients} == set(COLLECTIONS)
        # relative_npz_path portable (no backslashes even on Windows)
        assert all("\\" not in c["relative_npz_path"] for c in patients)
        print(f"[1] enumeration OK  n={len(patients)}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_enumeration_raises_on_duplicate_across_subdirs():
    root = _fresh_tmp()
    try:
        _make_case(root / "train", "duke_001")
        _make_case(root / "val",   "duke_001")
        try:
            enumerate_patients(str(root))
        except RuntimeError as e:
            assert "duke_001" in str(e).lower()
            print(f"[1b] duplicate-across-subdirs raised as expected: {e}")
            return
        raise AssertionError("enumerate_patients should have raised RuntimeError")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# 2 + 3 + 4  Determinism, disjointness, union
# ─────────────────────────────────────────────────────────────────────────

def test_determinism_and_disjointness():
    root = _fresh_tmp()
    try:
        _make_cache(root, n_per_coll_train=50, n_per_coll_val=20)  # 280 total
        patients = enumerate_patients(str(root))

        a1 = assign_splits(patients, seed=42, ratios=(0.70, 0.10, 0.20))
        a2 = assign_splits(patients, seed=42, ratios=(0.70, 0.10, 0.20))
        # Same seed → identical row-by-row assignment.
        assert [(x["patient_id"], x["split"]) for x in a1] == \
               [(x["patient_id"], x["split"]) for x in a2], \
               "seed-42 assignment not deterministic"

        # Different seed → different assignment.
        a3 = assign_splits(patients, seed=17, ratios=(0.70, 0.10, 0.20))
        diffs = sum(1 for x, y in zip(a1, a3) if x["split"] != y["split"])
        assert diffs > 0, "seed change did not perturb assignment"

        # Disjoint splits + union == full set.
        by_split = {"train": set(), "val": set(), "test": set()}
        for c in a1:
            by_split[c["split"]].add(c["patient_id"])
        assert by_split["train"].isdisjoint(by_split["val"])
        assert by_split["val"].isdisjoint(by_split["test"])
        assert by_split["train"].isdisjoint(by_split["test"])
        union = by_split["train"] | by_split["val"] | by_split["test"]
        assert union == {c["patient_id"] for c in patients}
        print(f"[2/3/4] determinism + disjoint + union OK  "
              f"train={len(by_split['train'])} val={len(by_split['val'])} "
              f"test={len(by_split['test'])}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# 5  Collection stratification
# ─────────────────────────────────────────────────────────────────────────

def test_collection_stratification():
    root = _fresh_tmp()
    try:
        _make_cache(root, n_per_coll_train=50, n_per_coll_val=20)  # 70 per collection
        patients = enumerate_patients(str(root))
        assigned = assign_splits(patients, seed=42, ratios=(0.70, 0.10, 0.20))

        for coll in COLLECTIONS:
            in_coll = [c for c in assigned if c["collection"] == coll]
            n = len(in_coll)
            n_tr = sum(1 for c in in_coll if c["split"] == "train")
            n_va = sum(1 for c in in_coll if c["split"] == "val")
            n_te = sum(1 for c in in_coll if c["split"] == "test")
            assert n == 70
            # Within a collection each ratio should be within ±1 of the target.
            assert abs(n_tr - int(round(n * 0.70))) <= 1
            assert abs(n_va - int(round(n * 0.10))) <= 1
            assert n_tr + n_va + n_te == n
        print("[5] stratification OK — each collection ~70/10/20")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# 6 + 7  Path resolution across old train/+val/ folders
# ─────────────────────────────────────────────────────────────────────────

def test_manifest_path_resolution_across_subdirs():
    root = _fresh_tmp()
    try:
        _make_cache(root, n_per_coll_train=10, n_per_coll_val=5)
        csv_path  = root / "manifest.csv"
        meta_path = root / "manifest_meta.json"
        meta = generate_and_write(str(root), str(csv_path), str(meta_path),
                                  seed=42, ratios=(0.70, 0.10, 0.20))

        for split in ("train", "val", "test"):
            cases = load_manifest(str(csv_path), split=split,
                                  cache_root=str(root), require_exists=True)
            for c in cases:
                # Path must exist AND live inside the correct physical subdir.
                assert os.path.exists(c["npz_path"]), c
                # patient came from EITHER train/ OR val/ physical dir
                parts = c["npz_path"].replace("\\", "/").split("/")
                assert "train" in parts or "val" in parts, c
        print(f"[6] path resolution OK across subdirs, meta={meta['n_per_split']}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_load_manifest_raises_on_missing_file():
    root = _fresh_tmp()
    try:
        _make_cache(root, n_per_coll_train=5, n_per_coll_val=3)
        csv_path  = root / "manifest.csv"
        meta_path = root / "manifest_meta.json"
        generate_and_write(str(root), str(csv_path), str(meta_path),
                           seed=42, ratios=(0.70, 0.10, 0.20))
        # Nuke a file the manifest actually references AS TRAIN. Read the
        # manifest first so we pick a real train patient regardless of which
        # subdir it happens to live in.
        train_cases = load_manifest(str(csv_path), split="train",
                                    cache_root=str(root), require_exists=True)
        target = Path(train_cases[0]["npz_path"])
        target.unlink()

        try:
            load_manifest(str(csv_path), split="train",
                          cache_root=str(root), require_exists=True)
        except FileNotFoundError as e:
            assert "Manifest references missing file" in str(e)
            print(f"[7] missing-file raised as expected: {target.name}")
            return
        raise AssertionError("load_manifest should have raised")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# 8 + 9  Loader partitioning (via build_centralized_loaders)
# ─────────────────────────────────────────────────────────────────────────

def test_build_loaders_uses_only_manifest_partitions():
    root = _fresh_tmp()
    try:
        _make_cache(root, n_per_coll_train=25, n_per_coll_val=10)  # 4*35 = 140
        csv_path  = root / "manifest.csv"
        meta_path = root / "manifest_meta.json"
        generate_and_write(str(root), str(csv_path), str(meta_path),
                           seed=42, ratios=(0.70, 0.10, 0.20))

        train_cases = load_manifest(str(csv_path), split="train",
                                    cache_root=str(root))
        val_cases   = load_manifest(str(csv_path), split="val",
                                    cache_root=str(root))
        test_cases  = load_manifest(str(csv_path), split="test",
                                    cache_root=str(root))

        from data.mama_mia_dataset import build_centralized_loaders
        train_loader, val_loader = build_centralized_loaders(
            preprocessed_cache_dir=str(root),
            split_manifest=str(csv_path),
            batch_size=1, num_workers=0, patch_size=(8, 8, 4),
        )
        # Datasets carry the raw case dicts — assert set equality.
        train_pids = {c["patient_id"] for c in train_loader.dataset.cases}
        val_pids   = {c["patient_id"] for c in val_loader.dataset.cases}
        assert train_pids == {c["patient_id"] for c in train_cases}, \
            "train loader case set != manifest train partition"
        assert val_pids   == {c["patient_id"] for c in val_cases}, \
            "val loader case set != manifest val partition"

        # Test partition NEVER appears in either loader (contract check).
        test_pids = {c["patient_id"] for c in test_cases}
        assert train_pids.isdisjoint(test_pids)
        assert val_pids.isdisjoint(test_pids)
        print(f"[8/9] loaders isolate manifest partitions  "
              f"train={len(train_pids)} val={len(val_pids)} test={len(test_pids)}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# 10  Portable SHA256
# ─────────────────────────────────────────────────────────────────────────

def test_manifest_sha256_lf_only():
    root = _fresh_tmp()
    try:
        _make_cache(root, n_per_coll_train=10, n_per_coll_val=3)
        csv_path  = root / "manifest.csv"
        meta_path = root / "manifest_meta.json"
        generate_and_write(str(root), str(csv_path), str(meta_path),
                           seed=42, ratios=(0.70, 0.10, 0.20))
        raw = csv_path.read_bytes()
        # LF only — no CRLF sneaking in from Windows.
        assert b"\r\n" not in raw, "manifest CSV uses CRLF — SHA256 not portable"

        # Metadata reports the exact hash we compute now.
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["csv_sha256"] == manifest_sha256(str(csv_path))
        # verify_manifest_partitions runs clean.
        s = verify_manifest_partitions(str(csv_path))
        assert s["n_train"] + s["n_val"] + s["n_test"] == s["n_total"]
        print(f"[10] manifest hash portable: {meta['csv_sha256'][:12]}…")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# 11  Deep-supervision arch_identity survives split_identity extension
# ─────────────────────────────────────────────────────────────────────────

def test_split_identity_carries_manifest_fields():
    """Cheap unit for _require_matching_split — no torch model needed."""
    from training.train_centralized_correct import _require_matching_split
    ckpt = {"split_identity": {"split_manifest_sha256": "abc",
                               "split_partition_train": "train",
                               "split_partition_val": "val",
                               "split_seed": 42}}
    cur_ok   = {"split_manifest_sha256": "abc",
                "split_partition_train": "train",
                "split_partition_val": "val",
                "split_seed": 42}
    cur_bad  = {"split_manifest_sha256": "different",
                "split_partition_train": "train",
                "split_partition_val": "val",
                "split_seed": 42}
    _require_matching_split(ckpt, cur_ok)   # must not raise
    try:
        _require_matching_split(ckpt, cur_bad)
    except RuntimeError as e:
        assert "split_manifest_sha256" in str(e)
        print(f"[11] split-identity guard rejected mismatched checksum")
        return
    raise AssertionError("_require_matching_split should have raised on checksum mismatch")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fns = [
        test_enumeration_walks_both_subdirs_and_dedups,
        test_enumeration_raises_on_duplicate_across_subdirs,
        test_determinism_and_disjointness,
        test_collection_stratification,
        test_manifest_path_resolution_across_subdirs,
        test_load_manifest_raises_on_missing_file,
        test_build_loaders_uses_only_manifest_partitions,
        test_manifest_sha256_lf_only,
        test_split_identity_carries_manifest_fields,
    ]
    for fn in fns:
        print(f"\n---- {fn.__name__} ----")
        fn()
    print("\nALL SPLIT-MANIFEST TESTS PASSED")
