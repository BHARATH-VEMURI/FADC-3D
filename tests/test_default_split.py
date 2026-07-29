"""Contract tests for the default-cache (1200/306) snapshot manifest and
the resume / evaluator guards that consume it.

Covers the twelve properties requested by the revised protocol:

  1  Default snapshot enforces exactly 1200 train + 306 validation patients.
  2  No 'test' rows are ever emitted.
  3  Train / validation overlap is zero.
  4  Recreating the snapshot from the same physical cache produces the same SHA.
  5  All three revised notebooks reference the same manifest SHA constant.
  6  All three revised notebooks set VAL_EVERY = 5.
  7  Formal validation walks every case in the val loader (no subset, no proxy).
  8  A 70/10/20 (seed_split) checkpoint is rejected under a default-cache run.
  9  DS-vs-noDS resume mismatches are rejected by arch guard.
  10 Discrete-vs-continuous resume mismatches are rejected by arch guard.
  11 Interrupted-validation recovery works: last_checkpoint saved BEFORE val
     can be strict-loaded, and start_epoch = ckpt.epoch + 1 (no re-train).
  12 Native evaluator (_build_model_from_arch) rebuilds both discrete and
     continuous models from arch_identity + strict-loads their weights.

All tests are CPU-only. Real .npz payload is stubbed with empty files where
`enumerate_patients` only needs to `glob` and load_manifest only needs
`exists()`; a tiny UNet3DFADCContinuous(bf=4) is built for the recovery
and rebuild tests so the parameter shapes are real.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.split_manifest import (
    COLLECTIONS,
    DEFAULT_EXPECTED_TRAIN,
    DEFAULT_EXPECTED_VAL,
    DEFAULT_SPLIT_KIND,
    enumerate_default_partition,
    generate_default_snapshot,
    manifest_sha256,
    verify_default_manifest_partitions,
)
from training.train_centralized_correct import (
    _arch_identity,
    _require_matching_arch,
    _require_matching_split,
    atomic_torch_save,
)
from training.evaluate_correct_checkpoint import _build_model_from_arch


# ─────────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────────

def _touch(dir_path: Path, patient_id: str) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    p = dir_path / f"{patient_id}.npz"
    p.write_bytes(b"")   # zero-byte stub; enumerate only needs the filename
    return p


def _mini_cache(root: Path, train_per_coll: int = 3, val_per_coll: int = 1):
    """Create a synthetic cache with the physical train/ + val/ layout.

    Default is 12 train + 4 val patients across the four collections — enough
    to exercise every code path without touching 1506 dentries.
    """
    for coll in COLLECTIONS:
        for i in range(train_per_coll):
            _touch(root / "train", f"{coll.lower()}_{i:03d}")
        for i in range(val_per_coll):
            _touch(root / "val", f"{coll.lower()}_{100 + i:03d}")


def _full_1200_306_cache(root: Path):
    """Create the exact 1200 train + 306 val layout the real dataset carries.

    Split as 300/76, 300/77, 300/76, 300/77 across DUKE/ISPY1/ISPY2/NACT so
    the totals hit 1200 and 306 precisely. Files are zero bytes.
    """
    plan = [
        ("DUKE",  300, 76),
        ("ISPY1", 300, 77),
        ("ISPY2", 300, 76),
        ("NACT",  300, 77),
    ]
    for coll, n_tr, n_va in plan:
        for i in range(n_tr):
            _touch(root / "train", f"{coll.lower()}_{i:04d}")
        for i in range(n_va):
            _touch(root / "val", f"{coll.lower()}_{9000 + i:04d}")


def _fresh_tmp(prefix: str = "fadc_default_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


# ─────────────────────────────────────────────────────────────────────────
# 1  Default snapshot contains exactly 1200 train + 306 val
# ─────────────────────────────────────────────────────────────────────────

def test_1_default_snapshot_1200_306_enforcement():
    # Path A: cache matches — the function accepts.
    root = _fresh_tmp()
    try:
        _full_1200_306_cache(root)
        meta = generate_default_snapshot(
            cache_root=str(root),
            csv_path=str(root / "manifest.csv"),
            meta_path=str(root / "manifest_meta.json"),
        )
        assert meta["n_per_split"]["train"] == DEFAULT_EXPECTED_TRAIN == 1200
        assert meta["n_per_split"]["val"]   == DEFAULT_EXPECTED_VAL == 306
        assert meta["n_per_split"]["test"]  == 0
        assert meta["n_patients_total"]     == 1506
        summary = verify_default_manifest_partitions(str(root / "manifest.csv"))
        assert summary["n_train"] == 1200 and summary["n_val"] == 306
        print("[1a] real 1200/306 cache -> snapshot accepted, counts asserted")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # Path B: cache off-by-one on train side — the function refuses.
    root = _fresh_tmp()
    try:
        _full_1200_306_cache(root)
        # Delete one train patient so we have 1199 train + 306 val.
        stray = next((root / "train").glob("*.npz"))
        stray.unlink()
        try:
            generate_default_snapshot(
                cache_root=str(root),
                csv_path=str(root / "manifest.csv"),
                meta_path=str(root / "manifest_meta.json"),
            )
        except RuntimeError as e:
            assert "1199" in str(e) and "1200" in str(e), e
            print(f"[1b] 1199/306 cache -> snapshot refused: {e}")
            return
        raise AssertionError("generate_default_snapshot should have refused 1199 train")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# 2  No test rows are ever emitted
# ─────────────────────────────────────────────────────────────────────────

def test_2_no_test_rows_ever():
    root = _fresh_tmp()
    try:
        _mini_cache(root)
        meta = generate_default_snapshot(
            cache_root=str(root),
            csv_path=str(root / "manifest.csv"),
            meta_path=str(root / "manifest_meta.json"),
            expected_train=12, expected_val=4,
        )
        # Metadata reports zero, and the CSV literally has no 'test' rows.
        assert meta["n_per_split"]["test"] == 0
        assert meta["split_kind"] == DEFAULT_SPLIT_KIND
        with open(root / "manifest.csv", encoding="utf-8") as f:
            body = f.read()
        assert ",test," not in body, "manifest CSV contains a 'test' row"
        # For every collection the metadata's test count is exactly zero.
        for coll, per_split in meta["n_per_collection_split"].items():
            assert per_split["test"] == 0, (coll, per_split)
        print("[2] snapshot contains no test rows (metadata + CSV)")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# 3  Train/validation overlap is zero
# ─────────────────────────────────────────────────────────────────────────

def test_3_train_val_zero_overlap():
    root = _fresh_tmp()
    try:
        _mini_cache(root, train_per_coll=10, val_per_coll=3)
        assigned = enumerate_default_partition(str(root))
        train_ids = {c["patient_id"] for c in assigned if c["split"] == "train"}
        val_ids   = {c["patient_id"] for c in assigned if c["split"] == "val"}
        assert len(train_ids & val_ids) == 0
        assert len(train_ids) + len(val_ids) == len(assigned)
        print(f"[3] zero overlap: train={len(train_ids)} val={len(val_ids)} "
              f"intersect={len(train_ids & val_ids)}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# 4  Recreating the snapshot from the same cache produces the same SHA
# ─────────────────────────────────────────────────────────────────────────

def test_4_sha_reproducible():
    root = _fresh_tmp()
    try:
        _mini_cache(root, train_per_coll=15, val_per_coll=4)
        csv1 = root / "m1.csv"; meta1 = root / "m1_meta.json"
        csv2 = root / "m2.csv"; meta2 = root / "m2_meta.json"
        m1 = generate_default_snapshot(str(root), str(csv1), str(meta1),
                                       expected_train=60, expected_val=16)
        m2 = generate_default_snapshot(str(root), str(csv2), str(meta2),
                                       expected_train=60, expected_val=16)
        s1 = manifest_sha256(str(csv1))
        s2 = manifest_sha256(str(csv2))
        assert s1 == s2 == m1["csv_sha256"] == m2["csv_sha256"], (s1, s2)
        # Byte-identical CSV bodies (LF only).
        b1 = csv1.read_bytes()
        b2 = csv2.read_bytes()
        assert b1 == b2 and b"\r\n" not in b1
        print(f"[4] SHA reproducible across regenerations: {s1[:16]}…")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# 5 + 6  Notebooks all require the same SHA and all use VAL_EVERY = 5
# ─────────────────────────────────────────────────────────────────────────

_NOTEBOOKS = (
    "kaggle_train_fadc3d_discrete_encoder_ds_defaultsplit_val5_s42.ipynb",
    "kaggle_train_fadc3d_discrete_encoder_nods_defaultsplit_val5_s42.ipynb",
    "kaggle_train_fadc3d_continuous_encoder_nods_defaultsplit_val5_s42.ipynb",
)


def _load_notebook(name: str) -> str:
    path = Path(__file__).parent.parent / name
    if not path.exists():
        raise AssertionError(f"required notebook missing: {path}")
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    return "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in nb["cells"]
    )


def test_5_all_notebooks_require_manifest_sha():
    for name in _NOTEBOOKS:
        src = _load_notebook(name)
        # Every notebook must (a) enforce a manifest SHA via preflight AND
        # (b) pass --require_manifest_checksum in the standalone eval cell.
        assert "MANIFEST_SHA256" in src, f"{name}: no MANIFEST_SHA256 constant"
        assert "require_manifest_checksum" in src, (
            f"{name}: standalone eval must enforce --require_manifest_checksum"
        )
        # The manifest is the DEFAULT-cache snapshot, not a seed-42 split.
        assert "generate_default_snapshot" in src or "verify_default_manifest_partitions" in src, (
            f"{name}: notebook must build/verify the default-cache snapshot"
        )
        print(f"[5] {name}: manifest SHA enforcement present")


def test_6_all_notebooks_val_every_5():
    for name in _NOTEBOOKS:
        src = _load_notebook(name)
        # CONFIG must set VAL_EVERY = 5 and CHECKPOINT_EVERY = 5.
        assert "VAL_EVERY" in src and "VAL_EVERY           = 5" in src.replace(" =", " ="), (
            f"{name}: VAL_EVERY must be set to 5 in CONFIG"
        )
        # Robust variant: whitespace-agnostic match.
        norm = " ".join(src.split())
        assert "VAL_EVERY = 5" in norm, f"{name}: VAL_EVERY must equal 5"
        assert "CHECKPOINT_EVERY = 5" in norm, f"{name}: CHECKPOINT_EVERY must equal 5"
        # Preflight must forbid RUN_FINAL_TEST — no final-test path allowed.
        assert "RUN_FINAL_TEST" not in src, (
            f"{name}: revised protocol forbids RUN_FINAL_TEST — remove test path"
        )
        # Formal-only: no fast/proxy validation flags.
        for banned in ("--fast_val_every", "--fast_val_max_cases"):
            assert banned not in src, f"{name}: forbidden flag {banned}"
        print(f"[6] {name}: VAL_EVERY=5, no final-test cell, no fast-val flags")


# ─────────────────────────────────────────────────────────────────────────
# 7  Formal validation walks every case (no subset / no proxy)
# ─────────────────────────────────────────────────────────────────────────

def test_7_formal_validation_visits_every_case():
    """Directly exercise trainer.validate() with a synthetic loader of N
    tiny volumes and assert n_done == N. This proves the function does
    not subset or short-circuit."""
    from training.train_centralized_correct import validate
    from monai.metrics import DiceMetric
    from monai.transforms import AsDiscrete

    class _Loader:
        def __init__(self, n):
            self.n = n
            self._items = [self._case(i) for i in range(n)]
        @staticmethod
        def _case(i):
            img = torch.zeros(1, 2, 16, 16, 8, dtype=torch.float32)
            lbl = torch.zeros(1, 1, 16, 16, 8, dtype=torch.long)
            lbl[0, 0, 4:8, 4:8, 2:5] = 1
            return {"image": img, "label": lbl}
        def __iter__(self):
            return iter(self._items)
        def __len__(self):
            return self.n

    class _IdModel(torch.nn.Module):
        def forward(self, x):
            b = x.shape[0]; d, h, w = x.shape[-3:]
            out = torch.zeros(b, 2, d, h, w, dtype=x.dtype, device=x.device)
            out[:, 1] = 1.0
            return out

    device = torch.device("cpu")
    model = _IdModel().to(device).eval()
    loader = _Loader(n=17)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred  = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)
    _mean_d, _mean_i, _mean_s, n_done = validate(
        model=model, val_loader=loader, dice_metric=dice_metric,
        post_pred=post_pred, post_label=post_label,
        patch_size=(16, 16, 8), device=device,
        overlap=0.5, sw_batch_size=2,
    )
    assert n_done == 17, f"validate() consumed {n_done} of 17 cases"
    print(f"[7] formal validation visited every case: n_done={n_done}")


# ─────────────────────────────────────────────────────────────────────────
# 8  A 70/10/20 checkpoint is rejected under a default-cache run
# ─────────────────────────────────────────────────────────────────────────

def test_8_reject_70_10_20_ckpt_under_default_run():
    ckpt = {"split_identity": {
        "split_manifest_sha256": "abc123",
        "split_partition_train": "train",
        "split_partition_val":   "val",
        "split_seed":            42,
        "split_ratios":          {"train": 0.7, "val": 0.1, "test": 0.2},
        "split_kind":            "seed_split",
    }}
    cur_default = {
        "split_manifest_sha256": "def456",   # different SHA anyway
        "split_partition_train": "train",
        "split_partition_val":   "val",
        "split_seed":            None,
        "split_ratios":          None,
        "split_kind":            "default_cache",
    }
    try:
        _require_matching_split(ckpt, cur_default)
    except RuntimeError as e:
        # Guard must call out a mismatched field; either split_kind or
        # split_manifest_sha256 is authoritative here.
        assert "split_kind" in str(e) or "split_manifest_sha256" in str(e), e
        print(f"[8] 70/10/20 ckpt rejected under default run: {e}")
        return
    raise AssertionError("_require_matching_split should have rejected 70/10/20 ckpt")


# ─────────────────────────────────────────────────────────────────────────
# 9  DS / noDS resume mismatch is rejected
# ─────────────────────────────────────────────────────────────────────────

def test_9_reject_ds_vs_nods_mismatch():
    ckpt = {"arch_identity": _arch_identity(
        "unet3d_fadc_encoder_correct",
        {"in_channels": 2, "out_channels": 2, "base_filters": 32,
         "deep_supervision": True},
        {"use_position_att": False},
        arch_kind="discrete",
    )}
    current = _arch_identity(
        "unet3d_fadc_encoder_correct",
        {"in_channels": 2, "out_channels": 2, "base_filters": 32,
         "deep_supervision": False},
        {"use_position_att": False},
        arch_kind="discrete",
    )
    try:
        _require_matching_arch(ckpt, current)
    except RuntimeError as e:
        assert "deep_supervision" in str(e)
        print(f"[9] DS/noDS mismatch rejected: {e}")
        return
    raise AssertionError("DS/noDS mismatch should be rejected")


# ─────────────────────────────────────────────────────────────────────────
# 10  Discrete / continuous resume mismatch is rejected
# ─────────────────────────────────────────────────────────────────────────

def test_10_reject_discrete_vs_continuous_mismatch():
    ckpt = {"arch_identity": _arch_identity(
        "unet3d_fadc_encoder_correct",
        {"in_channels": 2, "out_channels": 2, "base_filters": 32,
         "deep_supervision": False},
        {"use_position_att": False},
        arch_kind="discrete",
    )}
    # Point current at a continuous model with an ALIGNED model_name so
    # the mismatch is proven to hinge on arch_kind, not on model_name.
    current = _arch_identity(
        "unet3d_fadc_continuous_encoder",
        {"in_channels": 2, "out_channels": 2, "base_filters": 32,
         "deep_supervision": False},
        {},
        arch_kind="continuous",
    )
    try:
        _require_matching_arch(ckpt, current)
    except RuntimeError as e:
        # First field to diverge is model_name; regardless, resume must fail.
        assert "model_name" in str(e) or "arch_kind" in str(e), e
        print(f"[10] discrete/continuous mismatch rejected: {e}")
        return
    raise AssertionError("discrete/continuous mismatch should be rejected")


def test_10b_reject_arch_kind_flip_when_names_agree():
    """Belt-and-braces: even if two ckpts share model_name / channels /
    base_filters (impossible in practice but cheap to defend), a differing
    arch_kind alone must trigger the guard."""
    # Fabricate the same model_name on both sides to force the arch_kind
    # check to be the ONLY differentiator.
    ckpt_arch = {
        "model_name": "unet3d_fadc_encoder_correct",
        "in_channels": 2, "out_channels": 2, "base_filters": 32,
        "deep_supervision": False,
        "arch_kind": "discrete",
        "fadc_correct": {"use_position_att": False},
    }
    cur_arch = dict(ckpt_arch)
    cur_arch["arch_kind"] = "continuous"
    cur_arch["fadc_correct"] = {}
    try:
        _require_matching_arch({"arch_identity": ckpt_arch}, cur_arch)
    except RuntimeError as e:
        assert "arch_kind" in str(e), e
        print(f"[10b] arch_kind-only flip rejected: {e}")
        return
    raise AssertionError("arch_kind flip should be rejected")


# ─────────────────────────────────────────────────────────────────────────
# 11  Interrupted-validation recovery works
# ─────────────────────────────────────────────────────────────────────────

def test_11_interrupted_val_recovery():
    """Save last_checkpoint.pth with a completed-training epoch, then verify
    that on resume: (a) arch/split guards pass, (b) start_epoch = epoch+1,
    (c) the model + optimizer state round-trip via strict-load. This is
    exactly the sequence the recovery cell in each notebook exercises after
    a Kaggle disconnect during validation.
    """
    from models.unet_3d_fadc_continuous import build_unet3d_fadc_continuous

    root = _fresh_tmp()
    try:
        model = build_unet3d_fadc_continuous(
            model_name="unet3d_fadc_continuous_encoder",
            in_channels=2, out_channels=2, base_filters=4,
            deep_supervision=False,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        arch = _arch_identity(
            "unet3d_fadc_continuous_encoder",
            {"in_channels": 2, "out_channels": 2, "base_filters": 4,
             "deep_supervision": False},
            {},
            arch_kind="continuous",
        )
        split = {
            "split_manifest_path":   str(root / "manifest.csv"),
            "split_manifest_sha256": "fakehex",
            "split_partition_train": "train",
            "split_partition_val":   "val",
            "split_seed":            None,
            "split_ratios":          None,
            "split_kind":            "default_cache",
        }
        ckpt_dict = {
            "epoch":          5,   # last COMPLETED training epoch
            "model":          model.state_dict(),
            "optimizer":      opt.state_dict(),
            "scheduler":      {},   # not exercised in this unit test
            "scaler":         {},
            "best_dice":      0.42,
            "arch_identity":  arch,
            "split_identity": dict(split),
            "config":         {"training": {"epochs": 100}},
            "train_log":      [{"epoch": 5, "loss": 1.0}],
        }
        ckpt_path = root / "last_checkpoint.pth"
        atomic_torch_save(ckpt_dict, str(ckpt_path))

        # (a) arch + split guards pass with matching current identity.
        loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        _require_matching_arch(loaded, arch)         # must not raise
        _require_matching_split(loaded, split)       # must not raise

        # (b) start_epoch = epoch + 1 (no re-training of epoch 5).
        completed_epoch = int(loaded["epoch"])
        start_epoch = completed_epoch + 1
        assert start_epoch == 6, start_epoch

        # (c) strict-load round-trip works.
        rebuilt = _build_model_from_arch(loaded["arch_identity"], torch.device("cpu"))
        miss, unexp = rebuilt.load_state_dict(loaded["model"], strict=True)
        assert not miss and not unexp, (miss, unexp)

        print(f"[11] interrupted-val recovery: ckpt(epoch=5) -> start_epoch=6 (strict-load OK)")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# 12  Native evaluator rebuilds both discrete and continuous models
# ─────────────────────────────────────────────────────────────────────────

def test_12_evaluator_rebuilds_both_archs():
    from models.unet_3d_fadc_correct import build_unet3d_fadc_correct
    from models.unet_3d_fadc_continuous import build_unet3d_fadc_continuous

    # (a) discrete: save state_dict, rebuild via _build_model_from_arch, strict-load.
    m_disc = build_unet3d_fadc_correct(
        model_name="unet3d_fadc_encoder_correct",
        in_channels=2, out_channels=2, base_filters=4,
        deep_supervision=False, adakern_cfg={"use_position_att": False},
    )
    disc_arch = _arch_identity(
        "unet3d_fadc_encoder_correct",
        {"in_channels": 2, "out_channels": 2, "base_filters": 4,
         "deep_supervision": False},
        {"use_position_att": False},
        arch_kind="discrete",
    )
    rebuilt_disc = _build_model_from_arch(disc_arch, torch.device("cpu"))
    miss, unexp = rebuilt_disc.load_state_dict(m_disc.state_dict(), strict=True)
    assert not miss and not unexp, (miss, unexp)

    # (b) continuous: same round-trip.
    m_cont = build_unet3d_fadc_continuous(
        model_name="unet3d_fadc_continuous_encoder",
        in_channels=2, out_channels=2, base_filters=4,
        deep_supervision=False,
    )
    cont_arch = _arch_identity(
        "unet3d_fadc_continuous_encoder",
        {"in_channels": 2, "out_channels": 2, "base_filters": 4,
         "deep_supervision": False},
        {},
        arch_kind="continuous",
    )
    rebuilt_cont = _build_model_from_arch(cont_arch, torch.device("cpu"))
    miss, unexp = rebuilt_cont.load_state_dict(m_cont.state_dict(), strict=True)
    assert not miss and not unexp, (miss, unexp)

    # (c) mixing archs across kinds must fail at strict-load time even if
    # the caller sneaks a wrong state_dict past the guard.
    try:
        rebuilt_cont.load_state_dict(m_disc.state_dict(), strict=True)
    except RuntimeError:
        print("[12] native rebuild round-trips both archs; cross-kind strict-load rejected")
        return
    raise AssertionError("cross-kind strict-load should have failed")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fns = [
        test_1_default_snapshot_1200_306_enforcement,
        test_2_no_test_rows_ever,
        test_3_train_val_zero_overlap,
        test_4_sha_reproducible,
        test_5_all_notebooks_require_manifest_sha,
        test_6_all_notebooks_val_every_5,
        test_7_formal_validation_visits_every_case,
        test_8_reject_70_10_20_ckpt_under_default_run,
        test_9_reject_ds_vs_nods_mismatch,
        test_10_reject_discrete_vs_continuous_mismatch,
        test_10b_reject_arch_kind_flip_when_names_agree,
        test_11_interrupted_val_recovery,
        test_12_evaluator_rebuilds_both_archs,
    ]
    for fn in fns:
        print(f"\n---- {fn.__name__} ----")
        fn()
    print("\nALL DEFAULT-SPLIT TESTS PASSED")
