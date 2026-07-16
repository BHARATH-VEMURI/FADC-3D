"""
Laptop-side smoke test for the FL pipeline.

Runs the full orchestrator end-to-end (partition → per-client local train →
aggregate → global+per-client validation → checkpoint) on synthetic .npz data
and a tiny UNet3D. Pure CPU, no Kaggle, no real MAMA-MIA cache.

This catches:
  - import / wiring bugs
  - state_dict aggregation shape/key mismatches
  - FedBN BN-key skipping
  - per-client / global eval bookkeeping

Run from FADC-main/:
  python -m pytest tests/test_fl_loop.py -v -s
or just:
  python tests/test_fl_loop.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))
from data.mama_mia_dataset import CLIENT_MAP
from models.unet_3d import UNet3D
from training.fl_algorithms import (
    fedavg_aggregate,
    fedbn_aggregate,
    get_bn_state_dict_keys,
)
from training.fl_partitions import (
    iid_random_partition,
    natural_partition,
    partition_summary,
)
from training.losses import DiceCELoss
from training.train_federated import run_fl


PATCH = (32, 32, 16)
TINY_F = 4   # base_filters — keep model tiny enough for CPU


def _make_synthetic_npz(path: Path, shape=(2, 48, 48, 24), seed=0):
    """Write a 2-channel image + binary label with a guaranteed FG region.

    RandCropByPosNegLabeld requires labels with at least one positive voxel, so
    we stamp a fixed cube of 1s in every label.
    """
    rng = np.random.default_rng(seed)
    image = rng.standard_normal(shape).astype(np.float16)
    label = np.zeros((1,) + shape[1:], dtype=np.uint8)
    label[:, 10:25, 10:25, 5:15] = 1  # guaranteed tumor cube
    np.savez_compressed(path, image=image, label=label)


def _build_cache(root: Path, train_per_client=4, val_per_client=2):
    """Create train/ and val/ .npz files matching real-data naming."""
    train_dir = root / "train"
    val_dir   = root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_cases, val_cases = [], []
    seed = 0
    for collection, client_id in CLIENT_MAP.items():
        for i in range(train_per_client):
            pid = f"{collection.lower()}_{i:03d}"
            _make_synthetic_npz(train_dir / f"{pid}.npz", seed=seed); seed += 1
            train_cases.append({
                "patient_id": pid, "collection": collection, "client_id": client_id,
            })
        for i in range(val_per_client):
            pid = f"{collection.lower()}_v{i:03d}"
            _make_synthetic_npz(val_dir / f"{pid}.npz", seed=seed); seed += 1
            val_cases.append({
                "patient_id": pid, "collection": collection, "client_id": client_id,
            })
    return train_cases, val_cases


def test_partitions_basic():
    train_cases, _ = _build_cache(Path(tempfile.mkdtemp(prefix="fl_smoke_partitions_")))
    nat = natural_partition(train_cases)
    iid = iid_random_partition(train_cases, n_clients=4, seed=42)
    assert sum(len(v) for v in nat.values()) == len(train_cases)
    assert sum(len(v) for v in iid.values()) == len(train_cases)
    # Natural: each client should hold exactly one collection
    for cid, cases in nat.items():
        assert len({c["collection"] for c in cases}) == 1
    print("\n[partitions] natural OK | iid OK")
    print(partition_summary(nat))


def test_bn_key_detection():
    model = UNet3D(in_channels=2, out_channels=2, base_filters=TINY_F)
    bn_keys = get_bn_state_dict_keys(model)
    # UNet3D has BN in every ConvBlock — 9 blocks (4 enc + 4 dec + 1 bn) x 2 BN/block = 18 BN modules
    # Each contributes (weight, bias, running_mean, running_var, num_batches_tracked) = 5 keys
    assert len(bn_keys) > 0
    # All keys must point at real BN modules — sanity check by loading a state_dict
    state = model.state_dict()
    for k in bn_keys:
        assert k in state, f"BN key {k} missing from state_dict"
    print(f"\n[bn_keys] detected {len(bn_keys)} BN state_dict keys")


def test_fedavg_aggregation_preserves_shapes():
    model = UNet3D(in_channels=2, out_channels=2, base_filters=TINY_F)
    s_a = {k: v.clone() for k, v in model.state_dict().items()}
    s_b = {k: v.clone() for k, v in model.state_dict().items()}
    agg = fedavg_aggregate([s_a, s_b], [10, 20])
    for k, v in model.state_dict().items():
        assert agg[k].shape == v.shape
        assert agg[k].dtype == v.dtype
    model.load_state_dict(agg)  # must load cleanly
    print("\n[fedavg] aggregate loads cleanly into model")


def test_fedbn_skips_bn_keys():
    model = UNet3D(in_channels=2, out_channels=2, base_filters=TINY_F)
    prior = {k: v.clone() for k, v in model.state_dict().items()}
    # Perturb both clients' BN keys; conv weights identical
    s_a = {k: v.clone() for k, v in model.state_dict().items()}
    s_b = {k: v.clone() for k, v in model.state_dict().items()}
    bn_keys = get_bn_state_dict_keys(model)
    for k in bn_keys:
        if s_a[k].dtype.is_floating_point:
            s_a[k] = s_a[k] + 99.0
            s_b[k] = s_b[k] - 99.0

    agg = fedbn_aggregate([s_a, s_b], [10, 20], bn_keys, prior)
    for k in bn_keys:
        assert torch.equal(agg[k], prior[k]), f"FedBN leaked BN key: {k}"
    print(f"\n[fedbn] {len(bn_keys)} BN keys correctly preserved as prior_global_state")


def test_full_fl_loop_fedavg():
    tmp_root = Path(tempfile.mkdtemp(prefix="fl_smoke_loop_"))
    cache_root = tmp_root / "cache"
    out_dir    = tmp_root / "out"
    try:
        train_cases, val_cases = _build_cache(cache_root, train_per_client=4, val_per_client=2)
        partition = natural_partition(train_cases)

        def model_fn():
            return UNet3D(in_channels=2, out_channels=2, base_filters=TINY_F)
        loss_fn = DiceCELoss(dice_weight=0.5, ce_weight=0.5)

        run_fl(
            model_fn=model_fn,
            loss_fn=loss_fn,
            partition=partition,
            val_cases=val_cases,
            train_cache_dir=str(cache_root / "train"),
            val_cache_dir=str(cache_root / "val"),
            algorithm="fedavg",
            rounds=2,
            local_epochs=1,
            lr=1e-3,
            client_fraction=1.0,
            batch_size=1,
            num_workers=0,
            patch_size=PATCH,
            output_dir=str(out_dir),
            seed=42,
            save_every=1,
        )

        # Checkpoint + log written
        assert (out_dir / "best_model.pth").exists()
        assert (out_dir / "fl_log.json").exists()
        assert (out_dir / "meta.json").exists()

        log = json.loads((out_dir / "fl_log.json").read_text())
        assert len(log) == 2
        for r in log:
            assert "global" in r and "per_client" in r
            assert len(r["per_client"]) == 4   # 4 natural clients in val
        print(f"\n[fedavg loop] 2 rounds completed. Best Dice: {log[-1]['global']['dice']:.4f}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_full_fl_loop_fedbn():
    tmp_root = Path(tempfile.mkdtemp(prefix="fl_smoke_loop_bn_"))
    cache_root = tmp_root / "cache"
    out_dir    = tmp_root / "out"
    try:
        train_cases, val_cases = _build_cache(cache_root, train_per_client=4, val_per_client=2)
        partition = natural_partition(train_cases)

        def model_fn():
            return UNet3D(in_channels=2, out_channels=2, base_filters=TINY_F)
        loss_fn = DiceCELoss(dice_weight=0.5, ce_weight=0.5)

        run_fl(
            model_fn=model_fn,
            loss_fn=loss_fn,
            partition=partition,
            val_cases=val_cases,
            train_cache_dir=str(cache_root / "train"),
            val_cache_dir=str(cache_root / "val"),
            algorithm="fedbn",
            rounds=2,
            local_epochs=1,
            lr=1e-3,
            client_fraction=0.5,  # exercise client_fraction sampling too
            batch_size=1,
            num_workers=0,
            patch_size=PATCH,
            output_dir=str(out_dir),
            seed=42,
        )
        log = json.loads((out_dir / "fl_log.json").read_text())
        # client_fraction=0.5 with 4 clients should yield 2 participants per round
        for r in log:
            assert len(r["participants"]) == 2
        print(f"\n[fedbn loop] 2 rounds completed with client_fraction=0.5")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    test_partitions_basic()
    test_bn_key_detection()
    test_fedavg_aggregation_preserves_shapes()
    test_fedbn_skips_bn_keys()
    test_full_fl_loop_fedavg()
    test_full_fl_loop_fedbn()
    print("\nALL SMOKE TESTS PASSED.")
