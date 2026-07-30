"""Contract tests for training/patch_validation.py + a small forward pass.

Run on CPU. No CUDA, no dataset, no MONAI transforms.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch

from training.patch_validation import validate_patch_size, REQUIRED_DIVISOR


# ─────────────────────────────────────────────────────────────────────────
# 1. Divisibility contract
# ─────────────────────────────────────────────────────────────────────────

def _expect_reject(ps, why: str) -> None:
    try:
        validate_patch_size(ps)
    except ValueError as e:
        assert str(ps[-1]) in str(e) or str(ps) in str(e) or "dim" in str(e), \
            f"error message does not identify the offender: {e!r}"
        return
    raise AssertionError(f"expected {ps} to be rejected because {why}")


def _expect_accept(ps) -> None:
    out = validate_patch_size(ps)
    assert out == tuple(ps), (out, ps)


def test_rejects_smoke_patch_that_broke_prod() -> None:
    # The exact patch that made continuous smoke fail in cell 8 (Kaggle).
    _expect_reject((48, 48, 24), why="depth 24 is not a multiple of 16")


def test_rejects_other_common_offenders() -> None:
    _expect_reject((48, 48, 8),  why="depth 8")
    _expect_reject((24, 48, 32), why="dim 0 = 24")
    _expect_reject((16, 20, 16), why="dim 1 = 20")
    _expect_reject((0, 16, 16),  why="dim 0 must be > 0")
    _expect_reject((-16, 16, 16), why="negative dim")


def test_accepts_the_fixed_smoke_patch() -> None:
    _expect_accept([48, 48, 32])


def test_accepts_production_patch() -> None:
    _expect_accept([128, 128, 64])


def test_accepts_smallest_multiple_of_divisor() -> None:
    _expect_accept([REQUIRED_DIVISOR] * 3)  # [16, 16, 16]


def test_wrong_rank_raises() -> None:
    try:
        validate_patch_size([64, 64])
    except ValueError as e:
        assert "3-tuple" in str(e)
        return
    raise AssertionError("expected ValueError for 2-element patch")


# ─────────────────────────────────────────────────────────────────────────
# 2. Continuous model forward pass at the smallest valid shape
#
# Ensures the model actually runs end-to-end on a div-16 patch and the
# output spatial shape equals the input spatial shape. Uses base_filters=4
# to keep CPU wall time small; the shape contract does not depend on width.
# ─────────────────────────────────────────────────────────────────────────

def test_continuous_forward_smallest_shape_preserves_spatial() -> None:
    from models.unet_3d_fadc_continuous import build_unet3d_fadc_continuous

    torch.manual_seed(0)
    model = build_unet3d_fadc_continuous(
        model_name="unet3d_fadc_continuous_encoder",
        in_channels=2, out_channels=2, base_filters=4,
        deep_supervision=False,
    ).eval()
    x = torch.randn(1, 2, 16, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 2, 16, 16, 16), y.shape


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"\n{len(fns)} test(s) passed.")
