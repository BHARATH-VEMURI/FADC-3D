"""Shape-invariant enforcement for the FADC-3D UNet family.

Every UNet3D variant in this repo has four MaxPool3d(kernel=2, stride=2)
stages in the encoder and four ConvTranspose3d(stride=2) stages in the
decoder. That means every spatial input dimension must be divisible by
2**4 = 16 for skip connections at each decoder stage to align without
implicit interpolation or cropping. When a dimension is not divisible by
16, floor-division on the way down produces a bottleneck that does not
symmetrically double back up (e.g. 24 -> 12 -> 6 -> 3 -> 1, and 1 * 2 = 2
does not match the depth-3 skip), so torch.cat raises a RuntimeError deep
inside the decoder.

This module centralises the check so trainer, evaluator, and any future
inference entry point reject a bad patch BEFORE constructing loaders or
starting training, with a message that pinpoints the offending dimension.
"""
from __future__ import annotations

from typing import Iterable

UNET_POOL_STAGES = 4
REQUIRED_DIVISOR = 2 ** UNET_POOL_STAGES  # 16


def validate_patch_size(patch_size: Iterable[int], *, name: str = "patch_size") -> tuple[int, int, int]:
    """Return a 3-tuple after asserting the shape contract.

    Raises ValueError with a message identifying the offending value(s).
    """
    ps = tuple(int(v) for v in patch_size)
    if len(ps) != 3:
        raise ValueError(
            f"{name} must be a 3-tuple (D, H, W); got {ps!r} with length {len(ps)}."
        )
    bad = [(i, v) for i, v in enumerate(ps) if v <= 0 or (v % REQUIRED_DIVISOR) != 0]
    if bad:
        offending = ", ".join(f"dim {i}={v}" for i, v in bad)
        raise ValueError(
            f"Invalid {name}={list(ps)!r}: {offending}. "
            f"The FADC-3D UNet has {UNET_POOL_STAGES} pooling stages, so every "
            f"spatial dimension must be a positive multiple of {REQUIRED_DIVISOR}. "
            f"Fix the caller's patch size (e.g. round 24 up to 32 or down to 16)."
        )
    return ps
