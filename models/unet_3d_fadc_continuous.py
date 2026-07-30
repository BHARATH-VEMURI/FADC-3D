"""UNet3D with continuous FADC3D encoder — 8 adaptive blocks.

Mirrors the discrete `models/unet_3d_fadc_correct.py::UNet3DFADCCorrect`
architecture (channel widths, pooling, decoder wiring) but replaces the
two `AdaptiveDilatedConv3D` modules in every encoder block with
`ContinuousDilatedConv3D`. Bottleneck + decoder stay plain Conv3d.

Design contract (see the request for the full rationale):
  - Encoder placement only.
  - Deep supervision DISABLED. The primary segmentation head is the only
    output — `forward` always returns a single tensor, never a tuple.
  - Preserves the discrete package as a reproducible baseline (this file
    imports FROM but never modifies fadc_3d_correct or fadc_3d_continuous).
  - Base-dilation = 1 (inherited from the plain-UNet Conv3D it replaces).

The model NAME registered here is `unet3d_fadc_continuous_encoder` — a
different identity from the discrete `unet3d_fadc_encoder_correct`.
Checkpoints from one arch will not load into the other under strict=True.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fadc_3d_continuous import (
    ContinuousDilatedConv3D, ContinuousAdaDR3D, CONTINUOUS_ADADR3D_META,
)


# ─────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────


class ContinuousFADCConvBlock(nn.Module):
    """Two continuous-dilation adaptive convs + BN + ReLU + residual skip.

    Mirrors `FADCConvBlockCorrect` in the discrete package. Kernel-side and
    frequency-selection identity guarantees still hold at init because the
    reused FreqSelection3D + AdaKern3D preserve their zero-logit init.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout: float = 0.15,
        chunk_positions: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = ContinuousDilatedConv3D(
            in_channels=in_ch, out_channels=out_ch, bias=False,
            base_dilation=1, chunk_positions=chunk_positions,
        )
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.act1 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(p=dropout)
        self.conv2 = ContinuousDilatedConv3D(
            in_channels=out_ch, out_channels=out_ch, bias=False,
            base_dilation=1, chunk_positions=chunk_positions,
        )
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.act2 = nn.ReLU(inplace=True)

        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        h = self.act1(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return self.act2(h + identity)


class PlainConvBlock(nn.Module):
    """Standard double 3x3x3 Conv3d + BN + ReLU. Bottleneck + decoder."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_fadc: bool, chunk_positions: int) -> None:
        super().__init__()
        if use_fadc:
            self.conv = ContinuousFADCConvBlock(in_ch, out_ch, chunk_positions=chunk_positions)
        else:
            self.conv = PlainConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = PlainConvBlock(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────


class UNet3DFADCContinuous(nn.Module):
    """Encoder-only continuous-FADC3D UNet.

    Encoder (all continuous FADC blocks):
        enc1: in_channels -> base_filters             (2 continuous convs)
        enc2: base_filters -> base_filters * 2        (2 continuous convs)
        enc3: base_filters * 2 -> base_filters * 4    (2 continuous convs)
        enc4: base_filters * 4 -> base_filters * 8    (2 continuous convs)

    Bottleneck + decoder: plain Conv3D. DS DISABLED.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_filters: int = 32,
        deep_supervision: bool = False,
        chunk_positions: int = 1,
    ) -> None:
        super().__init__()
        if deep_supervision:
            raise NotImplementedError(
                "The continuous FADC3D encoder-only variant is DS-off by "
                "contract; pass deep_supervision=False. If you need a DS "
                "variant, add a new model registration — do not silently "
                "toggle DS here."
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_filters = int(base_filters)
        self.deep_supervision = False
        self.chunk_positions = int(chunk_positions)

        bf = self.base_filters

        # Encoder: 4 continuous FADC blocks.
        self.enc1 = DownBlock(in_channels, bf,      use_fadc=True,  chunk_positions=chunk_positions)
        self.enc2 = DownBlock(bf,          bf * 2,  use_fadc=True,  chunk_positions=chunk_positions)
        self.enc3 = DownBlock(bf * 2,      bf * 4,  use_fadc=True,  chunk_positions=chunk_positions)
        self.enc4 = DownBlock(bf * 4,      bf * 8,  use_fadc=True,  chunk_positions=chunk_positions)

        # Bottleneck: plain Conv3D.
        self.bottleneck = PlainConvBlock(bf * 8, bf * 16)

        # Decoder: plain Conv3D.
        self.dec4 = UpBlock(bf * 16, bf * 8)
        self.dec3 = UpBlock(bf * 8,  bf * 4)
        self.dec2 = UpBlock(bf * 4,  bf * 2)
        self.dec1 = UpBlock(bf * 2,  bf)

        # Output head: 1x1x1 conv.
        self.head = nn.Conv3d(bf, out_channels, kernel_size=1)

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)                                       # (B, out_channels, D, H, W)

    # ---------------------------------------------------------------- helpers
    def count_adaptive_blocks(self) -> int:
        """Number of `ContinuousDilatedConv3D` modules in this model."""
        return sum(1 for m in self.modules() if isinstance(m, ContinuousDilatedConv3D))

    def adaptive_block_names(self) -> list[str]:
        return [name for name, m in self.named_modules()
                if isinstance(m, ContinuousDilatedConv3D)]

    def arch_identity(self) -> dict:
        """Serialisable architecture identity for checkpoint verification."""
        return {
            "model_name":         "unet3d_fadc_continuous_encoder",
            "in_channels":        int(self.in_channels),
            "out_channels":       int(self.out_channels),
            "base_filters":       int(self.base_filters),
            "deep_supervision":   False,
            "adaptive_blocks":    int(self.count_adaptive_blocks()),
            **{k: v for k, v in CONTINUOUS_ADADR3D_META.items()},
        }


# ─────────────────────────────────────────────────────────────────────────
# Factory + name registration
# ─────────────────────────────────────────────────────────────────────────

MODEL_NAMES = ("unet3d_fadc_continuous_encoder",)
EXPECTED_ADAPTIVE_BLOCK_COUNT = {"encoder": 8}


def build_unet3d_fadc_continuous(
    model_name: str = "unet3d_fadc_continuous_encoder",
    in_channels: int = 2,
    out_channels: int = 2,
    base_filters: int = 32,
    deep_supervision: bool = False,
    chunk_positions: int = 1,
) -> UNet3DFADCContinuous:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; expected one of {MODEL_NAMES}")
    return UNet3DFADCContinuous(
        in_channels=in_channels, out_channels=out_channels,
        base_filters=base_filters, deep_supervision=deep_supervision,
        chunk_positions=chunk_positions,
    )
