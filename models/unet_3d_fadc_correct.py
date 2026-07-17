"""UNet3DFADCCorrect — 3D U-Net using the CORRECTED FADC block.

Placements (mirrors historical UNet3DFADC_V2 semantics but restricted to
the four names required for this experiment):

    encoder    -> corrected FADC in enc1..enc4 only    (8 adaptive convs)
    decoder    -> corrected FADC in dec4..dec1 only    (8 adaptive convs)
    bottleneck -> corrected FADC in bottleneck only    (2 adaptive convs)
    full       -> encoder + bottleneck + decoder       (18 adaptive convs)

For every placement the same double-conv block is used
(`FADCConvBlockCorrect`), containing two `AdaptiveDilatedConv3D` modules.
Non-FADC positions use a plain double-Conv3d block, unchanged from
`models/unet_3d.py`.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fadc_3d_correct.adaptive_dilated_conv_3d import AdaptiveDilatedConv3D


# ------------------------------------------------------------------ blocks


class FADCConvBlockCorrect(nn.Module):
    """Double corrected-FADC block: FADC -> BN -> ReLU -> FADC -> BN -> ReLU + skip.

    The corrected FADC block already returns a per-voxel-mixed feature; there
    is no post-BN f_att re-multiplication (kernel-side f attention is baked
    into W_adaptive, not into the feature).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout: float = 0.15,
        fs_cfg: Optional[dict] = None,
        adakern_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.conv1 = AdaptiveDilatedConv3D(
            in_channels=in_ch, out_channels=out_ch, bias=False,
            fs_cfg=fs_cfg, adakern_cfg=adakern_cfg,
        )
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.act1 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(p=dropout)
        self.conv2 = AdaptiveDilatedConv3D(
            in_channels=out_ch, out_channels=out_ch, bias=False,
            fs_cfg=fs_cfg, adakern_cfg=adakern_cfg,
        )
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.act2 = nn.ReLU(inplace=True)

        # Residual connection at the block level.
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
    """Standard double 3x3x3 Conv3d + BN + ReLU + residual (matches UNet3D style)."""

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


def _make_block(
    in_ch: int, out_ch: int, use_fadc: bool,
    fs_cfg: Optional[dict], adakern_cfg: Optional[dict],
) -> nn.Module:
    if use_fadc:
        return FADCConvBlockCorrect(in_ch, out_ch, fs_cfg=fs_cfg, adakern_cfg=adakern_cfg)
    return PlainConvBlock(in_ch, out_ch)


class DownBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, use_fadc: bool,
        fs_cfg: Optional[dict], adakern_cfg: Optional[dict],
    ) -> None:
        super().__init__()
        self.conv = _make_block(in_ch, out_ch, use_fadc, fs_cfg, adakern_cfg)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, use_fadc: bool,
        fs_cfg: Optional[dict], adakern_cfg: Optional[dict],
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        # Concat doubles the input channels after the up-conv.
        self.conv = _make_block(in_ch, out_ch, use_fadc, fs_cfg, adakern_cfg)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ------------------------------------------------------------------ model


_PLACEMENTS = ("encoder", "decoder", "bottleneck", "full")

_MODEL_NAME_TO_PLACEMENT = {
    "unet3d_fadc_encoder_correct":    "encoder",
    "unet3d_fadc_decoder_correct":    "decoder",
    "unet3d_fadc_bottleneck_correct": "bottleneck",
    "unet3d_fadc_full_correct":       "full",
}


class UNet3DFADCCorrect(nn.Module):
    """3D U-Net with the corrected FADC block at a chosen placement.

    Args:
        in_channels      : image channels (2 = pre + post).
        out_channels     : 2 for binary segmentation.
        base_filters     : channels at the first encoder level (default 32).
        fadc_placement   : one of {'encoder','decoder','bottleneck','full'}.
        deep_supervision : if True, emit auxiliary heads at dec2/dec3/dec4 during train().
        fs_cfg           : override for FrequencySelection3D kwargs (None -> spec defaults).
        adakern_cfg      : override for AdaKern3D kwargs (None -> spec defaults, s_att OFF).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_filters: int = 32,
        fadc_placement: str = "encoder",
        deep_supervision: bool = False,
        fs_cfg: Optional[dict] = None,
        adakern_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if fadc_placement not in _PLACEMENTS:
            raise ValueError(
                f"unknown fadc_placement {fadc_placement!r}, expected one of {_PLACEMENTS}"
            )

        self.fadc_placement = fadc_placement
        self.deep_supervision = bool(deep_supervision)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_filters = int(base_filters)

        enc_fadc = fadc_placement in ("encoder", "full")
        bn_fadc = fadc_placement in ("bottleneck", "full")
        dec_fadc = fadc_placement in ("decoder", "full")

        f = base_filters

        self.enc1 = DownBlock(in_channels, f,     enc_fadc, fs_cfg, adakern_cfg)
        self.enc2 = DownBlock(f,           f * 2, enc_fadc, fs_cfg, adakern_cfg)
        self.enc3 = DownBlock(f * 2,       f * 4, enc_fadc, fs_cfg, adakern_cfg)
        self.enc4 = DownBlock(f * 4,       f * 8, enc_fadc, fs_cfg, adakern_cfg)

        self.bottleneck = _make_block(f * 8, f * 16, bn_fadc, fs_cfg, adakern_cfg)

        self.dec4 = UpBlock(f * 16, f * 8, dec_fadc, fs_cfg, adakern_cfg)
        self.dec3 = UpBlock(f * 8,  f * 4, dec_fadc, fs_cfg, adakern_cfg)
        self.dec2 = UpBlock(f * 4,  f * 2, dec_fadc, fs_cfg, adakern_cfg)
        self.dec1 = UpBlock(f * 2,  f,     dec_fadc, fs_cfg, adakern_cfg)

        self.head = nn.Conv3d(f, out_channels, kernel_size=1)

        if self.deep_supervision:
            self.ds_head2 = nn.Conv3d(f * 2, out_channels, kernel_size=1)
            self.ds_head3 = nn.Conv3d(f * 4, out_channels, kernel_size=1)
            self.ds_head4 = nn.Conv3d(f * 8, out_channels, kernel_size=1)

    # ---------------------------------------------------------- API
    def set_temperature(self, t: float) -> None:
        """Cascade a new k_att softmax temperature to every AdaptiveDilatedConv3D."""
        for m in self.modules():
            if isinstance(m, AdaptiveDilatedConv3D):
                m.set_temperature(t)

    def count_adaptive_convs(self) -> int:
        """How many corrected AdaptiveDilatedConv3D modules the model contains."""
        return sum(1 for m in self.modules() if isinstance(m, AdaptiveDilatedConv3D))

    def adaptive_conv_names(self) -> list[str]:
        return [n for n, m in self.named_modules() if isinstance(m, AdaptiveDilatedConv3D)]

    # ---------------------------------------------------------- forward
    def forward(self, x: torch.Tensor):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        x = self.bottleneck(x)

        d4 = self.dec4(x, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        main = self.head(d1)
        if self.deep_supervision and self.training:
            return main, self.ds_head2(d2), self.ds_head3(d3), self.ds_head4(d4)
        return main


# ------------------------------------------------------------------ factory


def build_unet3d_fadc_correct(
    model_name: str,
    in_channels: int = 2,
    out_channels: int = 2,
    base_filters: int = 32,
    deep_supervision: bool = False,
    fs_cfg: Optional[dict] = None,
    adakern_cfg: Optional[dict] = None,
) -> UNet3DFADCCorrect:
    """Instantiate one of the four registered corrected-FADC models by name."""
    if model_name not in _MODEL_NAME_TO_PLACEMENT:
        raise ValueError(
            f"Unknown model_name {model_name!r}. "
            f"Expected one of {list(_MODEL_NAME_TO_PLACEMENT)}"
        )
    placement = _MODEL_NAME_TO_PLACEMENT[model_name]
    return UNet3DFADCCorrect(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=base_filters,
        fadc_placement=placement,
        deep_supervision=deep_supervision,
        fs_cfg=fs_cfg,
        adakern_cfg=adakern_cfg,
    )


MODEL_NAMES = tuple(_MODEL_NAME_TO_PLACEMENT.keys())


EXPECTED_ADAPTIVE_CONV_COUNT = {
    "encoder":    8,
    "decoder":    8,
    "bottleneck": 2,
    "full":       18,
}
