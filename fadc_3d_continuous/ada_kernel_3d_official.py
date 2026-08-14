"""AdaKern3DOfficial — dedicated two-trunk kernel-attention for the
continuous FADC3D package.

Motivation
==========

The discrete package's `fadc_3d_correct.AdaKern3D` uses ONE shared trunk to
predict all four (c_low, f_low, c_high, f_high) heads. The official 2D
reference (`FADC_only/conv_custom.py`, class `AdaptiveDilatedConv`, branch
`kernel_decompose='both'`) uses TWO independent `OmniAttention` trunks:

    OMNI_ATT1 : trunk_1 -> c_att1 (c_low), f_att1 (f_low)     [DC / smoothing]
    OMNI_ATT2 : trunk_2 -> c_att2 (c_high), f_att2 (f_high)   [high-freq residual]

Each trunk owns its own avgpool + fc + norm + relu chain, so the two
components do not share representation capacity. This module implements the
same layout in 3D so the continuous package matches the official topology.

The apply-to-kernel math is identical to the single-trunk variant:

    W_low  = W.mean(dim=(-3,-2,-1), keepdim=True)   # (O, I, 1, 1, 1)
    W_high = W - W_low                              # (O, I, 3, 3, 3)

    W_adaptive =
          W_low  * (2 * c_low ) * (2 * f_low )
        + W_high * (2 * c_high) * (2 * f_high)

Identity at init
================

- Every trunk uses its OWN Conv3d(in_channels, att_ch) + GroupNorm(1, att_ch)
  + ReLU. Trunks keep their Kaiming-init weights so gradients flow through
  them from the first backward.
- The OUTPUT heads (c_low_fc, f_low_fc, c_high_fc, f_high_fc, optional s_fc)
  are zero-initialised in both weight AND bias. sigmoid(0) = 0.5 →
  2 * sigmoid(0) = 1 → identity at forward-init.

At init the block is numerically identical to the single-trunk AdaKern3D,
so ContinuousDilatedConv3D preserves its "starts as a regular Conv3d"
guarantee when this module is dropped in.

Explicit non-features
=====================

- Only `groups=1` is supported (same as single-trunk). Grouped base kernels
  would need per-group attention broadcasting; the official code doesn't
  support it in this branch either.
"""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


class _OmniTrunk3D(nn.Module):
    """One official-style OmniAttention 3D trunk.

    Emits two attention tensors (c, f) from a pooled descriptor of the input
    feature map, plus an optional 27-position spatial attention `s`.

        c : (B, in_channels,  1, 1, 1)     -- input-channel gain
        f : (B, out_channels, 1, 1, 1)     -- output-filter gain
        s : (B, 27, 1, 1, 1) if use_spatial=True else None

    Output-head weights and biases are zero-init so 2*sigmoid(0) = 1 gives
    an identity multiplier at forward-init. The shared trunk (avgpool→fc→
    norm→relu) keeps Kaiming init so gradients still flow.
    """

    KERNEL_SIZE: int = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: float = 0.0625,
        min_channel: int = 16,
        use_spatial: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.use_spatial = bool(use_spatial)

        att_ch = max(int(min(in_channels, out_channels) * reduction), int(min_channel))
        self.att_ch = att_ch

        # -------- shared trunk (one per instance = one per component)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Conv3d(in_channels, att_ch, kernel_size=1, bias=False)
        # GroupNorm(1, att_ch): per-sample; safe with any B (avoids the
        # single-sample BN pathology from earlier iterations).
        self.norm = nn.GroupNorm(1, att_ch)
        self.relu = nn.ReLU(inplace=True)

        # -------- attention heads (zero-init)
        self.channel_fc = nn.Conv3d(att_ch, in_channels, kernel_size=1, bias=True)
        self.filter_fc  = nn.Conv3d(att_ch, out_channels, kernel_size=1, bias=True)
        self.spatial_fc = (nn.Conv3d(att_ch, self.KERNEL_SIZE ** 3, kernel_size=1, bias=True)
                           if self.use_spatial else None)
        self._zero_init_heads()

    def _zero_init_heads(self) -> None:
        for head in (self.channel_fc, self.filter_fc):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.spatial_fc is not None:
            nn.init.zeros_(self.spatial_fc.weight)
            nn.init.zeros_(self.spatial_fc.bias)

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if x.dim() != 5:
            raise ValueError(f"_OmniTrunk3D expects 5D input, got {tuple(x.shape)}")
        b = x.size(0)
        h = self.avgpool(x)             # (B, C_in, 1, 1, 1)
        h = self.fc(h)                  # (B, att_ch, 1, 1, 1)
        h = self.norm(h)
        h = self.relu(h)
        c = torch.sigmoid(self.channel_fc(h))    # (B, C_in, 1, 1, 1)
        f = torch.sigmoid(self.filter_fc(h))     # (B, C_out, 1, 1, 1)
        s = None
        if self.spatial_fc is not None:
            s_logits = self.spatial_fc(h)        # (B, 27, 1, 1, 1)
            s = torch.sigmoid(s_logits).view(
                b, 1, 1, self.KERNEL_SIZE, self.KERNEL_SIZE, self.KERNEL_SIZE
            )
        return c, f, s


class AdaKern3DOfficial(nn.Module):
    """Two-trunk official-style AdaKern for the continuous FADC3D package.

    Args:
        in_channels        : I. Matches the base kernel's second dim.
        out_channels       : O. Matches the base kernel's first dim.
        reduction          : bottleneck ratio for each trunk's descriptor.
        min_channel        : lower bound on each trunk's bottleneck width.
        use_position_att   : if True, trunk 2 also emits a 27-position
                             `s_att` applied AFTER the low+high reconstruction.
        groups             : only 1 is supported.
    """

    KERNEL_SIZE: int = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: float = 0.0625,
        min_channel: int = 16,
        use_position_att: bool = False,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if groups != 1:
            raise NotImplementedError(
                f"AdaKern3DOfficial only supports groups=1 today, got groups={groups}. "
                "Grouped base kernels need per-group attention broadcasting; "
                "silently doing so would misalign c/f channels."
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.use_position_att = bool(use_position_att)

        # Two independent trunks, one per kernel component.
        self.omni_low = _OmniTrunk3D(
            in_channels=in_channels, out_channels=out_channels,
            reduction=reduction, min_channel=min_channel,
            use_spatial=False,     # low-freq gets no spatial attention (matches paper).
        )
        self.omni_high = _OmniTrunk3D(
            in_channels=in_channels, out_channels=out_channels,
            reduction=reduction, min_channel=min_channel,
            use_spatial=use_position_att,
        )

    # --------------------------------------------------------------------- forward
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Predict (c_low, f_low, c_high, f_high, s_att) — same signature as
        the single-trunk `fadc_3d_correct.AdaKern3D.forward`."""
        c_low, f_low, _ = self.omni_low(x)
        c_high, f_high, s_att = self.omni_high(x)
        return c_low, f_low, c_high, f_high, s_att

    # --------------------------------------------------------------------- kernel builder
    @staticmethod
    def decompose_kernel(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split a base kernel W of shape (O, I, K, K, K) into (W_low, W_high).

            W_low  = W.mean(dim=(-3,-2,-1), keepdim=True)
            W_high = W - W_low
        """
        W_low = W.mean(dim=(-3, -2, -1), keepdim=True)
        W_high = W - W_low
        return W_low, W_high

    def apply_to_kernel(
        self,
        W: torch.Tensor,
        c_low: torch.Tensor,
        f_low: torch.Tensor,
        c_high: torch.Tensor,
        f_high: torch.Tensor,
        s_att: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build the per-sample adaptive kernel of shape (B, O, I, 3, 3, 3).

        Broadcasting layout (same as the single-trunk AdaKern3D):

            W_low        : (   1, O, I, 1, 1, 1)
            (2 * c_low)  : (B,    1, I, 1, 1, 1)
            (2 * f_low)  : (B,    O, 1, 1, 1, 1)
        """
        if W.dim() != 5:
            raise ValueError(f"W must be (O,I,K,K,K), got {tuple(W.shape)}")
        O, I, K1, K2, K3 = W.shape
        if K1 != self.KERNEL_SIZE or K2 != self.KERNEL_SIZE or K3 != self.KERNEL_SIZE:
            raise ValueError(f"Only 3x3x3 base kernels are supported, got {(K1, K2, K3)}")
        B = c_low.size(0)

        W_low, W_high = self.decompose_kernel(W)                # (O,I,1,1,1) / (O,I,K,K,K)
        W_low  = W_low.unsqueeze(0)                             # (1, O, I, 1, 1, 1)
        W_high = W_high.unsqueeze(0)                            # (1, O, I, K, K, K)

        c_low  = c_low.view(B, 1, I, 1, 1, 1)
        f_low  = f_low.view(B, O, 1, 1, 1, 1)
        c_high = c_high.view(B, 1, I, 1, 1, 1)
        f_high = f_high.view(B, O, 1, 1, 1, 1)

        term_low  = W_low  * (2 * c_low ) * (2 * f_low )        # (B, O, I, 1, 1, 1)
        term_high = W_high * (2 * c_high) * (2 * f_high)        # (B, O, I, K, K, K)
        W_adaptive = term_low + term_high                       # (B, O, I, K, K, K)

        if s_att is not None:
            W_adaptive = W_adaptive * (2 * s_att)               # (B, 1, 1, K, K, K)

        return W_adaptive


__all__ = ["AdaKern3DOfficial"]
