"""ContinuousDilatedConv3D — top-level block for the continuous FADC3D
implementation.

Mirrors the official 2D pipeline:

    x --(FrequencySelection3D)--> x_fs
    W_adaptive = AdaKern3D(x_fs, W)                    # per-sample kernel
    y = ContinuousAdaDR3D(x_fs, W_adaptive, bias)      # continuous-dilation conv

At initialization every subcomponent produces identity, so at forward-init
this block behaves numerically like a standard Conv3d with the same weight
and bias.

Not implemented in this block on purpose
========================================

- No auxiliary attention-diversity loss. The paper does not use one; the
  request forbids it.
- No `k_att` softmax over discrete dilation branches. The discrete
  `fadc_3d_correct.AdaptiveDilatedConv3D` still owns that mechanism and
  remains untouched.
- `pre_fs=True` only (matches AdaDR default). The `pre_fs=False` variant of
  the official code (which passes the offset back into FS) is not part of
  the continuous primary path.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from fadc_3d_correct.freq_select_3d import FrequencySelection3D
from fadc_3d_continuous.ada_kernel_3d_official import AdaKern3DOfficial
from fadc_3d_continuous.continuous_adadr_3d import (
    ContinuousAdaDR3D, CONTINUOUS_ADADR3D_META,
)


class ContinuousDilatedConv3D(nn.Module):
    """Drop-in replacement for a Conv3d(kernel=3, padding=1) that uses the
    continuous-dilation AdaDR3D mechanism inside.

    Args:
        in_channels, out_channels : as Conv3d.
        bias                      : whether to include a bias term.
        base_dilation             : base dilation D_0 (inherited from the
                                    conv being replaced). Feeds
                                    conv_offset init and abs() scaling.
        deform_groups             : G. 1 is the primary path.
        padding_mode              : 'border' (default; matches official) or 'zeros'.
        align_corners             : True (default).
        chunk_positions           : sequential-chunk size for the 27
                                    kernel-position sampling loop.
                                    1 = minimum memory (default).
        fs_cfg                    : forwarded to FrequencySelection3D. None
                                    disables FreqSelect.
        adakern_reduction         : AdaKern3D reduction ratio.
        adakern_min_channel       : AdaKern3D min-channel width.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        base_dilation: int = 1,
        deform_groups: int = 1,
        padding_mode: str = "border",
        align_corners: bool = True,
        chunk_positions: int = 1,
        fs_cfg: Optional[dict] = None,
        adakern_reduction: float = 0.0625,
        adakern_min_channel: int = 16,
        use_position_att: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_dilation = int(base_dilation)

        # Base kernel W of shape (O, I, 3, 3, 3). Kaiming-init (matches Conv3d).
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, 3, 3, 3)
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # Sub-modules.
        if fs_cfg is None:
            fs_cfg = {"k_list": (2, 4, 8), "lowfreq_att": False,
                      "lp_type": "freq", "act": "sigmoid",
                      "spatial_group": 1, "spatial_kernel": 3}
        self.fs = FrequencySelection3D(in_channels=in_channels, **fs_cfg)
        # Two-trunk official-style AdaKern (matches the reference 2D layout).
        self.ada_kern = AdaKern3DOfficial(
            in_channels=in_channels, out_channels=out_channels,
            reduction=adakern_reduction, min_channel=adakern_min_channel,
            use_position_att=use_position_att, groups=1,
        )
        self.adadr = ContinuousAdaDR3D(
            in_channels=in_channels, out_channels=out_channels,
            deform_groups=deform_groups, base_dilation=base_dilation,
            padding_mode=padding_mode, align_corners=align_corners,
            chunk_positions=chunk_positions,
        )

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Frequency selection (identity at init).
        x_fs = self.fs(x)
        # 2) AdaKern — build per-sample adaptive kernel.
        c_low, f_low, c_high, f_high, s_att = self.ada_kern(x_fs)
        W_adaptive = self.ada_kern.apply_to_kernel(
            self.weight, c_low, f_low, c_high, f_high, s_att,
        )                                                     # (B, O, I, 3, 3, 3)
        # 3) Continuous-dilation modulated conv.
        return self.adadr(x_fs, W_adaptive, self.bias)


# Re-export the metadata dict for use by the model + notebook.
__all__ = ["ContinuousDilatedConv3D", "CONTINUOUS_ADADR3D_META"]
