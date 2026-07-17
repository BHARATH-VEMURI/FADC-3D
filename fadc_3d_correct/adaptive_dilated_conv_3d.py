"""AdaptiveDilatedConv3D (corrected) — one shared base kernel, discrete 3D dilation.

Data flow
---------
    input x
        -> FrequencySelection3D
        -> AdaKern3D predicts (c_low, f_low, c_high, f_high, s_att)
        -> W_adaptive = AdaKern3D.apply_to_kernel(W_base, ...)
                        # shape (B, O, I, 3, 3, 3)
        -> for each d in dilation_list=[1,2,3]:
              branch_d = per-sample grouped F.conv3d(
                   x_fs, W_adaptive, padding=(d,d,d), dilation=(d,d,d))
              # each branch preserves (D, H, W)
        -> stack -> (B, 3, O, D, H, W)
        -> voxelwise k_att softmax over dim=1 (temperature-scaled)
        -> weighted sum along dim=1
        -> BatchNorm3d -> ReLU

Key correctness properties
--------------------------
1. Exactly ONE base kernel parameter `self.weight` per module.
2. NO per-dilation weights.
3. All three dilation branches use the SAME W_adaptive per sample.
4. Dilations are exactly (1,1,1), (2,2,2), (3,3,3), each with matching
   padding so branch outputs preserve (D, H, W).
5. k_att shape is (B, 3, D, H, W); softmax over dim=1 with temperature;
   expected_dilation = k_att[:,0]*1 + k_att[:,1]*2 + k_att[:,2]*3.

Groups
------
groups=1 only. AdaKern3D refuses other values; the per-sample grouped conv
below packs the batch into groups=B, and the base kernel's own groups=1 is
what makes that reshape sound. Grouped base kernels would need a different
packing.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fadc_3d_correct.freq_select_3d import FrequencySelection3D
from fadc_3d_correct.ada_kernel_3d import AdaKern3D


class KAttHead(nn.Module):
    """Voxelwise k_att head.

    Consumes the frequency-selected feature map to preserve local 3D spatial
    information (spec: don't use only global pooling).
    Two 3x3x3 convs + a 1x1x1 output.

        (B, I, D, H, W) -> (B, att_ch, D, H, W) -> (B, num_branches, D, H, W)

    Temperature applied only in the softmax over dim=1.
    """

    def __init__(
        self,
        in_channels: int,
        num_branches: int,
        att_ch: int,
    ) -> None:
        super().__init__()
        self.num_branches = int(num_branches)
        self.temperature = 1.0
        self.trunk = nn.Sequential(
            nn.Conv3d(in_channels, att_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, att_ch),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv3d(att_ch, num_branches, kernel_size=1, bias=True)
        # Zero-init the OUTPUT so the initial softmax is uniform over branches
        # (softmax(0) = 1/num_branches). Uniform mixing is a sensible identity
        # start for the corrected block.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def set_temperature(self, t: float) -> None:
        self.temperature = float(t)

    def forward(self, x_fs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x_fs)
        logits = self.head(h)                     # (B, num_branches, D, H, W)
        return F.softmax(logits / self.temperature, dim=1)


class AdaptiveDilatedConv3D(nn.Module):
    """Corrected FADC 3D block — one shared base kernel, discrete dilation set.

    Args:
        in_channels    : I.
        out_channels   : O.
        kernel_size    : must be 3 (only value supported).
        stride         : must be 1 (only value supported so that all three
                         dilation branches preserve D, H, W with symmetric padding).
        bias           : whether the base kernel carries a bias.
        dilation_list  : dilation values to mix. Default (1, 2, 3).
        fs_cfg         : dict of FrequencySelection3D kwargs; defaults to the
                         spec: k_list=[2,4,8], lowfreq_att=False, sigmoid,
                         spatial_group=1, spatial_kernel=3.
        adakern_cfg    : dict of AdaKern3D kwargs; defaults to
                         use_position_att=False.
        k_att_ch       : bottleneck channel width of the voxelwise k_att head.
        groups         : 1 only.
    """

    KERNEL_SIZE: int = 3
    STRIDE: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        dilation_list=(1, 2, 3),
        fs_cfg: Optional[dict] = None,
        adakern_cfg: Optional[dict] = None,
        k_att_ch: Optional[int] = None,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if kernel_size != self.KERNEL_SIZE:
            raise NotImplementedError(
                f"AdaptiveDilatedConv3D (corrected) only supports kernel_size=3, "
                f"got {kernel_size}."
            )
        if stride != self.STRIDE:
            raise NotImplementedError(
                f"AdaptiveDilatedConv3D (corrected) only supports stride=1, "
                f"got {stride}."
            )
        if groups != 1:
            raise NotImplementedError(
                f"AdaptiveDilatedConv3D (corrected) only supports groups=1, "
                f"got {groups}. Grouped bases need a different per-sample packing."
            )
        if any(int(d) <= 0 for d in dilation_list):
            raise ValueError(f"dilation_list must contain positive ints, got {dilation_list}")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dilation_list = tuple(int(d) for d in dilation_list)
        self.num_branches = len(self.dilation_list)

        # ---- ONE base kernel per module. This is the only kernel parameter.
        self.weight = nn.Parameter(torch.empty(
            self.out_channels, self.in_channels, self.KERNEL_SIZE, self.KERNEL_SIZE, self.KERNEL_SIZE))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter("bias", None)

        # ---- frequency selection
        if fs_cfg is None:
            fs_cfg = dict(
                k_list=(2, 4, 8),
                lowfreq_att=False,
                lp_type="freq",
                act="sigmoid",
                spatial_group=1,
                spatial_kernel=3,
            )
        self.fs = FrequencySelection3D(in_channels=self.in_channels, **fs_cfg)

        # ---- kernel-side attention (AdaKern3D)
        if adakern_cfg is None:
            adakern_cfg = dict(use_position_att=False)
        self.ada_kern = AdaKern3D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            **adakern_cfg,
        )

        # ---- voxelwise k_att head
        if k_att_ch is None:
            k_att_ch = max(int(self.in_channels * 0.125), 32)
        self.k_att = KAttHead(
            in_channels=self.in_channels,
            num_branches=self.num_branches,
            att_ch=k_att_ch,
        )

        # ---- diagnostic caches (populated on every forward)
        self.last_k_att: Optional[torch.Tensor] = None
        self.last_expected_dilation: Optional[torch.Tensor] = None
        self.last_c_low: Optional[torch.Tensor] = None
        self.last_f_low: Optional[torch.Tensor] = None
        self.last_c_high: Optional[torch.Tensor] = None
        self.last_f_high: Optional[torch.Tensor] = None
        self.last_s_att: Optional[torch.Tensor] = None
        self.last_x_fs: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ helpers
    def set_temperature(self, t: float) -> None:
        """Set the k_att softmax temperature (called by the training loop)."""
        self.k_att.set_temperature(t)

    def _batched_conv3d(
        self,
        x_fs: torch.Tensor,
        W_adaptive: torch.Tensor,
        dilation: int,
    ) -> torch.Tensor:
        """Per-sample F.conv3d via groups=B trick.

        x_fs        : (B, I, D, H, W)
        W_adaptive  : (B, O, I, K, K, K)
        Result      : (B, O, D, H, W)

        Packing:
          weight -> (B*O, I, K, K, K)
          x      -> (1, B*I, D, H, W)
          groups = B
          bias   -> repeated B times to shape (B*O,)
        Padding is dilation*(K-1)/2 = dilation (K=3) so output spatial shape
        equals input spatial shape.
        """
        B, I, D, H, W = x_fs.shape
        O = W_adaptive.size(1)
        K = self.KERNEL_SIZE

        weight = W_adaptive.reshape(B * O, I, K, K, K)
        x_pack = x_fs.reshape(1, B * I, D, H, W)

        if self.bias is not None:
            bias = self.bias.repeat(B)
        else:
            bias = None

        pad = dilation * (K - 1) // 2
        y = F.conv3d(
            x_pack,
            weight,
            bias=bias,
            stride=self.STRIDE,
            padding=pad,
            dilation=dilation,
            groups=B,
        )
        # y shape: (1, B*O, D, H, W)
        return y.view(B, O, D, H, W)

    # ------------------------------------------------------------------ main
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"in_channels mismatch: module {self.in_channels}, tensor {x.size(1)}"
            )

        # 1) frequency selection
        x_fs = self.fs(x)

        # 2) kernel-side attention (per-sample multipliers on W_low / W_high)
        c_low, f_low, c_high, f_high, s_att = self.ada_kern(x_fs)

        # 3) per-sample adaptive kernel (uses the ONE base kernel self.weight)
        W_adaptive = self.ada_kern.apply_to_kernel(
            self.weight, c_low, f_low, c_high, f_high, s_att
        )

        # 4) same kernel at every dilation choice
        branch_outs = []
        for d in self.dilation_list:
            branch_outs.append(self._batched_conv3d(x_fs, W_adaptive, dilation=d))
        branch_stack = torch.stack(branch_outs, dim=1)          # (B, num_br, O, D, H, W)

        # 5) voxelwise k_att over branches, softmax with temperature
        k_att = self.k_att(x_fs)                                 # (B, num_br, D, H, W)

        # 6) mix
        out = (branch_stack * k_att.unsqueeze(2)).sum(dim=1)     # (B, O, D, H, W)

        # 7) diagnostics
        with torch.no_grad():
            self.last_k_att = k_att.detach()
            # expected_dilation[b, d, h, w] = sum_i k_att[b,i,d,h,w] * dilation_list[i]
            dilation_tensor = torch.tensor(
                self.dilation_list, device=k_att.device, dtype=k_att.dtype
            ).view(1, -1, 1, 1, 1)
            self.last_expected_dilation = (k_att * dilation_tensor).sum(dim=1).detach()
            self.last_c_low = c_low.detach()
            self.last_f_low = f_low.detach()
            self.last_c_high = c_high.detach()
            self.last_f_high = f_high.detach()
            self.last_s_att = s_att.detach() if s_att is not None else None
            # Keep x_fs light: diag will recompute if needed. Just store shape marker.
            self.last_x_fs = None

        return out
