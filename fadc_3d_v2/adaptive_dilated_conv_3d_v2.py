"""AdaptiveDilatedConv3DV2 — same skeleton as v1 but uses OmniAttention3DSpatial.

The forward pass is identical to v1 except for how k_att broadcasts:

  v1: k_att shape (b, num_branches, 1, 1, 1)
       .unsqueeze(2) -> (b, num_branches, 1, 1, 1, 1)
       broadcasts uniformly across c, d, h, w when multiplied with
       branch_outs of shape (b, num_branches, out_channels, d, h, w)
       → SAME weight everywhere in space.

  v2: k_att shape (b, num_branches, d, h, w)
       .unsqueeze(2) -> (b, num_branches, 1, d, h, w)
       broadcasts across the channel dim, varies in space
       → DIFFERENT weight per voxel.

v2.2 additions (this file):

  * s_att ENABLED. The inner OmniAttention3DSpatial is constructed with the
    real conv kernel_size (e.g. 3), so spatial_fc exists and the module
    returns a real (B, 1, 1, 1, 1, K, K, K) spatial-attention tensor rather
    than the constant 1.0 skip.

  * s_att APPLIED. Each dilated branch is now run through a per-sample
    grouped F.conv3d whose weight is `conv.weight * (s_att * 2)`. This
    modulates every kernel-position gain per sample without touching
    stride / padding / dilation / groups / bias semantics.

  * f_att MOVED OUTSIDE the conv (optional). BatchNorm downstream can wash
    the pre-BN filter scaling out. When `apply_filter_attention=False`
    the module stores `self.last_filter_attention` and the caller (see
    FADCConvBlockV2) applies it AFTER BN, where it actually survives.

  * GATE MODE for c_att and f_att.
      'multiply': legacy v2/v2.1 behaviour, out = x * (att * 2). Destructive
                  — att=0 kills the signal entirely, gradient is small when
                  output magnitude is small.
      'residual': out = x * (1 + gamma * (2*att - 1)). Identity at att=0.5,
                  range [1-gamma, 1+gamma]. gamma=1.0 preserves the [0, 2]
                  span of 'multiply' but centred on identity; gamma<1.0 is
                  more conservative (never fully kills the signal).
    Both modes leave the attention values themselves unchanged — only how
    they modulate the feature map differs. The caller (FADCConvBlockV2)
    reads the ready-to-multiply tensor from `self.last_filter_attention_mul`
    so it applies the same gate f_att would have used inside the conv.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fadc_3d_v2.omni_attention_3d_spatial import OmniAttention3DSpatial
from fadc_3d_v2.freq_select_3d import FrequencySelection3D


def _apply_gate(x, att, mode: str, gamma: float):
    """Modulate `x` by attention `att` (sigmoid in [0, 1]).
      'multiply': x * (att * 2)                      — legacy destructive gate.
      'residual': x * (1 + gamma * (2*att - 1))      — identity at att=0.5.
    """
    if mode == 'multiply':
        return x * (att * 2)
    if mode == 'residual':
        return x * (1.0 + gamma * (2.0 * att - 1.0))
    raise ValueError(f"unknown gate mode: {mode!r}")


def _gate_multiplier(att, mode: str, gamma: float):
    """Return the ready-to-multiply tensor for a given gate mode.
    Callers that want to apply the gate later (e.g. post-BN) can just do
      out = raw_out * _gate_multiplier(att, mode, gamma).
    """
    if mode == 'multiply':
        return att * 2
    if mode == 'residual':
        return 1.0 + gamma * (2.0 * att - 1.0)
    raise ValueError(f"unknown gate mode: {mode!r}")


def _conv_with_spatial_att(conv: nn.Conv3d, x: torch.Tensor, s_att):
    """Run `conv` on `x` with per-sample s_att modulation of kernel positions.

    If s_att is not a tensor (skip path), falls back to plain conv(x).

    s_att shape: (B, 1, 1, 1, 1, K, K, K) — reshaped internally to
                 (B, 1, 1, K, K, K) so it broadcasts across (out_ch, in_ch/groups).

    The multiplier is `s_att * 2`, matching the standard ODConv / OmniAttention
    convention where sigmoid(0) → 0.5 → *2 → 1.0 identity.

    Implementation: one grouped F.conv3d over the whole batch.
      weight -> (B*O, I_g, K, K, K)
      x      -> (1, B*C_in, D, H, W)
      groups = B * conv.groups
    stride / padding / dilation / bias are preserved from `conv`.
    """
    if not torch.is_tensor(s_att):
        return conv(x)

    B, C_in, D, H, W = x.shape
    weight = conv.weight                        # (O, I_g, K1, K2, K3)
    O, I_g, K1, K2, K3 = weight.shape

    # s_att comes in as (B, 1, 1, 1, 1, K, K, K); reshape for broadcast.
    s = s_att.view(B, 1, 1, K1, K2, K3)
    modulated = weight.unsqueeze(0) * (s * 2)    # (B, O, I_g, K, K, K)
    modulated = modulated.reshape(B * O, I_g, K1, K2, K3)

    x_grouped = x.reshape(1, B * C_in, D, H, W)
    bias = None
    if conv.bias is not None:
        bias = conv.bias.repeat(B)               # (B*O,)

    out = F.conv3d(
        x_grouped, modulated, bias=bias,
        stride=conv.stride, padding=conv.padding,
        dilation=conv.dilation, groups=B * conv.groups,
    )
    _, _, D_o, H_o, W_o = out.shape
    return out.view(B, O, D_o, H_o, W_o)


class AdaptiveDilatedConv3DV2(nn.Module):
    """3D FADC v2 — spatial k_att variant.

    Drop-in replacement for the original AdaptiveDilatedConv3D at any place
    in the U-Net.

    Args identical to v1 plus:
      k_att_kernel_size (int, default 3) — kernel size of the v2 spatial
                                            k_att head (passed to OmniAttention3DSpatial).
      bias_init         (float, default 0.5) — warm bias for c_att/f_att.
      apply_filter_attention (bool, default True) — if True, keep v1/v2.1
                                            behavior of scaling the conv output
                                            by (f_att * 2) inside this module.
                                            Set False when the caller (e.g.
                                            FADCConvBlockV2) prefers to apply
                                            f_att AFTER BatchNorm; the last
                                            f_att is exposed as
                                            self.last_filter_attention so the
                                            caller can pick it up.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 bias=True,
                 dilation_list=None,
                 reduction=0.125,
                 min_channel=32,
                 fs_cfg=None,
                 k_att_kernel_size=3,
                 bias_init=0.5,
                 apply_filter_attention=True,
                 channel_gate_mode='residual',
                 filter_gate_mode='residual',
                 gate_gamma=1.0):
        super().__init__()
        if dilation_list is None:
            dilation_list = [1, 2]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation_list = dilation_list
        self.apply_filter_attention = apply_filter_attention
        self.channel_gate_mode = channel_gate_mode
        self.filter_gate_mode  = filter_gate_mode
        self.gate_gamma        = float(gate_gamma)
        # Populated on every forward so callers that want f_att post-BN can
        # read it without re-running the attention module.
        self.last_filter_attention     = None   # raw sigmoid in [0, 1]
        self.last_filter_attention_mul = None   # ready-to-multiply tensor

        # One dilated conv branch per dilation rate (unchanged from v1)
        self.conv_branches = nn.ModuleList()
        for dil in dilation_list:
            pad = dil * (kernel_size - 1) // 2
            self.conv_branches.append(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=pad,
                          dilation=dil,
                          groups=groups,
                          bias=bias))

        # v2.2: pass the REAL kernel_size so spatial_fc is constructed and
        # s_att comes out as a live tensor (see get_spatial_attention).
        self.omni_att = OmniAttention3DSpatial(
            in_planes=in_channels,
            out_planes=out_channels,
            kernel_size=kernel_size,
            kernel_num=len(dilation_list),
            reduction=reduction,
            min_channel=min_channel,
            k_att_kernel_size=k_att_kernel_size,
            bias_init=bias_init,
        )

        # Frequency pre-selection (v2 uses an unchanged copy of v1's module)
        if fs_cfg is None:
            fs_cfg = dict(
                k_list=[2, 4, 8],
                lowfreq_att=False,
                lp_type='freq',
                act='sigmoid',
                spatial_group=1)
        self.fs = FrequencySelection3D(in_channels, **fs_cfg)

        self._initialize_weights()

    def _initialize_weights(self):
        for conv in self.conv_branches:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)

    def set_temperature(self, t: float):
        """Forward to the inner attention so the training loop can anneal."""
        self.omni_att.set_temperature(t)

    def forward(self, x):
        # Step 1 — frequency pre-selection
        x_fs = self.fs(x)

        # Step 2 — attention signals. s_att is a real tensor (v2.2), k_att is
        # spatial (v2), c/f are pooled sigmoids in [0, 1].
        c_att, f_att, s_att, k_att = self.omni_att(x_fs)

        # Cache f_att raw + the ready multiplier for FADCConvBlockV2 to
        # apply AFTER BN with the same gate mode configured here.
        self.last_filter_attention     = f_att
        self.last_filter_attention_mul = _gate_multiplier(
            f_att, self.filter_gate_mode, self.gate_gamma)

        # Step 3 — channel-gate the input (residual-centred by default in v2.2)
        x_in = _apply_gate(x_fs, c_att, self.channel_gate_mode, self.gate_gamma)

        # Step 4 — run each dilated branch. s_att modulates the kernel
        # positions per sample via grouped conv (falls back to plain conv
        # if s_att is a scalar skip). s_att gate is *multiply* (unchanged) —
        # per user directive s_att grouped-conv application stays as-is.
        branch_outs = torch.stack(
            [_conv_with_spatial_att(conv, x_in, s_att) for conv in self.conv_branches],
            dim=1,
        )
        # branch_outs: (b, num_branches, out_channels, d, h, w)

        # Step 5 — weighted sum over branches (k_att is per-voxel).
        out = (branch_outs * k_att.unsqueeze(2)).sum(dim=1)

        # Step 6 — filter-gate the output ONLY if the caller wants us to.
        # When apply_filter_attention=False, the caller reads
        # self.last_filter_attention_mul and applies it after BN.
        if self.apply_filter_attention:
            out = out * self.last_filter_attention_mul
        return out


if __name__ == '__main__':
    # Sanity check: shape passes through, k_att is per-voxel, s_att is a tensor.
    x = torch.rand(1, 32, 16, 32, 32)
    m = AdaptiveDilatedConv3DV2(in_channels=32, out_channels=64)
    y = m(x)
    print('Input :', x.shape)
    print('Output:', y.shape)

    captured = []

    def hook(_mod, _inp, out):
        captured.append(tuple(t.shape if torch.is_tensor(t) else type(t).__name__ for t in out))

    h = m.omni_att.register_forward_hook(hook)
    with torch.no_grad():
        _ = m(x)
    h.remove()
    print('att shapes (c, f, s, k):', captured[0])
