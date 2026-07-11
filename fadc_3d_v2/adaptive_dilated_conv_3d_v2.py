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

No other op changes are required.
"""
import torch
import torch.nn as nn

from fadc_3d_v2.omni_attention_3d_spatial import OmniAttention3DSpatial
from fadc_3d_v2.freq_select_3d import FrequencySelection3D


class AdaptiveDilatedConv3DV2(nn.Module):
    """3D FADC v2 — spatial k_att variant.

    Drop-in replacement for the original AdaptiveDilatedConv3D at any place
    in the U-Net.

    Args identical to v1 plus:
      k_att_kernel_size (int, default 3) — kernel size of the v2 spatial
                                            k_att head (passed to OmniAttention3DSpatial).
      bias_init         (float, default 0.5) — warm bias for c_att/f_att.
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
                 bias_init=0.5):
        super().__init__()
        if dilation_list is None:
            dilation_list = [1, 2]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation_list = dilation_list

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

        # v2 attention — spatial k_att
        self.omni_att = OmniAttention3DSpatial(
            in_planes=in_channels,
            out_planes=out_channels,
            kernel_size=1,          # spatial att unused — skip path
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

        # Step 2 — attention signals (k_att is now SPATIAL: (b, n_br, d, h, w))
        c_att, f_att, _, k_att = self.omni_att(x_fs)

        # Step 3 — channel-gate the input, then run all dilation branches
        x_in = x_fs * (c_att * 2)
        branch_outs = torch.stack([conv(x_in) for conv in self.conv_branches], dim=1)
        # branch_outs: (b, num_branches, out_channels, d, h, w)

        # Step 4 — weighted sum over branches.
        # k_att.unsqueeze(2): (b, num_branches, 1, d, h, w) — broadcasts across out_channels.
        out = (branch_outs * k_att.unsqueeze(2)).sum(dim=1)

        # Step 5 — filter-gate the output
        out = out * (f_att * 2)
        return out


if __name__ == '__main__':
    # Sanity check: shape passes through, k_att is per-voxel.
    x = torch.rand(1, 32, 16, 32, 32)
    m = AdaptiveDilatedConv3DV2(in_channels=32, out_channels=64)
    y = m(x)
    print('Input :', x.shape)
    print('Output:', y.shape)

    # Verify k_att shape via a fresh hook
    captured = []
    def hook(_mod, _inp, out):
        captured.append(out[3].shape)  # k_att is the 4th in the tuple
    h = m.omni_att.register_forward_hook(hook)
    with torch.no_grad():
        _ = m(x)
    h.remove()
    print('k_att shape:', captured[0])    # expect (1, 2, 16, 32, 32) — per-voxel
