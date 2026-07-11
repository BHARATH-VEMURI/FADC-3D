"""OmniAttention3DSpatial — v2 with per-voxel kernel attention.

Differences vs original OmniAttention3D (in fadc_3d/omni_attention_3d.py):

  1. SPATIAL k_att.
       Original: global avg pool -> 1x1x1 conv -> softmax. Output shape
                 (b, kernel_num, 1, 1, 1) — one weight per whole input image.
       v2:      Direct 3x3x3 conv on the full feature map -> 1x1x1 conv -> softmax.
                Output shape (b, kernel_num, d, h, w) — one softmax per voxel.

  2. WARM bias init for channel_fc and filter_fc.
       Original: bias=0 -> sigmoid(0)*2 = 1.0 — exact identity, dead gradient.
       v2:      bias=0.5 -> sigmoid(0.5)*2 = 1.24 — starts slightly above identity
                so gradients are non-trivial.

  3. TEMPERATURE annealing supported via set_temperature(t).
       The training loop schedules self.temperature from a high value (e.g. 4.0)
       down to 1.0 over training epochs. Applied only inside the k_att softmax.

  4. RICHER pooled descriptor (v2.1 capacity bump).
       Original: single AdaptiveAvgPool3d(1) -> shared FC.
       v2.1:    Concatenate AdaptiveAvgPool3d(1) with AdaptiveMaxPool3d(1)
                along the channel dim before the shared FC, so the c_att /
                f_att / s_att paths see BOTH mean and peak statistics. The
                shared FC takes 2*in_planes channels in.

  5. WIDER attention bottleneck (v2.1 capacity bump).
       Original: reduction=0.0625, min_channel=16 -> attention_channel often
                 pinned at 16 for shallow layers.
       v2.1:    reduction=0.125, min_channel=32 -> attention_channel is at
                least 32 everywhere, and doubles for wide deep layers.

  6. ALWAYS-ON filter attention (v2.1 correctness).
       Original: skipped filter_fc when in_planes == groups == out_planes,
                 returning the constant 1.0 for f_att.
       v2.1:    filter_fc is always constructed; f_att is always computed.

The interface (forward returns 4-tuple c_att, f_att, s_att, k_att) is preserved
so AdaptiveDilatedConv3DV2 can be a drop-in for the original AdaptiveDilatedConv3D
in any code that consumes the same tuple shape.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OmniAttention3DSpatial(nn.Module):
    """v2 attention — richer avg+max pooled descriptor for c_att / f_att / s_att,
    dedicated spatial path for k_att.

    Args:
      in_planes, out_planes, kernel_size, groups, reduction, kernel_num, min_channel
        — same semantics as the original OmniAttention3D.
      k_att_kernel_size : kernel size of the spatial k_att conv head (default 3).
      bias_init         : initial bias for channel_fc and filter_fc (default 0.5).
                          0.5 -> sigmoid(0.5)*2 ~ 1.24 (above identity).
    """

    def __init__(self, in_planes, out_planes, kernel_size,
                 groups=1, reduction=0.125, kernel_num=4, min_channel=32,
                 k_att_kernel_size=3, bias_init=0.5):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0
        self.bias_init = bias_init

        # ── shared pooled descriptor path — avg + max concat (v2.1) ──
        # Used for c_att, f_att, s_att (NOT k_att in v2).
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        # Concat doubles the channel dim before the FC.
        self.fc = nn.Conv3d(in_planes * 2, attention_channel, 1, bias=False)
        self.gn = nn.GroupNorm(1, attention_channel)
        self.relu = nn.ReLU(inplace=True)

        # ── channel attention ──
        self.channel_fc = nn.Conv3d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        # ── filter attention — always on (v2.1) ──
        self.filter_fc = nn.Conv3d(attention_channel, out_planes, 1, bias=True)
        self.func_filter = self.get_filter_attention

        # ── spatial attention (still global-pooled — kernel_size=1 path uses skip) ──
        if kernel_size == 1:
            self.func_spatial = self.skip
            self.spatial_fc = None
        else:
            self.spatial_fc = nn.Conv3d(attention_channel, kernel_size ** 3, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        # ── kernel attention (THE V2 CHANGE) ──
        # Per-voxel softmax over branches.
        if kernel_num == 1:
            self.func_kernel = self.skip
            self.kernel_spatial_trunk = None
            self.kernel_spatial_head = None
        else:
            self.kernel_spatial_trunk = nn.Sequential(
                nn.Conv3d(in_planes, attention_channel,
                          kernel_size=k_att_kernel_size,
                          padding=k_att_kernel_size // 2,
                          bias=False),
                nn.GroupNorm(1, attention_channel),
                nn.ReLU(inplace=True),
            )
            self.kernel_spatial_head = nn.Conv3d(attention_channel, kernel_num,
                                                 kernel_size=1, bias=True)
            self.func_kernel = self.get_kernel_attention_spatial

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # ── WARM bias for c_att / f_att (v2 fix, kept in v2.1) ──
        nn.init.constant_(self.channel_fc.bias, self.bias_init)
        nn.init.constant_(self.filter_fc.bias, self.bias_init)

    def set_temperature(self, t: float):
        """Set softmax temperature for kernel attention (called by training loop)."""
        self.temperature = float(t)

    # Backwards-compat alias used by the original OmniAttention3D codebase.
    def update_temperature(self, temperature):
        self.set_temperature(temperature)

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x_pooled):
        # x_pooled: (b, attention_channel, 1, 1, 1)
        # Sigmoid has no temperature term — temperature belongs to k_att softmax only.
        return torch.sigmoid(self.channel_fc(x_pooled).view(
            x_pooled.size(0), -1, 1, 1, 1))

    def get_filter_attention(self, x_pooled):
        return torch.sigmoid(self.filter_fc(x_pooled).view(
            x_pooled.size(0), -1, 1, 1, 1))

    def get_spatial_attention(self, x_pooled):
        k = self.kernel_size
        spatial_attention = self.spatial_fc(x_pooled).view(
            x_pooled.size(0), 1, 1, 1, 1, k, k, k)
        return torch.sigmoid(spatial_attention)

    def get_kernel_attention_spatial(self, x_full):
        """v2 spatial k_att path. x_full: (b, in_planes, d, h, w).
        Returns: (b, kernel_num, d, h, w)."""
        h = self.kernel_spatial_trunk(x_full)
        logits = self.kernel_spatial_head(h)         # (b, kernel_num, d, h, w)
        return F.softmax(logits / self.temperature, dim=1)

    def forward(self, x):
        # Pooled descriptor for c_att / f_att / s_att — avg + max concat.
        avg = self.avgpool(x)                       # (b, in_planes, 1, 1, 1)
        mx  = self.maxpool(x)                       # (b, in_planes, 1, 1, 1)
        x_pooled = torch.cat([avg, mx], dim=1)      # (b, 2*in_planes, 1, 1, 1)
        x_pooled = self.fc(x_pooled)
        x_pooled = self.gn(x_pooled)
        x_pooled = self.relu(x_pooled)

        c_att = self.func_channel(x_pooled)
        f_att = self.func_filter(x_pooled)
        s_att = self.func_spatial(x_pooled)

        # k_att uses the FULL feature map for per-voxel adaptation
        if self.kernel_num == 1:
            k_att = self.func_kernel(x_pooled)  # scalar 1.0 via skip
        else:
            k_att = self.func_kernel(x)

        return c_att, f_att, s_att, k_att
