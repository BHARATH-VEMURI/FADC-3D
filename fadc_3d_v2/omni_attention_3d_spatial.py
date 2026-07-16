"""OmniAttention3DSpatial — v2.2 with richer descriptor and separated c/f heads.

Differences from v2.1:

  1. RICHER pooled descriptor.
       v2.1: avg + max concat (2 * in_planes channels).
       v2.2: avg + max + std concat (3 * in_planes channels).
             The std pool adds per-channel spread information — helps c_att and
             f_att distinguish "this input has flat features" from "this input
             has textured features," which pooled mean+max alone cannot.

  2. SEPARATE LIGHTWEIGHT HEADS FOR c_att AND f_att.
       v2.1: a single shared FC fed channel_fc, filter_fc, and spatial_fc.
       v2.2: shared_fc feeds spatial_fc (kept — s_att is working);
             c_trunk_fc feeds channel_fc;
             f_trunk_fc feeds filter_fc.
             Decouples c and f gradients so f_att no longer competes with c_att
             for the same bottleneck capacity — the diagnostic showed deep-layer
             c/f collapse even after v2.2's post-BN move, and shared-bottleneck
             capacity contention was one of the two remaining suspects.

Preserved from v2 / v2.1:

  * Spatial k_att (per-voxel softmax over branches).
  * Warm bias init for channel_fc / filter_fc (bias=0.5 -> sigmoid=0.622).
  * Temperature annealing via set_temperature(t) (applied only inside k_att).
  * Always-on filter attention (spatial_fc, filter_fc constructed at every layer).

Interface: forward returns (c_att, f_att, s_att, k_att) — unchanged shape/order.
c/f/s are raw sigmoids in [0, 1]; k_att is a per-voxel softmax over branches.
The GATE (multiply vs residual) lives in the caller (AdaptiveDilatedConv3DV2)
so this module stays pure attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OmniAttention3DSpatial(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 groups=1, reduction=0.125, kernel_num=4, min_channel=32,
                 k_att_kernel_size=3, bias_init=0.5):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0
        self.bias_init = bias_init

        # ── descriptor: avg + max + std, concatenated on the channel dim ──
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        descriptor_ch = in_planes * 3   # std pool is computed inline in forward

        # ── shared FC — feeds s_att only (kept from v2.1 because s_att works) ──
        self.shared_fc   = nn.Conv3d(descriptor_ch, attention_channel, 1, bias=False)
        self.shared_gn   = nn.GroupNorm(1, attention_channel)
        self.shared_relu = nn.ReLU(inplace=True)

        # ── separate lightweight trunks for c_att and f_att ──
        self.c_trunk_fc   = nn.Conv3d(descriptor_ch, attention_channel, 1, bias=False)
        self.c_trunk_gn   = nn.GroupNorm(1, attention_channel)
        self.c_trunk_relu = nn.ReLU(inplace=True)

        self.f_trunk_fc   = nn.Conv3d(descriptor_ch, attention_channel, 1, bias=False)
        self.f_trunk_gn   = nn.GroupNorm(1, attention_channel)
        self.f_trunk_relu = nn.ReLU(inplace=True)

        # ── head convs ──
        self.channel_fc = nn.Conv3d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        self.filter_fc = nn.Conv3d(attention_channel, out_planes, 1, bias=True)
        self.func_filter = self.get_filter_attention

        if kernel_size == 1:
            self.func_spatial = self.skip
            self.spatial_fc = None
        else:
            self.spatial_fc = nn.Conv3d(attention_channel, kernel_size ** 3, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        # ── kernel attention (v2 spatial path — UNCHANGED, still working) ──
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
        nn.init.constant_(self.channel_fc.bias, self.bias_init)
        nn.init.constant_(self.filter_fc.bias, self.bias_init)

    def set_temperature(self, t: float):
        self.temperature = float(t)

    def update_temperature(self, temperature):
        self.set_temperature(temperature)

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, c_pooled):
        return torch.sigmoid(self.channel_fc(c_pooled).view(
            c_pooled.size(0), -1, 1, 1, 1))

    def get_filter_attention(self, f_pooled):
        return torch.sigmoid(self.filter_fc(f_pooled).view(
            f_pooled.size(0), -1, 1, 1, 1))

    def get_spatial_attention(self, s_pooled):
        k = self.kernel_size
        spatial_attention = self.spatial_fc(s_pooled).view(
            s_pooled.size(0), 1, 1, 1, 1, k, k, k)
        return torch.sigmoid(spatial_attention)

    def get_kernel_attention_spatial(self, x_full):
        h = self.kernel_spatial_trunk(x_full)
        logits = self.kernel_spatial_head(h)
        return F.softmax(logits / self.temperature, dim=1)

    @staticmethod
    def _std_pool(x: torch.Tensor) -> torch.Tensor:
        """Per-channel spatial std -> (B, C, 1, 1, 1). fp32 for autocast stability."""
        s = x.float().flatten(2).std(dim=2, unbiased=False)
        return s.view(x.size(0), x.size(1), 1, 1, 1).to(x.dtype)

    def forward(self, x):
        avg = self.avgpool(x)
        mx  = self.maxpool(x)
        sd  = self._std_pool(x)
        descriptor = torch.cat([avg, mx, sd], dim=1)   # (B, 3*in_planes, 1, 1, 1)

        s_pooled = self.shared_relu(self.shared_gn(self.shared_fc(descriptor)))
        c_pooled = self.c_trunk_relu(self.c_trunk_gn(self.c_trunk_fc(descriptor)))
        f_pooled = self.f_trunk_relu(self.f_trunk_gn(self.f_trunk_fc(descriptor)))

        c_att = self.func_channel(c_pooled)
        f_att = self.func_filter(f_pooled)
        s_att = self.func_spatial(s_pooled)
        k_att = self.func_kernel(x)   # skip() ignores input; spatial path uses x

        return c_att, f_att, s_att, k_att
