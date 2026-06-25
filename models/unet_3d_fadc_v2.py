"""UNet3DFADC_V2 — same architecture as models/unet_3d_fadc.py but uses the
v2 AdaptiveDilatedConv3DV2 (spatial k_att + warm bias + temperature anneal).

Architectural depth, channel widths, dropout, residual connections, deep-
supervision heads, and forward shape are all unchanged from v1. Only the
underlying FADC building block is swapped.

Use the same fadc_placement values as the v1 model — meaning, for instance,
unet3d_fadc_bottleneck_v2 uses the same bottleneck-only placement strategy
as unet3d_fadc_bottleneck, just with the v2 FADC implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fadc_3d_v2.adaptive_dilated_conv_3d_v2 import AdaptiveDilatedConv3DV2

# Same minimal FrequencySelection cfg as v1 — k_list=[2] only.
_FS_CFG = dict(
    k_list=[2],
    lowfreq_att=False,
    lp_type='freq',
    act='sigmoid',
    spatial_group=1,
)


class FADCConvBlockV2(nn.Module):
    """Two AdaptiveDilatedConv3DV2 → BN → ReLU with residual connection."""
    def __init__(self, in_ch, out_ch, dropout=0.15,
                 k_att_kernel_size=3, bias_init=0.5):
        super().__init__()
        self.conv1 = AdaptiveDilatedConv3DV2(
            in_ch, out_ch, kernel_size=3, bias=False, fs_cfg=_FS_CFG,
            k_att_kernel_size=k_att_kernel_size, bias_init=bias_init)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout3d(p=dropout)
        self.conv2 = AdaptiveDilatedConv3DV2(
            out_ch, out_ch, kernel_size=3, bias=False, fs_cfg=_FS_CFG,
            k_att_kernel_size=k_att_kernel_size, bias_init=bias_init)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

        if in_ch != out_ch:
            self.skip_proj = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        identity = self.skip_proj(x)
        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + identity)


class ConvBlock(nn.Module):
    """Plain double conv block — same as v1 (no FADC)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def _make_block(in_ch, out_ch, use_fadc):
    return FADCConvBlockV2(in_ch, out_ch) if use_fadc else ConvBlock(in_ch, out_ch)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_fadc=False):
        super().__init__()
        self.conv = _make_block(in_ch, out_ch, use_fadc)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_fadc=False):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = _make_block(in_ch, out_ch, use_fadc)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3DFADC_V2(nn.Module):
    """3D U-Net with FADC v2 (spatial k_att). Same placement options as v1.

    fadc_placement:
      'full' | 'encoder' | 'bottleneck' | 'mid' | 'deep' | 'deep_lite'
    """
    def __init__(self, in_channels=1, out_channels=2, base_filters=32,
                 fadc_placement='full', deep_supervision=False):
        super().__init__()
        assert fadc_placement in ('full', 'encoder', 'bottleneck',
                                  'mid', 'deep', 'deep_lite'), \
            f"unknown fadc_placement: {fadc_placement}"

        enc_fadc = fadc_placement in ('full', 'encoder')
        bn_fadc  = fadc_placement in ('full', 'bottleneck', 'deep', 'deep_lite')
        dec_fadc = fadc_placement in ('full',)

        self.deep_supervision = deep_supervision
        self.fadc_placement = fadc_placement
        f = base_filters

        if fadc_placement == 'mid':
            self.enc1 = DownBlock(in_channels, f,     use_fadc=False)
            self.enc2 = DownBlock(f,      f * 2,      use_fadc=True)
            self.enc3 = DownBlock(f * 2,  f * 4,      use_fadc=True)
            self.enc4 = DownBlock(f * 4,  f * 8,      use_fadc=False)
        elif fadc_placement == 'deep':
            self.enc1 = DownBlock(in_channels, f,     use_fadc=False)
            self.enc2 = DownBlock(f,      f * 2,      use_fadc=False)
            self.enc3 = DownBlock(f * 2,  f * 4,      use_fadc=True)
            self.enc4 = DownBlock(f * 4,  f * 8,      use_fadc=True)
        elif fadc_placement == 'deep_lite':
            self.enc1 = DownBlock(in_channels, f,     use_fadc=False)
            self.enc2 = DownBlock(f,      f * 2,      use_fadc=False)
            self.enc3 = DownBlock(f * 2,  f * 4,      use_fadc=False)
            self.enc4 = DownBlock(f * 4,  f * 8,      use_fadc=True)
        else:
            self.enc1 = DownBlock(in_channels, f,     use_fadc=enc_fadc)
            self.enc2 = DownBlock(f,      f * 2,      use_fadc=enc_fadc)
            self.enc3 = DownBlock(f * 2,  f * 4,      use_fadc=enc_fadc)
            self.enc4 = DownBlock(f * 4,  f * 8,      use_fadc=enc_fadc)

        self.bottleneck = _make_block(f * 8, f * 16, use_fadc=bn_fadc)

        self.dec4 = UpBlock(f * 16, f * 8,  use_fadc=dec_fadc)
        self.dec3 = UpBlock(f * 8,  f * 4,  use_fadc=dec_fadc)
        self.dec2 = UpBlock(f * 4,  f * 2,  use_fadc=dec_fadc)
        self.dec1 = UpBlock(f * 2,  f,      use_fadc=dec_fadc)

        self.head = nn.Conv3d(f, out_channels, kernel_size=1)

        if deep_supervision:
            self.ds_head2 = nn.Conv3d(f * 2, out_channels, kernel_size=1)
            self.ds_head3 = nn.Conv3d(f * 4, out_channels, kernel_size=1)
            self.ds_head4 = nn.Conv3d(f * 8, out_channels, kernel_size=1)

    def set_temperature(self, t: float):
        """Cascade temperature to every AdaptiveDilatedConv3DV2 in the model.
        Called once per epoch by the training loop for cosine annealing."""
        for m in self.modules():
            if isinstance(m, AdaptiveDilatedConv3DV2):
                m.set_temperature(t)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        x = self.bottleneck(x)

        d4 = self.dec4(x,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        main = self.head(d1)
        if self.deep_supervision and self.training:
            return main, self.ds_head2(d2), self.ds_head3(d3), self.ds_head4(d4)
        return main


if __name__ == '__main__':
    for placement in ('bottleneck', 'deep'):
        m = UNet3DFADC_V2(in_channels=2, out_channels=2, base_filters=8,
                          fadc_placement=placement)
        params = sum(p.numel() for p in m.parameters())
        x = torch.randn(1, 2, 32, 32, 16)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 2, 32, 32, 16)
        print(f"placement={placement:12s} | params={params:,} | output={y.shape} OK")

    # Temperature anneal smoke
    m.set_temperature(4.0)
    assert all(b.omni_att.temperature == 4.0
               for b in m.modules() if isinstance(b, AdaptiveDilatedConv3DV2))
    m.set_temperature(1.0)
    assert all(b.omni_att.temperature == 1.0
               for b in m.modules() if isinstance(b, AdaptiveDilatedConv3DV2))
    print("set_temperature OK")
