import torch
import torch.nn as nn
import torch.nn.functional as F

from fadc_3d.adaptive_dilated_conv_3d import AdaptiveDilatedConv3D

# k_list=[2] only — prevents empty FFT masks at small 3D bottleneck depths
_FS_CFG = dict(
    k_list=[2],
    lowfreq_att=False,
    lp_type='freq',
    act='sigmoid',
    spatial_group=1,
)


class FADCConvBlock(nn.Module):
    """Two consecutive AdaptiveDilatedConv3D → BN → ReLU with residual connection."""
    def __init__(self, in_ch, out_ch, dropout=0.15):
        super().__init__()
        self.conv1 = AdaptiveDilatedConv3D(in_ch, out_ch, kernel_size=3, bias=False, fs_cfg=_FS_CFG)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout3d(p=dropout)
        self.conv2 = AdaptiveDilatedConv3D(out_ch, out_ch, kernel_size=3, bias=False, fs_cfg=_FS_CFG)
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
    """Standard double conv block — identical to unet_3d.py baseline."""
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
    return FADCConvBlock(in_ch, out_ch) if use_fadc else ConvBlock(in_ch, out_ch)


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


class UNet3DFADC(nn.Module):
    """
    3D U-Net with Frequency-Adaptive Dilated Convolution.

    fadc_placement controls where FADC blocks are used:
      'full'       — encoder + bottleneck + decoder  (FADC-Full)
      'encoder'    — encoder only                    (FADC-Encoder ablation)
      'bottleneck' — bottleneck only                 (FADC-Bottleneck ablation)
      'mid'        — enc2 + enc3 only                (FADC-Mid — selective placement)
      'deep'       — enc3 + enc4 + bottleneck        (FADC-Deep — deeper selective)
      'deep_lite'  — enc4 + bottleneck               (FADC-Deep-Lite — isolates enc4's contribution within Deep)

    Architecture depth and channel widths are identical across all placements —
    only the conv block type changes, making this a clean ablation.
    """
    def __init__(self, in_channels=1, out_channels=2, base_filters=32,
                 fadc_placement='full'):
        super().__init__()
        assert fadc_placement in ('full', 'encoder', 'bottleneck', 'mid', 'deep', 'deep_lite'), \
            f"fadc_placement must be 'full', 'encoder', 'bottleneck', 'mid', 'deep', or 'deep_lite', got '{fadc_placement}'"

        enc_fadc = fadc_placement in ('full', 'encoder')
        bn_fadc  = fadc_placement in ('full', 'bottleneck', 'deep', 'deep_lite')
        dec_fadc = fadc_placement in ('full',)

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

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.head(x)


if __name__ == '__main__':
    for placement in ('full', 'encoder', 'bottleneck', 'mid', 'deep', 'deep_lite'):
        model = UNet3DFADC(in_channels=2, out_channels=2, base_filters=32,
                           fadc_placement=placement)
        total = sum(p.numel() for p in model.parameters())
        x = torch.randn(1, 2, 96, 96, 48)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 2, 96, 96, 48)
        print(f"placement={placement:12s} | params={total:,} | output={y.shape} OK")
