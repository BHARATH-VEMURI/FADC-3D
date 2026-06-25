"""FrequencySelection3D — verbatim copy of fadc_3d/freq_select_3d.py.

Duplicated here so the v2 package is self-contained and the original is not
imported at runtime by v2 code. The frequency-selection pillar already works
correctly (see project_frequency_analysis memory); v2 does not modify it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencySelection3D(nn.Module):
    """
    3D extension of FrequencySelection from FADC (CVPR 2024).

    Decomposes a 3D feature map into frequency bands (high / low) and
    learns a spatially-varying weight for each band, allowing the network
    to selectively amplify or suppress specific frequency content before
    the dilated convolution.

    Two decomposition modes are supported:
      lp_type='freq'    — FFT-based: exact frequency-domain masking via fftn/ifftn
      lp_type='avgpool' — Spatial average-pooling approximation (cheaper, less precise)

    Key 3D changes vs the 2D original:
      fft2  / ifft2  → fftn  / ifftn  on dim=(-3, -2, -1)
      fftshift       → fftshift        on dim=(-3, -2, -1)
      2D centre mask → 3D cuboid mask  (d0:d1, h0:h1, w0:w1)
      Conv2d         → Conv3d
      ReplicationPad2d → ReplicationPad3d
      AvgPool2d      → AvgPool3d
    """

    def __init__(self,
                 in_channels,
                 k_list=[2],
                 lowfreq_att=True,
                 fs_feat='feat',
                 lp_type='freq',
                 act='sigmoid',
                 spatial='conv',
                 spatial_group=1,
                 spatial_kernel=3,
                 init='zero',
                 global_selection=False):
        super().__init__()
        self.k_list = k_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.fs_feat = fs_feat
        self.lp_type = lp_type
        self.in_channels = in_channels
        if spatial_group > 64:
            spatial_group = in_channels
        self.spatial_group = spatial_group
        self.lowfreq_att = lowfreq_att
        self.act = act

        if spatial != 'conv':
            raise NotImplementedError

        _n = len(k_list) + (1 if lowfreq_att else 0)
        for _ in range(_n):
            conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=self.spatial_group,
                stride=1,
                kernel_size=spatial_kernel,
                groups=self.spatial_group,
                padding=spatial_kernel // 2,
                bias=True)
            if init == 'zero':
                conv.weight.data.zero_()
                conv.bias.data.zero_()
            self.freq_weight_conv_list.append(conv)

        if lp_type == 'avgpool':
            for k in k_list:
                self.lp_list.append(nn.Sequential(
                    nn.ReplicationPad3d(k // 2),
                    nn.AvgPool3d(kernel_size=k, padding=0, stride=1)))
        elif lp_type == 'freq':
            pass
        else:
            raise NotImplementedError

    def sp_act(self, freq_weight):
        if self.act == 'sigmoid':
            return freq_weight.sigmoid() * 2
        elif self.act == 'softmax':
            return freq_weight.softmax(dim=1) * freq_weight.shape[1]
        raise NotImplementedError

    def forward(self, x, att_feat=None):
        if att_feat is None:
            att_feat = x
        b, c, d, h, w = x.shape
        sg = self.spatial_group
        x_list = []

        if self.lp_type == 'avgpool':
            pre_x = x
            for idx, avg in enumerate(self.lp_list):
                low_part = avg(x)
                high_part = pre_x - low_part
                pre_x = low_part
                fw = self.sp_act(self.freq_weight_conv_list[idx](att_feat))
                tmp = fw.reshape(b, sg, -1, d, h, w) * high_part.reshape(b, sg, -1, d, h, w)
                x_list.append(tmp.reshape(b, -1, d, h, w))
            if self.lowfreq_att:
                fw = self.sp_act(self.freq_weight_conv_list[len(x_list)](att_feat))
                tmp = fw.reshape(b, sg, -1, d, h, w) * pre_x.reshape(b, sg, -1, d, h, w)
                x_list.append(tmp.reshape(b, -1, d, h, w))
            else:
                x_list.append(pre_x)

        elif self.lp_type == 'freq':
            pre_x = x.clone()
            # FFT requires float32 — ComplexHalf (from AMP float16) is unsupported
            x_f32 = x.float()
            x_fft = torch.fft.fftshift(
                torch.fft.fftn(x_f32, dim=(-3, -2, -1), norm='ortho'),
                dim=(-3, -2, -1))

            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros(b, 1, d, h, w, device=x.device)
                d0, d1 = round(d/2 - d/(2*freq)), round(d/2 + d/(2*freq))
                h0, h1 = round(h/2 - h/(2*freq)), round(h/2 + h/(2*freq))
                w0, w1 = round(w/2 - w/(2*freq)), round(w/2 + w/(2*freq))
                mask[:, :, d0:d1, h0:h1, w0:w1] = 1.0

                low_part = torch.fft.ifftn(
                    torch.fft.ifftshift(x_fft * mask, dim=(-3, -2, -1)),
                    dim=(-3, -2, -1), norm='ortho').real.to(x.dtype)
                high_part = pre_x - low_part
                pre_x = low_part

                fw = self.sp_act(self.freq_weight_conv_list[idx](att_feat))
                tmp = fw.reshape(b, sg, -1, d, h, w) * high_part.reshape(b, sg, -1, d, h, w)
                x_list.append(tmp.reshape(b, -1, d, h, w))

            if self.lowfreq_att:
                fw = self.sp_act(self.freq_weight_conv_list[len(x_list)](att_feat))
                tmp = fw.reshape(b, sg, -1, d, h, w) * pre_x.reshape(b, sg, -1, d, h, w)
                x_list.append(tmp.reshape(b, -1, d, h, w))
            else:
                x_list.append(pre_x)

        return sum(x_list)
