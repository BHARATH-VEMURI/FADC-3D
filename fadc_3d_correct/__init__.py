"""fadc_3d_correct — corrected discrete-3D FADC implementation.

Method: "Discrete voxelwise 3D frequency-adaptive dilation with shared kernels."

Not continuous AdaDR, not deformable Conv3d. Adaptively mixes fixed isotropic
dilation choices [1, 2, 3] at every voxel, using ONE shared learnable base
kernel per adaptive convolution.

Public surface:
  FrequencySelection3D      — FFT-based 3D band decomposition + spatial reweight.
  AdaKern3D                 — c_low/f_low/c_high/f_high kernel-side attention.
  AdaptiveDilatedConv3D     — one base kernel, three dilations, voxelwise k_att mix.
"""

from fadc_3d_correct.freq_select_3d import FrequencySelection3D
from fadc_3d_correct.ada_kernel_3d import AdaKern3D
from fadc_3d_correct.adaptive_dilated_conv_3d import AdaptiveDilatedConv3D

__all__ = [
    "FrequencySelection3D",
    "AdaKern3D",
    "AdaptiveDilatedConv3D",
]
