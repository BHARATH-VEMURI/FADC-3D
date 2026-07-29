"""Continuous isotropic FADC3D: a volumetric extension of FADC.

Preserves the discrete `fadc_3d_correct` package as a reproducible baseline
and adds a new, mathematically-faithful continuous-dilation extension that
mirrors the CVPR 2024 FADC paper's AdaDR mechanism:

    sample_position(p, q) = p + q + s(p) * q  =  p + (1 + s(p)) * q
    effective_dilation D(p) = 1 + s(p)

where p is a voxel index, q ∈ {-1,0,1}³ is a 3x3x3 kernel lattice coordinate,
and s(p) is a non-negative scalar predicted from the input feature map.

The scientific name of this implementation is:

    "Continuous isotropic FADC3D: a volumetric extension of FADC for
     3D breast MRI segmentation."

It is NOT called an author-provided implementation — the CVPR 2024 paper
and its official reference code are 2D. This package extends the same
math to 3D volumes.
"""
from fadc_3d_continuous.continuous_adadr_3d import ContinuousAdaDR3D
from fadc_3d_continuous.continuous_dilated_conv_3d import (
    ContinuousDilatedConv3D, CONTINUOUS_ADADR3D_META,
)
from fadc_3d_continuous.ada_kernel_3d_official import AdaKern3DOfficial
# Re-export FreqSelect3D from the discrete package — the FFT math is a
# direct 3D lift of the official 2D module and the discrete tests have
# already validated its identity guarantees.
from fadc_3d_correct.freq_select_3d import FrequencySelection3D

__all__ = [
    "ContinuousAdaDR3D",
    "ContinuousDilatedConv3D",
    "CONTINUOUS_ADADR3D_META",
    "AdaKern3DOfficial",
    "FrequencySelection3D",
]
