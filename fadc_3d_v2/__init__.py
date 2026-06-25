"""3D-FADC v2 — spatial k_att + warm bias init + temperature annealing.

Background:
-----------
The original `fadc_3d/` is a faithful 3D port of FADC (CVPR 2024) EXCEPT for one
forced substitution: 2D FADC uses ModulatedDeformConv2d for per-spatial-location
adaptive dilation, an op that has no 3D equivalent. The original 3D port
substituted that with parallel dilated convs blended by softmax weights
computed from a global-average-pool descriptor — making the dilation choice
PER-IMAGE rather than PER-VOXEL.

Diagnostic runs on trained Bottleneck (s=42) and Deep (s=999) checkpoints
(2026-06-25) showed that this per-image softmax collapses to a hard one-hot
choice (k_att std < 0.001 across diverse inputs) in every block except enc3.
Channel and filter attention also collapsed to near-identity.

v2 fixes this with three changes, all isolated to this new directory:

  1. SPATIAL k_att — replaces the avgpool-fc bottleneck for k_att with a small
     Conv3d(3) + Conv3d(1) network that produces a per-voxel softmax over
     the dilation branches. Inspired by SAC (Switchable Atrous Conv,
     Qiao et al. CVPR 2021).
  2. WARM bias init — `channel_fc` and `filter_fc` biases initialized to 0.5
     instead of 0, so c_att and f_att start slightly off identity and have
     non-trivial gradients to learn from.
  3. TEMPERATURE annealing — k_att softmax temperature anneals from a high
     value (default 4.0 → mixed early) to 1.0 over training, preventing
     premature one-hot collapse.

Nothing in `fadc_3d/` is changed.
"""

from fadc_3d_v2.adaptive_dilated_conv_3d_v2 import AdaptiveDilatedConv3DV2
from fadc_3d_v2.omni_attention_3d_spatial import OmniAttention3DSpatial
from fadc_3d_v2.freq_select_3d import FrequencySelection3D

__all__ = [
    "AdaptiveDilatedConv3DV2",
    "OmniAttention3DSpatial",
    "FrequencySelection3D",
]
