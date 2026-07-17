"""AdaKern3D — kernel-side attention (corrected implementation).

Given ONE learnable base kernel

    W of shape (O, I, 3, 3, 3)                     # groups=1

it is decomposed additively as

    W_low  = W.mean(dim=(-3,-2,-1), keepdim=True)  # (O, I, 1, 1, 1)   — DC / smoothing
    W_high = W - W_low                             # (O, I, 3, 3, 3)   — high-freq residual
    W == W_low + W_high                            # exactly, by construction

AdaKern3D predicts, PER SAMPLE, four attention tensors from a pooled descriptor
of the input feature map:

    c_low  : (B, I, 1, 1, 1)     — input-channel gain on the low component
    f_low  : (B, O, 1, 1, 1)     — output-filter gain on the low component
    c_high : (B, I, 1, 1, 1)     — input-channel gain on the high component
    f_high : (B, O, 1, 1, 1)     — output-filter gain on the high component

which are combined into the per-sample adaptive kernel

    W_adaptive =
          W_low  * (2 * c_low ) * (2 * f_low )
        + W_high * (2 * c_high) * (2 * f_high)

with final shape (B, O, I, 3, 3, 3) via broadcasting.

Optional 27-position s_att (kernel-position attention)
------------------------------------------------------
An optional per-sample scalar-vector s_att of shape (B, 1, 1, 3, 3, 3) can be
applied AFTER the low/high reconstruction:

    W_adaptive = W_adaptive * (2 * s_att)

s_att is NOT voxelwise. It emits 27 numbers per sample, one for every position
in the 3x3x3 kernel. It is applied to the same base kernel — i.e. before the
conv branches — so every dilation choice uses the same s-modulated kernel.

Initialisation
--------------
- All predictor heads are zero-initialised (weights AND biases).
- With zero logits, every sigmoid produces 0.5, every multiplier `2 * sigmoid`
  produces 1.0. Therefore at init:

      W_adaptive = W_low * 1 * 1 + W_high * 1 * 1 = W_low + W_high = W

  and if s_att is enabled it multiplies by 1 everywhere, so still W.

Grouped convs
-------------
The corrected implementation supports groups=1 explicitly. Any other grouping
raises a clear error at construction — silent broadcasting to grouped weights
would misalign channels between the base kernel and the (c, f) attentions.
"""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


class AdaKern3D(nn.Module):
    """Kernel-side attention for a shared 3x3x3 base kernel.

    Args:
        in_channels    : I. Matches the base kernel's second dim.
        out_channels   : O. Matches the base kernel's first dim.
        reduction      : bottleneck ratio for the pooled descriptor -> attention.
        min_channel    : lower bound on the bottleneck width.
        use_position_att : if True, also emit a 27-position s_att for the
                           3x3x3 kernel and apply it in `apply_to_kernel`.
        groups         : only 1 is supported.
    """

    KERNEL_SIZE: int = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: float = 0.125,
        min_channel: int = 32,
        use_position_att: bool = False,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if groups != 1:
            raise NotImplementedError(
                f"AdaKern3D only supports groups=1 today, got groups={groups}. "
                "Grouped base kernels need per-group attention broadcasting; "
                "silently doing this today would misalign c/f channels."
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.use_position_att = bool(use_position_att)

        att_ch = max(int(min(in_channels, out_channels) * reduction), int(min_channel))
        self.att_ch = att_ch

        # ------------- pooled descriptor: (B, I, 1, 1, 1) shared trunk
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.shared_fc = nn.Conv3d(in_channels, att_ch, kernel_size=1, bias=False)
        self.shared_norm = nn.GroupNorm(1, att_ch)  # per-sample; safe with any B.
        self.shared_act = nn.ReLU(inplace=True)

        # ------------- four zero-logit heads
        # c_* heads emit (B, I, 1, 1, 1); f_* heads emit (B, O, 1, 1, 1).
        self.c_low_fc = nn.Conv3d(att_ch, in_channels, kernel_size=1, bias=True)
        self.f_low_fc = nn.Conv3d(att_ch, out_channels, kernel_size=1, bias=True)
        self.c_high_fc = nn.Conv3d(att_ch, in_channels, kernel_size=1, bias=True)
        self.f_high_fc = nn.Conv3d(att_ch, out_channels, kernel_size=1, bias=True)

        # ------------- optional 27-position s_att head, emits (B, 27, 1, 1, 1).
        if self.use_position_att:
            self.s_fc = nn.Conv3d(
                att_ch, self.KERNEL_SIZE ** 3, kernel_size=1, bias=True
            )
        else:
            self.s_fc = None

        self._zero_init_heads()

    # --------------------------------------------------------------------- init
    def _zero_init_heads(self) -> None:
        """Zero-init OUTPUT heads only so 2*sigmoid(0) = 1 identity at init.

        Head weight AND bias are both zero -> logits are exactly zero regardless
        of the trunk feature distribution. The shared trunk keeps its Kaiming
        init so gradients flow back into the head weights via the non-zero
        trunk output during the FIRST backward step.
        """
        for head in (self.c_low_fc, self.f_low_fc, self.c_high_fc, self.f_high_fc):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.s_fc is not None:
            nn.init.zeros_(self.s_fc.weight)
            nn.init.zeros_(self.s_fc.bias)
        # GroupNorm keeps its default gamma=1, beta=0.

    # --------------------------------------------------------------------- forward
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Predict (c_low, f_low, c_high, f_high, s_att) from a feature map.

        Returns tensors ready to broadcast against the base kernel components:
          c_low, c_high : (B, I, 1, 1, 1)
          f_low, f_high : (B, O, 1, 1, 1)
          s_att         : (B, 1, 1, 3, 3, 3) if use_position_att else None
        """
        if x.dim() != 5:
            raise ValueError(f"AdaKern3D expects 5D input, got {tuple(x.shape)}")
        b = x.size(0)

        h = self.avgpool(x)             # (B, I, 1, 1, 1)
        h = self.shared_fc(h)           # (B, att_ch, 1, 1, 1)
        h = self.shared_norm(h)
        h = self.shared_act(h)

        c_low = torch.sigmoid(self.c_low_fc(h))    # (B, I, 1, 1, 1)
        f_low = torch.sigmoid(self.f_low_fc(h))    # (B, O, 1, 1, 1)
        c_high = torch.sigmoid(self.c_high_fc(h))  # (B, I, 1, 1, 1)
        f_high = torch.sigmoid(self.f_high_fc(h))  # (B, O, 1, 1, 1)

        s_att = None
        if self.s_fc is not None:
            s_logits = self.s_fc(h)                # (B, 27, 1, 1, 1)
            s_att = torch.sigmoid(s_logits).view(
                b, 1, 1, self.KERNEL_SIZE, self.KERNEL_SIZE, self.KERNEL_SIZE
            )

        return c_low, f_low, c_high, f_high, s_att

    # --------------------------------------------------------------------- kernel builder
    @staticmethod
    def decompose_kernel(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split a base kernel W of shape (O, I, K, K, K) into (W_low, W_high).

            W_low  = W.mean(dim=(-3,-2,-1), keepdim=True)   # (O, I, 1, 1, 1)
            W_high = W - W_low                              # (O, I, K, K, K)

        Together W_low + W_high == W exactly.
        """
        W_low = W.mean(dim=(-3, -2, -1), keepdim=True)
        W_high = W - W_low
        return W_low, W_high

    def apply_to_kernel(
        self,
        W: torch.Tensor,
        c_low: torch.Tensor,
        f_low: torch.Tensor,
        c_high: torch.Tensor,
        f_high: torch.Tensor,
        s_att: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build the per-sample adaptive kernel of shape (B, O, I, 3, 3, 3).

        W shape: (O, I, 3, 3, 3), groups=1.

        Broadcasting layout for the low/high multipliers:

            W_low        : (   1, O, I, 1, 1, 1)   (unsqueeze batch on W_low)
            (2 * c_low)  : (B,    1, I, 1, 1, 1)
            (2 * f_low)  : (B,    O, 1, 1, 1, 1)

        Resulting term shape after broadcast: (B, O, I, 1, 1, 1) for the low
        term and (B, O, I, 3, 3, 3) for the high term. Their sum is
        (B, O, I, 3, 3, 3).

        s_att (if enabled): (B, 1, 1, 3, 3, 3), broadcasts across O and I.
        """
        if W.dim() != 5:
            raise ValueError(f"W must be (O,I,K,K,K), got {tuple(W.shape)}")
        O, I, K1, K2, K3 = W.shape
        if K1 != self.KERNEL_SIZE or K2 != self.KERNEL_SIZE or K3 != self.KERNEL_SIZE:
            raise ValueError(f"Only 3x3x3 base kernels are supported, got {(K1, K2, K3)}")
        B = c_low.size(0)

        W_low, W_high = self.decompose_kernel(W)              # (O,I,1,1,1) / (O,I,K,K,K)

        # Prepend a batch axis to the shared kernel components.
        W_low = W_low.unsqueeze(0)                            # (1, O, I, 1, 1, 1)
        W_high = W_high.unsqueeze(0)                          # (1, O, I, 3, 3, 3)

        # Reshape (c, f) attentions to broadcast:
        #   c_*: (B, I, 1, 1, 1) -> (B, 1, I, 1, 1, 1)
        #   f_*: (B, O, 1, 1, 1) -> (B, O, 1, 1, 1, 1)
        c_low = c_low.view(B, 1, I, 1, 1, 1)
        f_low = f_low.view(B, O, 1, 1, 1, 1)
        c_high = c_high.view(B, 1, I, 1, 1, 1)
        f_high = f_high.view(B, O, 1, 1, 1, 1)

        term_low = W_low * (2 * c_low) * (2 * f_low)          # (B, O, I, 1, 1, 1)
        term_high = W_high * (2 * c_high) * (2 * f_high)      # (B, O, I, K, K, K)
        W_adaptive = term_low + term_high                     # (B, O, I, K, K, K)

        if s_att is not None:
            # s_att: (B, 1, 1, K, K, K) -> broadcast across O, I.
            W_adaptive = W_adaptive * (2 * s_att)

        return W_adaptive
