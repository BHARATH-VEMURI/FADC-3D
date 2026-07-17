"""FrequencySelection3D (corrected) — 3D band decomposition + per-band spatial
reweight.

Signal path
-----------
    x  --FFT-->  X  --band masks-->  {B_hi_2, B_hi_4, B_hi_8, B_low}
                                     |
                                     v
                                 iFFT each band
                                     |
                                     v
              sum ( sp_act(conv_i(x)) * band_i for i in [k_list] )
              + band_low                     (if lowfreq_att=False)

- Bands are constructed as a nested lowpass hierarchy on the shifted spectrum:
  cutoff k=2  keeps the central [-N/(2k), N/(2k)) fraction, k=4 halves that
  window again, k=8 halves it again. High-freq band[i] = pre_x - low_part[i].
  The last low-frequency residual is either passed through unchanged
  (lowfreq_att=False, spec default) or reweighted by its own predictor.

Identity guarantee
------------------
- Every `freq_weight_conv_list` module is zero-initialised (weights AND bias).
- `sp_act` is `sigmoid * 2`.  sigmoid(0)*2 = 1.  With identity weights and
  lowfreq_att=False the output equals sum(band_i) + band_low = x, so
  FrequencySelection3D at init reconstructs the input exactly (up to FFT
  numerical noise).

Gradient flow
-------------
- Every band predictor is applied to `att_feat` (defaults to `x`), so
  segmentation gradients reach every conv in `freq_weight_conv_list`.
- The FFT branch runs in float32 under AMP (ComplexHalf is unsupported)
  then casts the reconstructed features back to the input dtype.

Cache policy
------------
- Frequency masks depend on (device, dtype, spatial shape). We cache with
  those three as the key. A test asserts identity reconstruction on odd
  and even sizes to catch cache-key mismatches.

Not implemented (kept out on purpose)
-------------------------------------
- Softmax activation (unused by any experiment).
- avgpool decomposition path (kept only in the historical v2 module).
- global_selection scalar gate.
- fs_feat != 'feat' (att_feat can still be passed at forward time).
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencySelection3D(nn.Module):
    """3D FADC frequency-selection module (corrected implementation).

    Args:
        in_channels     : channel dim of the input.
        k_list          : lowpass cutoff denominators. Default [2, 4, 8].
                          Each entry keeps a centre cuboid of side
                          N / k in the fftshifted spectrum.
        lowfreq_att     : if True, reweight the final low-frequency residual
                          with its own predictor; if False (default) pass it
                          through so identity init reconstructs x.
        lp_type         : 'freq' (only supported value here) — FFT decomposition.
        act             : 'sigmoid' -> multiplier is sigmoid * 2 (default).
                          Zero-logit init therefore gives identity.
        spatial_group   : channel-group count for the band-weight predictors.
                          1 (default) = share the same weight across channels;
                          >1 groups channels so each subset has its own weight
                          field. Must divide in_channels.
        spatial_kernel  : spatial kernel size of every band-weight predictor
                          (default 3).
    """

    def __init__(
        self,
        in_channels: int,
        k_list=(2, 4, 8),
        lowfreq_att: bool = False,
        lp_type: str = "freq",
        act: str = "sigmoid",
        spatial_group: int = 1,
        spatial_kernel: int = 3,
    ) -> None:
        super().__init__()
        if lp_type != "freq":
            raise NotImplementedError("Only lp_type='freq' is supported.")
        if act != "sigmoid":
            raise NotImplementedError("Only act='sigmoid' is supported.")
        if spatial_group < 1:
            raise ValueError("spatial_group must be >= 1")
        if in_channels % spatial_group != 0:
            raise ValueError(
                f"spatial_group ({spatial_group}) must divide in_channels "
                f"({in_channels}); adaptive per-group broadcasting won't work otherwise."
            )

        self.in_channels = int(in_channels)
        self.k_list = tuple(int(k) for k in k_list)
        self.lowfreq_att = bool(lowfreq_att)
        self.spatial_group = int(spatial_group)
        self.act = act

        n_predictors = len(self.k_list) + (1 if self.lowfreq_att else 0)
        self.freq_weight_conv_list = nn.ModuleList()
        for _ in range(n_predictors):
            conv = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.spatial_group,
                kernel_size=spatial_kernel,
                stride=1,
                padding=spatial_kernel // 2,
                groups=self.spatial_group,
                bias=True,
            )
            # Zero-init: sigmoid(0)*2 = 1 identity multiplier at init.
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            self.freq_weight_conv_list.append(conv)

        # {(device, dtype, D, H, W): [mask_k for k in k_list]}. Bounded by the
        # small number of feature-map shapes a UNet sees per model.
        self._mask_cache: dict = {}

        # Diagnostic cache — filled on every forward with the ACTUAL runtime
        # band multipliers `2 * sigmoid(logits)` per band predictor. Detached
        # so the diag reporter can read them without retaining a graph.
        # List length matches len(k_list) + int(lowfreq_att); each entry has
        # shape (B, spatial_group, D, H, W).
        self.last_band_multipliers: list[torch.Tensor] = []

    # ---------------------------------------------------------------- helpers
    def _sp_act(self, logits: torch.Tensor) -> torch.Tensor:
        # sigmoid * 2 makes zero-logit init a true identity.
        return torch.sigmoid(logits) * 2.0

    def _band_masks(
        self, shape: Tuple[int, int, int], device: torch.device, dtype: torch.dtype
    ) -> list[torch.Tensor]:
        """Return the list of centred cuboid low-pass masks for each cutoff in k_list.

        Each mask has shape (1, 1, D, H, W). Cached per (device, dtype, shape).
        """
        d, h, w = shape
        key = (str(device), dtype, d, h, w)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        masks = []
        for k in self.k_list:
            m = torch.zeros(1, 1, d, h, w, device=device, dtype=dtype)
            # Centred cuboid of side N/k around the DC-shifted centre.
            # Use round() so that odd and even sizes behave symmetrically.
            d0 = int(round(d / 2 - d / (2 * k)))
            d1 = int(round(d / 2 + d / (2 * k)))
            h0 = int(round(h / 2 - h / (2 * k)))
            h1 = int(round(h / 2 + h / (2 * k)))
            w0 = int(round(w / 2 - w / (2 * k)))
            w1 = int(round(w / 2 + w / (2 * k)))
            d0 = max(0, d0); d1 = min(d, d1)
            h0 = max(0, h0); h1 = min(h, h1)
            w0 = max(0, w0); w1 = min(w, w1)
            m[:, :, d0:d1, h0:h1, w0:w1] = 1.0
            masks.append(m)
        self._mask_cache[key] = masks
        return masks

    # ------------------------------------------------------------------ main
    def forward(
        self, x: torch.Tensor, att_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected 5D (B,C,D,H,W), got shape {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"in_channels mismatch: module {self.in_channels}, tensor {x.size(1)}"
            )

        if att_feat is None:
            att_feat = x

        b, c, d, h, w = x.shape
        sg = self.spatial_group

        # -- FFT in float32 (ComplexHalf is unsupported).
        x_f32 = x.float()
        X = torch.fft.fftn(x_f32, dim=(-3, -2, -1), norm="ortho")
        X = torch.fft.fftshift(X, dim=(-3, -2, -1))

        masks = self._band_masks((d, h, w), x.device, x_f32.dtype)

        x_list: list[torch.Tensor] = []
        band_muls: list[torch.Tensor] = []                                  # for diagnostics
        pre_x = x
        for idx, mask in enumerate(masks):
            X_low = X * mask
            low_part = torch.fft.ifftn(
                torch.fft.ifftshift(X_low, dim=(-3, -2, -1)),
                dim=(-3, -2, -1),
                norm="ortho",
            ).real.to(x.dtype)
            high_part = pre_x - low_part
            pre_x = low_part

            fw_logits = self.freq_weight_conv_list[idx](att_feat)          # (B, sg, D, H, W)
            fw = self._sp_act(fw_logits)                                    # (B, sg, D, H, W)
            band_muls.append(fw.detach())                                   # no graph retained
            # Broadcast fw across channels-per-group.
            tmp = fw.reshape(b, sg, 1, d, h, w) * high_part.reshape(b, sg, -1, d, h, w)
            x_list.append(tmp.reshape(b, c, d, h, w))

        if self.lowfreq_att:
            fw_logits = self.freq_weight_conv_list[len(self.k_list)](att_feat)
            fw = self._sp_act(fw_logits)
            band_muls.append(fw.detach())                                   # no graph retained
            tmp = fw.reshape(b, sg, 1, d, h, w) * pre_x.reshape(b, sg, -1, d, h, w)
            x_list.append(tmp.reshape(b, c, d, h, w))
        else:
            x_list.append(pre_x)

        # Publish the runtime multipliers for diagnostics AFTER the forward
        # arithmetic is complete. Detached above, so no graph is retained.
        self.last_band_multipliers = band_muls

        return sum(x_list)
