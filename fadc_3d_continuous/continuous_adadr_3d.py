"""ContinuousAdaDR3D — mathematically-faithful 3D extension of the CVPR 2024
FADC AdaDR mechanism.

Official 2D formulation (Chen et al. 2024, `FADC_only/conv_custom.py`)
======================================================================

For every spatial location p in a 2D feature map, the official code:

    raw_offset = conv_offset(PAD(x))                  # (B, G, H, W), G = deform_groups
    s          = raw_offset.abs() * base_dilation     # non-negative scalar

    dilated_offset = fixed 3x3 lattice, one (dy,dx) pair per kernel position
    offset_expanded = s.reshape(B, G, 1, H, W) * dilated_offset  # (B, G, 18, H, W)

    modulated_deform_conv2d(PAD(x), offset_expanded, mask, weight, ...)

produces sample positions

    sample_position(p, q) = p + q + s(p) * q = p + (1 + s(p)) * q

for every 3x3 kernel coordinate q ∈ {-1,0,1}^2.

3D extension implemented here
=============================

Same math, one axis added: q ∈ {-1,0,1}^3 (27 positions), isotropic scalar
s(p) applied to z, y AND x components of q, sampling via
`torch.nn.functional.grid_sample` in trilinear + border-padding mode.

    sample_position(p, q) = p + (1 + s(p)) * q       for every q ∈ {-1,0,1}^3
    D(p) = 1 + s(p)                                  isotropic per-voxel dilation

Isotropy is a deliberate choice because the preprocessing pipeline resamples
every volume to 1x1x1 mm physical voxel spacing.

Correctness / performance repairs vs earlier drafts
===================================================

- **Padding semantics** — conv_offset and conv_mask now use
  `padding=0` and consume `self.PAD(x)` (ReplicationPad3d(k//2)) exactly the
  way the official 2D code does. Preserves ambient boundary statistics into
  the offset/mask predictors instead of zero-padding.
- **Cached base coord grid** — the (pz, py, px) meshgrid is built once per
  (D, H, W, device, dtype) and reused across all 27 kernel positions and
  across subsequent forward passes at the same shape.
- **No CUDA .item() calls** — the 27-position lattice is stored as three
  Python-int lists (kzs, kys, kxs), consumed directly as scalars for the
  coordinate shifts and as `torch.long` buffers for weight indexing. No
  device-to-host syncs inside the forward loop.
- **Genuinely vectorized chunk_positions** — the loop batches
  `chunk_positions` kernel positions into one `grid_sample` call (grid
  reshape trick: (B, C_pos*D, H, W, 3)) and one `einsum` contraction, cutting
  per-forward CUDA launches by up to 27x. `chunk_positions=1` still exists
  and matches the old sequential path exactly. `chunk_positions=27` gives
  one grid_sample + one einsum per forward.
- **deform_groups > 1 rejected at construction** — the untested >1 code
  path is gone. Reintroducing it needs a real test suite, not a fallback.

grid_sample conventions used
============================

- Input tensor:  (B, C, D_in, H_in, W_in)
- Grid tensor:   (B, D_out, H_out, W_out, 3), coordinates in **(x, y, z)** order
- align_corners=True: -1 <-> pixel 0, +1 <-> pixel N-1  (avoids half-voxel shift
  for integer input positions)
- padding_mode='border' (replication) — matches the official 2D code's
  `padding_mode='repeat'` (ReplicationPad2d).

Initialization contract
=======================

- conv_offset weights zero, bias = (base_dilation - 1) / base_dilation + epsilon.
  For base_dilation=1: bias = epsilon = 1e-4, s ≈ 1e-4, effective D ≈ 1.
- conv_mask weights and bias both zero. sigmoid(0) = 0.5 mask at init.

Not implemented on purpose
==========================

- No bounded-[1,3] sigmoid dilation on the primary path (paper doesn't use it).
  Exposed only as `bounded_ablation=True` — clearly named, never used by the
  encoder-only model.
- No CUDA kernel: pure PyTorch via grid_sample. Verifiable, portable,
  works on CPU for tests.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────
# Static kernel-lattice buffer (27 positions of a 3x3x3 kernel)
# ─────────────────────────────────────────────────────────────────────────

def _kernel_lattice_3d_int_lists() -> Tuple[list[int], list[int], list[int]]:
    """Return three Python-int lists giving the 27 lattice coords.

    Ordering is (qz, qy, qx) with qz varying slowest, matching the natural
    memory layout of a Conv3d weight tensor `W[o, i, kd, kh, kw]`. Center
    coord (0, 0, 0) is included at index 13. Using Python ints (not device
    tensors) for the scalar shifts inside forward eliminates .item() syncs.
    """
    kzs, kys, kxs = [], [], []
    for qz in (-1, 0, 1):
        for qy in (-1, 0, 1):
            for qx in (-1, 0, 1):
                kzs.append(qz); kys.append(qy); kxs.append(qx)
    assert len(kzs) == 27 and len(kys) == 27 and len(kxs) == 27
    assert len({(kzs[i], kys[i], kxs[i]) for i in range(27)}) == 27
    return kzs, kys, kxs


def _kernel_lattice_3d() -> torch.Tensor:
    """Compatibility shim: return the same 27 (qz, qy, qx) coordinates as a
    (27, 3) tensor. Preserves the old callable name for any test that
    inspects the lattice directly."""
    kzs, kys, kxs = _kernel_lattice_3d_int_lists()
    return torch.tensor(list(zip(kzs, kys, kxs)), dtype=torch.float32)


class ContinuousAdaDR3D(nn.Module):
    """3D volumetric extension of the official 2D AdaDR deformable conv.

    Args:
        in_channels        : C_in of the conv this module lives inside.
        out_channels       : C_out.
        deform_groups      : G. Only 1 is supported; anything else raises.
        base_dilation      : the dilation the parent Conv3d would have used.
                             Feeds abs()*base_dilation parameterisation AND
                             the conv_offset bias init.
        padding_mode       : 'border' (default) matches the official code's
                             ReplicationPad2d default. 'zeros' available.
        align_corners      : True (default; avoids half-voxel shift on int coords).
        chunk_positions    : how many kernel positions to batch per grid_sample
                             call. 1 = fully sequential (minimum memory).
                             27 = one grid_sample per forward (maximum speed,
                             maximum memory). Chunking is genuinely vectorised.
        epsilon            : matches the official offset-bias epsilon.
        bounded_ablation   : optional ablation — replaces abs()*base_dilation
                             with a sigmoid-bounded [1, 3] curve. NEVER on the
                             primary path.
    """

    KERNEL_SIZE: int = 3
    N_LATTICE:   int = 27

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deform_groups: int = 1,
        base_dilation: int = 1,
        padding_mode: str = "border",
        align_corners: bool = True,
        chunk_positions: int = 1,
        epsilon: float = 1e-4,
        bounded_ablation: bool = False,
    ) -> None:
        super().__init__()
        if deform_groups != 1:
            raise NotImplementedError(
                f"ContinuousAdaDR3D only supports deform_groups=1 today, got "
                f"deform_groups={deform_groups}. Grouped deformable conv is a "
                "non-trivial fold across offset/mask channel layout and would "
                "need a dedicated test suite; adding a silent code path is "
                "worse than raising."
            )
        if padding_mode not in ("border", "zeros"):
            raise ValueError(f"padding_mode must be 'border' or 'zeros', got {padding_mode!r}")
        if not (1 <= int(chunk_positions) <= self.N_LATTICE):
            raise ValueError(
                f"chunk_positions must be in [1, {self.N_LATTICE}], got {chunk_positions}"
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.deform_groups = 1                  # locked (see NotImplementedError above)
        self.base_dilation = int(base_dilation)
        self.padding_mode = padding_mode
        self.align_corners = bool(align_corners)
        self.chunk_positions = int(chunk_positions)
        self.epsilon = float(epsilon)
        self.bounded_ablation = bool(bounded_ablation)

        # Official-style: predictors take PAD(x) at padding=0.
        self.PAD = nn.ReplicationPad3d(self.KERNEL_SIZE // 2)

        # Predict ONE scalar offset per voxel — matches official.
        self.conv_offset = nn.Conv3d(
            in_channels, self.deform_groups,
            kernel_size=self.KERNEL_SIZE,
            padding=0,
            bias=True,
        )
        # Predict 27 modulation-mask values per voxel.
        self.conv_mask = nn.Conv3d(
            in_channels, self.deform_groups * self.N_LATTICE,
            kernel_size=self.KERNEL_SIZE,
            padding=0,
            bias=True,
        )

        # Fixed lattice as Python-int lists (for scalar shifts) AND as long
        # tensors registered as buffers (for weight-tensor indexing). Both
        # kept in sync — same 27 positions in the same order.
        _kzs, _kys, _kxs = _kernel_lattice_3d_int_lists()
        self._kzs_py: list[int] = list(_kzs)
        self._kys_py: list[int] = list(_kys)
        self._kxs_py: list[int] = list(_kxs)
        # Kernel index (kz+1, ky+1, kx+1) into weight[..., 3, 3, 3].
        self.register_buffer(
            "kzs_idx", torch.tensor([k + 1 for k in _kzs], dtype=torch.long)
        )
        self.register_buffer(
            "kys_idx", torch.tensor([k + 1 for k in _kys], dtype=torch.long)
        )
        self.register_buffer(
            "kxs_idx", torch.tensor([k + 1 for k in _kxs], dtype=torch.long)
        )

        # Base coord grid cache: (D, H, W, device, dtype) -> (pz, py, px)
        # each of shape (D, H, W). Rebuilt only when the input spatial shape
        # or device/dtype changes.
        self._grid_cache: dict = {}

        # Diagnostic caches — populated on every forward. Detached (no graph).
        self.last_scalar_s: Optional[torch.Tensor] = None            # (B, D, H, W)
        self.last_effective_dilation: Optional[torch.Tensor] = None  # (B, D, H, W)
        self.last_mask: Optional[torch.Tensor] = None                # (B, 27, D, H, W)

        self._init_weights()

    # --------------------------------------------------------------- lattice compat
    @property
    def lattice(self) -> torch.Tensor:
        """Compatibility view: return the 27 (qz, qy, qx) coords as a
        (27, 3) float32 CPU tensor. Derived from the Python-int lists so
        no state is duplicated. Handy for tests and diagnostics; NOT on the
        fast forward path."""
        return torch.tensor(
            list(zip(self._kzs_py, self._kys_py, self._kxs_py)),
            dtype=torch.float32,
        )

    # --------------------------------------------------------------- init
    def _init_weights(self) -> None:
        """Paper-faithful init.

        conv_offset weights zero + bias = (base_dilation - 1)/base_dilation + epsilon.
        For base_dilation=1: bias = epsilon = 1e-4, s ≈ 1e-4, effective D ≈ 1.
        conv_mask both zero. sigmoid(0) = 0.5 mask at init.
        """
        nn.init.zeros_(self.conv_offset.weight)
        bias_val = (self.base_dilation - 1) / self.base_dilation + self.epsilon
        nn.init.constant_(self.conv_offset.bias, bias_val)
        nn.init.zeros_(self.conv_mask.weight)
        nn.init.zeros_(self.conv_mask.bias)

    # --------------------------------------------------------------- helpers
    def _get_base_grid(
        self, D: int, H: int, W: int,
        device: torch.device, dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cached (pz, py, px) meshgrids of shape (D, H, W).

        Recomputes only when the (D, H, W, device, dtype) key is new. On a
        typical U-Net encoder the shape at each layer is stable across the
        run, so this cache stays warm from the second forward onward.
        """
        key = (int(D), int(H), int(W), device, dtype)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        z_lin = torch.arange(D, device=device, dtype=dtype)
        y_lin = torch.arange(H, device=device, dtype=dtype)
        x_lin = torch.arange(W, device=device, dtype=dtype)
        pz, py, px = torch.meshgrid(z_lin, y_lin, x_lin, indexing="ij")
        # Detach for safety — coord grid is a constant.
        cached = (pz.detach(), py.detach(), px.detach())
        # Bound cache size to avoid retaining stale shapes across many runs.
        if len(self._grid_cache) > 8:
            self._grid_cache.clear()
        self._grid_cache[key] = cached
        return cached

    def _scalar_offset(self, x_pad: torch.Tensor) -> torch.Tensor:
        """Predict s(p) of shape (B, D, H, W) from pre-padded input."""
        raw = self.conv_offset(x_pad)                       # (B, 1, D, H, W)
        if self.bounded_ablation:
            # Bounded ABLATION only: D bounded to (1, 3). Never on primary path.
            s = 2.0 * torch.sigmoid(raw)                    # in (0, 2), so D in (1, 3)
        else:
            s = raw.abs() * float(self.base_dilation)       # official parameterization
        return s.squeeze(1)                                  # (B, D, H, W)

    def _norm_coord(self, coord: torch.Tensor, N: int) -> torch.Tensor:
        """Map integer voxel coord to [-1, 1] with align_corners=True.

        Guards N=1 by returning zero (single-slice case; degenerate but must
        not divide by zero).
        """
        if N > 1:
            return 2.0 * coord / (N - 1) - 1.0
        return torch.zeros_like(coord)

    def _build_chunk_grid(
        self,
        s_shared: torch.Tensor,        # (B, D, H, W)
        chunk_kzs: list[int],
        chunk_kys: list[int],
        chunk_kxs: list[int],
        input_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Build a batched grid for `P = len(chunk_kzs)` kernel positions.

        Returns tensor of shape (B, P*D, H, W, 3) suitable for a single
        grid_sample call on a (B, C_in, D, H, W) input. Coordinates in
        (x, y, z) order.
        """
        B, D, H, W = s_shared.shape
        D_in, H_in, W_in = input_shape
        P = len(chunk_kzs)
        device, dtype = s_shared.device, s_shared.dtype

        pz, py, px = self._get_base_grid(D_in, H_in, W_in, device, dtype)  # (D,H,W)

        # Move axis-scale factor to (1, P, 1, 1, 1) — Python ints, no sync.
        # We use s_shared broadcast as (B, 1, D, H, W).
        s_b = s_shared.unsqueeze(1)                                          # (B, 1, D, H, W)
        one_plus_s = (1.0 + s_b)                                             # (B, 1, D, H, W)

        # For each axis: shift = (1 + s) * q_axis  where q_axis broadcasts (P,) -> (1,P,1,1,1)
        qz_t = torch.as_tensor(chunk_kzs, device=device, dtype=dtype).view(1, P, 1, 1, 1)
        qy_t = torch.as_tensor(chunk_kys, device=device, dtype=dtype).view(1, P, 1, 1, 1)
        qx_t = torch.as_tensor(chunk_kxs, device=device, dtype=dtype).view(1, P, 1, 1, 1)

        # base coord (broadcast over B, P): pz (D,H,W) -> (1,1,D,H,W)
        pz_b = pz.view(1, 1, D_in, H_in, W_in)
        py_b = py.view(1, 1, D_in, H_in, W_in)
        px_b = px.view(1, 1, D_in, H_in, W_in)

        sample_z = pz_b + one_plus_s * qz_t                                  # (B, P, D, H, W)
        sample_y = py_b + one_plus_s * qy_t
        sample_x = px_b + one_plus_s * qx_t

        gz = self._norm_coord(sample_z, D_in)                                # (B, P, D, H, W)
        gy = self._norm_coord(sample_y, H_in)
        gx = self._norm_coord(sample_x, W_in)

        # grid_sample expects (x, y, z) order in the last dim. Fold P into D.
        # Shape target: (B, P*D, H, W, 3).
        grid = torch.stack([gx, gy, gz], dim=-1)                             # (B, P, D, H, W, 3)
        grid = grid.reshape(B, P * D, H, W, 3)
        return grid

    # --------------------------------------------------------------- test helper
    def _sample_at_q(
        self,
        x: torch.Tensor,               # (B, C_in, D, H, W)
        s: torch.Tensor,               # (B, 1, D, H, W) or (B, D, H, W)
        q_zyx: torch.Tensor,           # (3,) — one (qz, qy, qx)
    ) -> torch.Tensor:
        """Sample `x` at position p + (1 + s(p)) * q for a SINGLE lattice
        coordinate. Returns (B, C_in, D, H, W).

        Not on the fast forward path — kept for tests and diagnostics that
        want to interrogate one position at a time.
        """
        if s.dim() == 5:
            s_shared = s.squeeze(1)                   # (B, D, H, W)
        elif s.dim() == 4:
            s_shared = s
        else:
            raise ValueError(f"s must be 4D or 5D, got {tuple(s.shape)}")
        # Use the vectorised chunk-1 grid builder with Python-int q components.
        qz = int(q_zyx[0].item())
        qy = int(q_zyx[1].item())
        qx = int(q_zyx[2].item())
        _, _, D, H, W = x.shape
        grid = self._build_chunk_grid(
            s_shared=s_shared,
            chunk_kzs=[qz], chunk_kys=[qy], chunk_kxs=[qx],
            input_shape=(D, H, W),
        )                                              # (B, D, H, W, 3)
        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

    # --------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,                # (B, C_in, D, H, W)
        weight: torch.Tensor,           # (B, C_out, C_in, 3, 3, 3)
        bias: Optional[torch.Tensor] = None,  # (C_out,)
    ) -> torch.Tensor:
        """Compute one continuous-dilation modulated conv step.

        `weight` is per-sample (AdaKern-modulated) with an explicit leading
        batch dim. `bias` is optional and shared across the batch.
        """
        if x.dim() != 5:
            raise ValueError(f"x must be 5D (B,C,D,H,W); got {tuple(x.shape)}")
        if weight.dim() != 6:
            raise ValueError(
                f"weight must be 6D (B,O,I,K,K,K); got {tuple(weight.shape)}"
            )
        B, C_in, D, H, W = x.shape
        Bw, C_out, C_inw, K1, K2, K3 = weight.shape
        if Bw != B:
            raise ValueError(f"weight batch {Bw} != input batch {B}")
        if C_inw != C_in:
            raise ValueError(f"weight C_in {C_inw} != input C_in {C_in}")
        if (K1, K2, K3) != (self.KERNEL_SIZE,) * 3:
            raise ValueError(f"weight must be 3x3x3, got {(K1, K2, K3)}")

        # 1) Predict s(p) and modulation mask m(p, q). Predictors run on the
        #    replication-padded feature map with padding=0 on the conv, matching
        #    the official 2D code (ReplicationPad2d + padding=0 on conv_offset).
        x_pad = self.PAD(x)                                                     # (B, C, D+2, H+2, W+2)
        s = self._scalar_offset(x_pad)                                          # (B, D, H, W)
        m = torch.sigmoid(self.conv_mask(x_pad))                                # (B, 27, D, H, W)

        # 2) Cache diagnostics (detached).
        self.last_scalar_s = s.detach()
        self.last_effective_dilation = (1.0 + s).detach()
        self.last_mask = m.detach()

        # 3) Vectorised chunked sampling + accumulation.
        out = torch.zeros(B, C_out, D, H, W, dtype=x.dtype, device=x.device)

        # Long-tensor buffers for weight indexing.
        kz_idx = self.kzs_idx.to(weight.device)     # (27,)
        ky_idx = self.kys_idx.to(weight.device)
        kx_idx = self.kxs_idx.to(weight.device)

        n_chunks = (self.N_LATTICE + self.chunk_positions - 1) // self.chunk_positions
        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.chunk_positions
            end = min(start + self.chunk_positions, self.N_LATTICE)
            P = end - start

            # (a) Grid for the chunk (batched over P). Python-int scalars.
            grid = self._build_chunk_grid(
                s_shared=s,
                chunk_kzs=self._kzs_py[start:end],
                chunk_kys=self._kys_py[start:end],
                chunk_kxs=self._kxs_py[start:end],
                input_shape=(D, H, W),
            )                                                                    # (B, P*D, H, W, 3)

            # (b) One grid_sample call for the whole chunk.
            sampled = F.grid_sample(
                x, grid, mode="bilinear", padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )                                                                    # (B, C_in, P*D, H, W)
            sampled = sampled.view(B, C_in, P, D, H, W)

            # (c) Slice mask for this chunk. deform_groups locked to 1 → drop G.
            m_chunk = m[:, start:end]                                            # (B, P, D, H, W)
            modulated = sampled * m_chunk.unsqueeze(1)                           # (B, C_in, P, D, H, W)

            # (d) Slice weight for this chunk.
            #     weight shape: (B, O, I, 3, 3, 3). Index (kz+1, ky+1, kx+1)
            #     for each of the P positions in the chunk.
            kz_c = kz_idx[start:end]
            ky_c = ky_idx[start:end]
            kx_c = kx_idx[start:end]
            W_chunk = weight[:, :, :, kz_c, ky_c, kx_c]                          # (B, O, I, P)

            # (e) Contract over C_in and P.
            #     modulated: (B, I, P, D, H, W)  ->  (b, i, p, d, h, w)
            #     W_chunk:   (B, O, I, P)         ->  (b, o, i, p)
            #     out       : (B, O, D, H, W)
            contrib = torch.einsum("boip,bipdhw->bodhw", W_chunk, modulated)
            out = out + contrib

            del sampled, modulated, contrib, grid, m_chunk, W_chunk

        if bias is not None:
            out = out + bias.view(1, -1, 1, 1, 1)
        return out


# ─────────────────────────────────────────────────────────────────────────
# Static metadata for checkpoint identity + downstream verification.
# ─────────────────────────────────────────────────────────────────────────

CONTINUOUS_ADADR3D_META = {
    "implementation":     "continuous_adadr3d",
    "sampling_equation":  "p + (1 + s(p)) * q  for q in {-1,0,1}^3",
    "isotropic":          True,
    "base_dilation":      1,
    "padding_mode":       "border",       # replication
    "align_corners":      True,
    "kernel_size":        3,
    "n_lattice":          27,
    "offset_param":       "abs(raw) * base_dilation  (paper default)",
    "mask_activation":    "sigmoid",
    "grid_sample_coord_order": "(x, y, z)",
    "predictor_pad":      "ReplicationPad3d(1) + conv(padding=0)",
    "deform_groups":      1,
}
