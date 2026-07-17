"""Real-MRI mechanism diagnostic for the corrected FADC3D encoder.

Loads a validation-set patch (real MRI, not random noise) and, for every
corrected AdaptiveDilatedConv3D layer in the model, reports:

  * expected-dilation statistics (mean, std, min, max);
  * fraction of voxels whose argmax dilation is 1, 2 or 3;
  * k_att entropy in bits (per voxel, averaged);
  * k_att spatial variation (per-input std over voxels);
  * FrequencySelection band-weight statistics per band predictor;
  * c_low / f_low / c_high / f_high multiplier means, per-input stds, |g|;
  * s_att position variation (if enabled);
  * gradient norms into every mechanism head after a small diagnostic backward.

Optionally correlates expected_dilation with a local high-frequency-energy
proxy `|x - avg_pool3d(x)|`. A negative correlation (high-freq -> small dil)
is the expected tendency; treat this only as a diagnostic hint, not a training
target.

Usage:
    python diag_fadc_3d_correct.py \
        --ckpt path/to/best_model.pth \
        --preprocessed_cache path/to/val/*.npz \
        --n_patches 4 --patch_size 96 96 48
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from models.unet_3d_fadc_correct import (
    build_unet3d_fadc_correct, EXPECTED_ADAPTIVE_CONV_COUNT,
)
from fadc_3d_correct.adaptive_dilated_conv_3d import AdaptiveDilatedConv3D
from fadc_3d_correct.freq_select_3d import FrequencySelection3D


# ─────────────────────────── data helpers

def _load_random_val_patches(cache_dir: str, n_patches: int,
                              patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Load N random patches from the preprocessed val cache."""
    cache = Path(cache_dir)
    npzs = sorted(cache.glob("*.npz"))
    if not npzs:
        raise FileNotFoundError(f"No .npz files found under {cache}")
    rng = np.random.default_rng(0)
    picks = rng.choice(len(npzs), size=min(n_patches, len(npzs)), replace=False)
    xs = []
    for i in picks:
        d = np.load(npzs[i])
        img = d["image"].astype(np.float32)               # (C, D, H, W)
        c, D, H, W = img.shape
        pd, ph, pw = patch_size
        # crop a random valid corner
        d0 = rng.integers(0, max(1, D - pd + 1))
        h0 = rng.integers(0, max(1, H - ph + 1))
        w0 = rng.integers(0, max(1, W - pw + 1))
        patch = img[:, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]
        # pad if smaller
        pad = [(0, 0)] + [
            (0, max(0, pd - patch.shape[1])),
            (0, max(0, ph - patch.shape[2])),
            (0, max(0, pw - patch.shape[3])),
        ]
        patch = np.pad(patch, pad, mode="constant")
        patch = patch[:, :pd, :ph, :pw]
        xs.append(patch)
    return torch.from_numpy(np.stack(xs, axis=0))  # (B, C, D, H, W)


# ─────────────────────────── local high-freq energy

def high_freq_energy(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """|x - avg_pool3d(x)| — a simple local high-frequency energy proxy."""
    x = x.float()
    smoothed = F.avg_pool3d(x, kernel_size=k, stride=1, padding=k // 2)
    return (x - smoothed).abs()


# ─────────────────────────── reporting

def _stat(t: torch.Tensor) -> str:
    t = t.detach().float().flatten()
    return (f"mean={t.mean().item():+.4f} std={t.std(unbiased=False).item():.4f} "
            f"min={t.min().item():+.4f} max={t.max().item():+.4f}")


@torch.no_grad()
def report_layer(name: str, m: AdaptiveDilatedConv3D,
                 hf_energy: Optional[torch.Tensor] = None) -> dict:
    """Report per-layer mechanism statistics from the cached diag tensors."""
    out = {"layer": name}

    k_att = m.last_k_att.detach().float()          # (B, 3, D, H, W)
    ed = m.last_expected_dilation.detach().float() # (B, D, H, W)
    print(f"\n--- {name}")
    print(f"  expected_dilation   {_stat(ed)}")

    p = k_att.clamp(min=1e-12)
    ent = -(p * p.log()).sum(dim=1)                 # nats
    ent_bits = ent / np.log(2)
    print(f"  k_att entropy [bit] {_stat(ent_bits)}")

    arg = k_att.argmax(dim=1)
    total = arg.numel()
    frac = [(arg == i).float().sum().item() / total for i in range(3)]
    print(f"  argmax fractions    d1={frac[0]:.3f}  d2={frac[1]:.3f}  d3={frac[2]:.3f}")

    # k_att spatial variation per (batch, branch) then mean
    kspat = k_att.flatten(2).std(dim=2, unbiased=False).mean().item()
    print(f"  k_att spatial-std   mean-over-branches = {kspat:.4f}")

    # FreqSel — PRIMARY: actual runtime band multipliers 2*sigmoid(logits)
    # captured during the last forward. Shape per band: (B, sg, D, H, W).
    band_muls = getattr(m.fs, "last_band_multipliers", None) or []
    if band_muls:
        cutoffs = list(m.fs.k_list) + (["low"] if m.fs.lowfreq_att else [])
        for i, mul in enumerate(band_muls):
            mul_f = mul.detach().float()
            tag = f"band[k={cutoffs[i]}]" if i < len(cutoffs) else f"band[{i}]"
            print(f"  fs.{tag:<11s} mult  mean={mul_f.mean().item():+.4f} "
                  f"std={mul_f.std(unbiased=False).item():.4f} "
                  f"min={mul_f.min().item():+.4f} max={mul_f.max().item():+.4f}")
    # SECONDARY: predictor parameter statistics (indirectly informative).
    for i, conv in enumerate(m.fs.freq_weight_conv_list):
        w = conv.weight.detach()
        bb = conv.bias.detach()
        print(f"  fs.pred[{i}]     W: mean={w.mean().item():+.4f} std={w.std().item():.4f} "
              f"bias: mean={bb.mean().item():+.4f}")

    # c/f low/high multiplier statistics (per-batch std as an adaptivity proxy)
    for tag, t in [
        ("c_low",  m.last_c_low),
        ("f_low",  m.last_f_low),
        ("c_high", m.last_c_high),
        ("f_high", m.last_f_high),
    ]:
        if t is None:
            continue
        t = t.detach().float()
        mean_mul = (2 * t).mean().item()
        if t.size(0) > 1:
            per_input = t.flatten(1).mean(dim=1)
            inp_std = per_input.std(unbiased=False).item()
        else:
            inp_std = float("nan")
        print(f"  {tag:6s} 2*sig  mean-multiplier={mean_mul:.4f}  batch-std={inp_std:.4f}")

    if m.last_s_att is not None:
        s = m.last_s_att.detach().float()  # (B, 1, 1, 3, 3, 3)
        s_mul = 2 * s
        pos_std = s_mul.flatten(3).std(dim=3, unbiased=False).mean().item()
        print(f"  s_att positions   mean={s_mul.mean().item():+.4f} "
              f"per-input pos-std={pos_std:.4f}")

    # Optional: correlation between expected_dilation and local hf energy.
    if hf_energy is not None:
        # Downsample hf energy to layer resolution.
        _, _, D, H, W = ed.unsqueeze(1).shape
        hf_pool = F.adaptive_avg_pool3d(hf_energy.mean(dim=1, keepdim=True), (D, H, W))
        hf_flat = hf_pool.flatten().float()
        ed_flat = ed.flatten().float()
        n = min(hf_flat.numel(), ed_flat.numel())
        hf_flat = hf_flat[:n]; ed_flat = ed_flat[:n]
        # Pearson correlation.
        hf_c = hf_flat - hf_flat.mean()
        ed_c = ed_flat - ed_flat.mean()
        denom = (hf_c.pow(2).sum().sqrt() * ed_c.pow(2).sum().sqrt())
        corr = float((hf_c * ed_c).sum() / (denom + 1e-12))
        print(f"  corr(high_freq, E[dil]) = {corr:+.4f}  "
              f"(expected trend: negative)")
    return out


# ─────────────────────────── diag entrypoint

def diag(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt_path = args.ckpt
    if ckpt_path is None or not os.path.exists(ckpt_path):
        # Fall back to a freshly-initialised model — still useful to
        # sanity-check the report formatting on real MRI.
        print(f"WARNING: no checkpoint at {ckpt_path}, running on freshly-init encoder.")
        model = build_unet3d_fadc_correct(
            "unet3d_fadc_encoder_correct",
            in_channels=args.in_channels, out_channels=args.out_channels,
            base_filters=args.base_filters,
        ).to(device).eval()
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        arch = ckpt.get("arch_identity") or {}
        model_name = arch.get("model_name") or ckpt.get("model_name") or "unet3d_fadc_encoder_correct"
        base_filters = int(arch.get("base_filters", args.base_filters))
        deep_sup = bool(arch.get("deep_supervision", False))
        model = build_unet3d_fadc_correct(
            model_name,
            in_channels=int(arch.get("in_channels", args.in_channels)),
            out_channels=int(arch.get("out_channels", args.out_channels)),
            base_filters=base_filters,
            deep_supervision=deep_sup,
        ).to(device).eval()
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"loaded {model_name} epoch={ckpt.get('epoch')} best_dice={ckpt.get('best_dice')}")

    # Load real MRI patches.
    if args.preprocessed_cache:
        x = _load_random_val_patches(
            args.preprocessed_cache, n_patches=args.n_patches,
            patch_size=tuple(args.patch_size),
        ).to(device)
        print(f"patches from {args.preprocessed_cache}: shape={tuple(x.shape)}")
    else:
        print("WARNING: --preprocessed_cache not provided; falling back to random noise.")
        x = torch.randn(args.n_patches, 2, *args.patch_size, device=device)

    hf_e = high_freq_energy(x).detach()

    # 1) Non-graph pass for statistics.
    with torch.no_grad():
        _ = model(x)
    n = model.count_adaptive_convs()
    print(f"\nCorrected adaptive convs: {n}")
    print("=" * 74)
    for name, m in model.named_modules():
        if isinstance(m, AdaptiveDilatedConv3D):
            report_layer(name, m, hf_energy=hf_e)

    # 2) Small backward for gradient-norm sanity.
    print("\n" + "=" * 74)
    print("gradient norms (diag backward on y.pow(2).mean())")
    print("=" * 74)
    model.train()  # reactivate dropout so the backward is realistic
    x_ = x.clone().requires_grad_(False)
    y = model(x_)
    if isinstance(y, tuple):
        y = y[0]
    loss = y.float().pow(2).mean()
    model.zero_grad()
    loss.backward()

    def _g_norm(p) -> float:
        return float(p.grad.detach().norm().item()) if p.grad is not None else float("nan")

    for name, m in model.named_modules():
        if not isinstance(m, AdaptiveDilatedConv3D):
            continue
        norms = {
            "weight":     _g_norm(m.weight),
            "c_low_fc":   _g_norm(m.ada_kern.c_low_fc.weight),
            "f_low_fc":   _g_norm(m.ada_kern.f_low_fc.weight),
            "c_high_fc":  _g_norm(m.ada_kern.c_high_fc.weight),
            "f_high_fc":  _g_norm(m.ada_kern.f_high_fc.weight),
            "k_att_head": _g_norm(m.k_att.head.weight),
        }
        parts = "  ".join(f"{k}={v:.2e}" for k, v in norms.items())
        print(f"  {name:<30s}  {parts}")


# ─────────────────────────── CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--preprocessed_cache", type=str, default=None,
                   help="Directory of val .npz patches to draw real-MRI inputs from.")
    p.add_argument("--n_patches", type=int, default=4)
    p.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 48])
    p.add_argument("--in_channels", type=int, default=2)
    p.add_argument("--out_channels", type=int, default=2)
    p.add_argument("--base_filters", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    diag(parse_args())
