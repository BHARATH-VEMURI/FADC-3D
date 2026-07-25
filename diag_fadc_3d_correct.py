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

Usage (legacy, physical-directory mode — kept for backward compatibility):
    python diag_fadc_3d_correct.py \
        --ckpt path/to/best_model.pth \
        --preprocessed_cache path/to/val/*.npz \
        --n_patches 4 --patch_size 96 96 48

Usage (manifest mode — REQUIRED for the 70/10/20 experiments, since the
old train/ and val/ physical dirs mix logical partitions):
    python diag_fadc_3d_correct.py \
        --ckpt path/to/best_model.pth \
        --split_manifest path/to/split_70_10_20_seed42.csv \
        --split_partition val \
        --preprocessed_cache_dir path/to/cache_root \
        --n_patches 4 --patch_size 96 96 48

In manifest mode the diagnostic ONLY reads .npz files whose patient_id
belongs to the requested logical partition — the physical directory is
never scanned. This prevents test-set leakage when val and test patients
share the same on-disk folder.
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

def _extract_patch(img: np.ndarray, patch_size: tuple[int, int, int],
                   rng: np.random.Generator) -> np.ndarray:
    """Deterministic-under-`rng` random crop with zero-padding for undersized volumes."""
    c, D, H, W = img.shape
    pd, ph, pw = patch_size
    d0 = rng.integers(0, max(1, D - pd + 1))
    h0 = rng.integers(0, max(1, H - ph + 1))
    w0 = rng.integers(0, max(1, W - pw + 1))
    patch = img[:, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]
    pad = [(0, 0)] + [
        (0, max(0, pd - patch.shape[1])),
        (0, max(0, ph - patch.shape[2])),
        (0, max(0, pw - patch.shape[3])),
    ]
    patch = np.pad(patch, pad, mode="constant")
    return patch[:, :pd, :ph, :pw]


def _load_random_val_patches(cache_dir: str, n_patches: int,
                              patch_size: tuple[int, int, int]) -> torch.Tensor:
    """LEGACY: load N random patches by scanning a physical directory.

    Kept for backward compatibility with runs that don't use a split manifest.
    Not safe for the 70/10/20 experiments where the physical folder mixes
    train/val/test — use `_load_patches_from_paths` via manifest mode.
    """
    cache = Path(cache_dir)
    npzs = sorted(cache.glob("*.npz"))
    if not npzs:
        raise FileNotFoundError(f"No .npz files found under {cache}")
    rng = np.random.default_rng(0)
    picks = rng.choice(len(npzs), size=min(n_patches, len(npzs)), replace=False)
    xs = [_extract_patch(np.load(npzs[i])["image"].astype(np.float32),
                         patch_size, rng) for i in picks]
    return torch.from_numpy(np.stack(xs, axis=0))


def _load_patches_from_paths(paths: list[str], n_patches: int,
                              patch_size: tuple[int, int, int]
                              ) -> tuple[torch.Tensor, list[int]]:
    """Deterministic-seed-0 selection of `n_patches` files from an explicit list.

    Returns (tensor, indices_into_paths). Never touches the filesystem
    outside `paths`. Used by the manifest-restricted diagnostic mode.
    """
    if not paths:
        raise FileNotFoundError("Empty paths list passed to _load_patches_from_paths.")
    rng = np.random.default_rng(0)
    picks = rng.choice(len(paths), size=min(n_patches, len(paths)),
                       replace=False)
    picks = sorted(int(i) for i in picks)  # stable order for logging
    xs = [_extract_patch(np.load(paths[i])["image"].astype(np.float32),
                         patch_size, rng) for i in picks]
    return torch.from_numpy(np.stack(xs, axis=0)), picks


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
    if args.split_manifest:
        # ── MANIFEST-RESTRICTED MODE ─────────────────────────────────────
        # Loads only .npz files whose patient_id belongs to the requested
        # partition. Never scans the physical directory. This is the
        # required mode for the 70/10/20 experiments (val + test share
        # physical folders after re-partitioning).
        if not args.preprocessed_cache_dir:
            raise SystemExit(
                "diag: --split_manifest requires --preprocessed_cache_dir "
                "(cache root that the manifest's relative_npz_path is resolved against)."
            )
        # Deferred import — split_manifest lives under training/ which is
        # only importable when the repo root is on sys.path (done above).
        from training.split_manifest import (
            load_manifest, manifest_sha256, verify_manifest_partitions,
        )
        # Sanity: manifest CSV integrity + partition coverage.
        _ = verify_manifest_partitions(args.split_manifest)
        selected_cases = load_manifest(args.split_manifest,
                                       split=args.split_partition,
                                       cache_root=args.preprocessed_cache_dir,
                                       require_exists=True)
        if not selected_cases:
            raise SystemExit(
                f"diag: manifest partition {args.split_partition!r} is empty "
                f"under {args.split_manifest!r}."
            )
        # Defence-in-depth: assert every loaded case belongs to the
        # requested partition. load_manifest already filters by split,
        # but re-check via the full manifest so a stale csv can't slip
        # a wrong-partition case through.
        import csv as _csv
        _all_by_pid = {}
        with open(args.split_manifest, "r", encoding="utf-8", newline="") as _f:
            for row in _csv.DictReader(_f):
                _all_by_pid[row["patient_id"]] = row["split"]
        for c in selected_cases:
            assigned = _all_by_pid.get(c["patient_id"])
            if assigned != args.split_partition:
                raise SystemExit(
                    f"diag: patient {c['patient_id']!r} claimed by loader as "
                    f"partition={args.split_partition!r} but manifest says {assigned!r}. "
                    "Refusing — this would leak a wrong-partition case."
                )
        # Choose the diagnostic subset deterministically under seed 0.
        paths_all = [c["npz_path"] for c in selected_cases]
        x, picks = _load_patches_from_paths(
            paths_all, n_patches=args.n_patches,
            patch_size=tuple(args.patch_size),
        )
        x = x.to(device)
        chosen = [selected_cases[i] for i in picks]

        # Final leakage guard on the actually-loaded subset.
        chosen_pids = {c["patient_id"] for c in chosen}
        for pid in chosen_pids:
            if _all_by_pid.get(pid) != args.split_partition:
                raise SystemExit(
                    f"diag: post-selection leakage — {pid!r} is not in the "
                    f"{args.split_partition!r} partition."
                )

        _mani_sha = manifest_sha256(args.split_manifest)
        print(f"manifest      : {args.split_manifest}")
        print(f"manifest SHA  : {_mani_sha}")
        print(f"partition     : {args.split_partition}  (n_in_partition={len(selected_cases)})")
        print(f"selected      : {len(chosen)} case(s), seed=0")
        for c in chosen:
            print(f"  - patient_id={c['patient_id']:20s}  "
                  f"collection={c['collection']:6s}  npz={c['npz_path']}")
        print(f"patch tensor  : shape={tuple(x.shape)}")
    elif args.preprocessed_cache:
        # ── LEGACY MODE (backward-compat) ────────────────────────────────
        print("WARNING: legacy directory-scan mode. This is UNSAFE for the "
              "70/10/20 experiments — use --split_manifest instead.")
        x = _load_random_val_patches(
            args.preprocessed_cache, n_patches=args.n_patches,
            patch_size=tuple(args.patch_size),
        ).to(device)
        print(f"patches from {args.preprocessed_cache}: shape={tuple(x.shape)}")
    else:
        print("WARNING: neither --split_manifest nor --preprocessed_cache "
              "provided; falling back to random noise.")
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
                   help="LEGACY. Directory of .npz patches to draw real-MRI inputs from. "
                        "UNSAFE for 70/10/20 experiments — use --split_manifest instead.")
    p.add_argument("--n_patches", type=int, default=4)
    p.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 48])
    p.add_argument("--in_channels", type=int, default=2)
    p.add_argument("--out_channels", type=int, default=2)
    p.add_argument("--base_filters", type=int, default=32)

    # ── Manifest-restricted mode (preferred for split-manifest experiments) ──
    p.add_argument("--split_manifest", type=str, default=None,
                   help="Path to a CSV manifest produced by training/split_manifest.py. "
                        "When set, the diagnostic reads ONLY .npz files whose patient_id "
                        "belongs to --split_partition; the physical folder is never scanned.")
    p.add_argument("--split_partition", type=str, default="val",
                   choices=("train", "val", "test"),
                   help="Which manifest partition to draw diagnostic patches from. "
                        "Default 'val'.  For DS/nods 70/10/20 experiments always use 'val'; "
                        "'test' is reserved for the locked final-test evaluator.")
    p.add_argument("--preprocessed_cache_dir", type=str, default=None,
                   help="Cache root that --split_manifest's relative_npz_path values are "
                        "resolved against. Required when --split_manifest is set.")
    return p.parse_args()


if __name__ == "__main__":
    diag(parse_args())
