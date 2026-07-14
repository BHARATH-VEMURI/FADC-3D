"""Attention-collapse diagnostic — but on REAL MAMA-MIA val patches,
not random noise. Companion to diag_v2_attention.py.

Loads N val cases from the local raw NIfTI dataset, runs the same
val transforms the model was trained under (spacing, orientation,
intensity, foreground crop, 2ch concat), tumor-centered crops each
one to a small patch, feeds through the model with attention hooks,
reports the same three metrics as the random-input version.

If c_att / f_att std stays sub-0.005 here TOO, we can rule out
"random-noise inputs weren't structured enough" as an explanation
for the collapse we saw in diag_v2_attention.py.

Runtime: ~1-3 min on CPU (NIfTI load + Spacingd is the slow part).

Run:
    conda activate fadc3d
    python diag_v2_attention_real.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))
from models.unet_3d_fadc_v2 import UNet3DFADC_V2
from fadc_3d_v2.omni_attention_3d_spatial import OmniAttention3DSpatial
from data.mama_mia_dataset import discover_cases, get_val_transforms, DATA_ROOT


CKPT = r"C:\Users\bhara\Downloads\check\fadcencoder_V2\seed42_epoch_25\best_model.pth"
N_CASES = 4
PATCH_SIZE = (64, 64, 32)   # smaller than training's 96x96x48 for CPU speed


def find_fadc_layers(model):
    for name, mod in model.named_modules():
        if isinstance(mod, OmniAttention3DSpatial):
            yield name, mod


def register_hooks(model, capture):
    handles = []
    for name, mod in find_fadc_layers(model):
        capture[name] = {"c": [], "f": [], "s": [], "k": []}

        def make_hook(nm):
            def hook(module, inp, out):
                c_att, f_att, s_att, k_att = out
                capture[nm]["c"].append(c_att.detach().cpu())
                capture[nm]["f"].append(f_att.detach().cpu())
                if torch.is_tensor(s_att):
                    capture[nm]["s"].append(s_att.detach().cpu())
                if torch.is_tensor(k_att):
                    capture[nm]["k"].append(k_att.detach().cpu())
            return hook

        handles.append(mod.register_forward_hook(make_hook(name)))
    return handles


def tumor_centered_crop(image, label, patch_size):
    """Crop `image` around the centroid of the tumor mask. Fallback to volume
    center if no tumor voxels. image: (2, D, H, W); label: (1, D, H, W)."""
    td, th, tw = patch_size
    _, D, H, W = image.shape

    mask = label[0] > 0
    if mask.any():
        idx = mask.nonzero(as_tuple=False)
        cz, cy, cx = idx.float().mean(dim=0).tolist()
    else:
        cz, cy, cx = D / 2, H / 2, W / 2

    z0 = int(np.clip(cz - td // 2, 0, D - td))
    y0 = int(np.clip(cy - th // 2, 0, H - th))
    x0 = int(np.clip(cx - tw // 2, 0, W - tw))
    return image[:, z0:z0 + td, y0:y0 + th, x0:x0 + tw]


def load_real_patches(n_cases, patch_size):
    split_csv = os.path.join(DATA_ROOT, "train_test_splits.csv")
    if not os.path.exists(split_csv):
        print(f"[warn] no split csv at {split_csv} -- using all cases as val")
        split_csv = None
    cases = discover_cases(DATA_ROOT, split_csv, split="test")
    if not cases:
        print(f"[warn] no val cases found under {DATA_ROOT}; falling back to train split")
        cases = discover_cases(DATA_ROOT, split_csv, split="train")
    print(f"discovered {len(cases)} val cases; using first {n_cases}")

    xform = get_val_transforms()
    patches = []
    for i, case in enumerate(cases[:n_cases]):
        t0 = time.time()
        data = xform(case)
        img = torch.as_tensor(data["image"])   # (2, D, H, W)
        lbl = torch.as_tensor(data["label"])   # (1, D, H, W)
        patch = tumor_centered_crop(img, lbl, patch_size).unsqueeze(0)  # (1, 2, D, H, W)
        elapsed = time.time() - t0
        tumor_vox = int((lbl > 0).sum().item())
        print(f"  [{i+1}/{n_cases}] {case['patient_id']:>18s} "
              f"({case['collection']:<6s}) "
              f"vol {tuple(img.shape[1:])} -> patch {tuple(patch.shape[2:])}  "
              f"tumor_vox={tumor_vox:>7d}  {elapsed:.1f}s")
        patches.append(patch.float())
    return patches


def summarize(capture, temperature_estimate, n_inputs):
    print(f"\nk_att temperature at this ckpt (cosine ep25/100): T ~ {temperature_estimate:.2f}")
    print(f"n real val patches: {n_inputs}")
    header = (f"{'Layer':<28} "
              f"{'c mean':>8} {'c inp-std':>10} "
              f"{'f mean':>8} {'f inp-std':>10} "
              f"{'s inp-std':>10} "
              f"{'k spat-std':>11} {'k inp-std':>10}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name in sorted(capture.keys()):
        c_stack = torch.stack(capture[name]["c"], dim=0)   # (N, 1, C, 1, 1, 1)
        f_stack = torch.stack(capture[name]["f"], dim=0)

        c_mean = c_stack.mean().item()
        f_mean = f_stack.mean().item()
        # Real-data path feeds one patch per forward, so batch dim is 1.
        # "inp-std" = std over N of the per-image mean.
        c_inp = c_stack.flatten(2).mean(dim=2).std().item()
        f_inp = f_stack.flatten(2).mean(dim=2).std().item()

        if capture[name]["s"]:
            s_stack = torch.stack(capture[name]["s"], dim=0)
            s_inp = s_stack.flatten(2).mean(dim=2).std().item()
            s_inp_s = f"{s_inp:.4f}"
        else:
            s_inp_s = "SKIP"

        if capture[name]["k"]:
            k_stack = torch.stack(capture[name]["k"], dim=0)   # (N, 1, n_br, D, H, W)
            k_flat = k_stack.float().flatten(3)                # (N, 1, n_br, D*H*W)
            spatial_std = k_flat.std(dim=3, unbiased=False).mean().item()
            # per-input branch-0 mean, std across N
            per_input_std = k_stack[:, 0, 0].flatten(1).mean(dim=1).std().item()
            k_spat_s = f"{spatial_std:.4f}"
            k_inp_s = f"{per_input_std:.4f}"
        else:
            k_spat_s = "SKIP"
            k_inp_s = "SKIP"

        print(f"{name:<28} "
              f"{c_mean:>8.4f} {c_inp:>10.4f} "
              f"{f_mean:>8.4f} {f_inp:>10.4f} "
              f"{s_inp_s:>10} "
              f"{k_spat_s:>11} {k_inp_s:>10}")
    print("=" * len(header))


def estimate_temperature(epoch, anneal_epochs=100, t_start=4.0, t_end=0.8):
    import math
    if anneal_epochs <= 1:
        return t_end
    e = min(max(epoch, 0), anneal_epochs - 1)
    cos_term = 0.5 * (1.0 + math.cos(math.pi * e / (anneal_epochs - 1)))
    return t_end + (t_start - t_end) * cos_term


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"ckpt  : {CKPT}\n")

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    epoch = ckpt.get("epoch", 0)
    print(f"epoch : {epoch} (0-indexed)")
    print(f"best_dice: {ckpt.get('best_dice'):.4f}\n")

    model = UNet3DFADC_V2(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_filters=cfg["model"]["base_filters"],
        fadc_placement="encoder",
        deep_supervision=cfg["model"].get("deep_supervision", False),
    ).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)

    T_est = estimate_temperature(epoch)
    if hasattr(model, "set_temperature"):
        model.set_temperature(T_est)

    print("loading real val patches ...")
    patches = load_real_patches(N_CASES, PATCH_SIZE)
    if not patches:
        print("[error] no patches loaded -- check DATA_ROOT")
        return

    capture = {}
    handles = register_hooks(model, capture)
    print(f"\nforward pass on {len(patches)} real patches (shape {tuple(patches[0].shape)}) ...")
    with torch.no_grad():
        for i, x in enumerate(patches):
            t0 = time.time()
            _ = model(x.to(device))
            print(f"  patch {i+1}: {time.time() - t0:.1f}s")
    for h in handles:
        h.remove()

    summarize(capture, T_est, len(patches))

    print("\nREADING NOTE")
    print("------------")
    print("Compare to diag_v2_attention.py (random noise). If c_att/f_att std")
    print("stays sub-0.005 here, real MRI structure doesn't save them --")
    print("collapse is training-driven, not input-driven.")
    print("If they jump to > 0.01, random noise was insufficient signal for")
    print("the diagnostic and v1's real-data diagnostic would need re-examining too.")


if __name__ == "__main__":
    main()
