"""Attention-collapse diagnostic on a v2 checkpoint.

Same idea as the v1 diagnostic that found the identity collapse:
for each FADC layer, hook the OmniAttention3DSpatial forward, feed
n=8 different inputs, and measure whether the attention values
actually change per input (adaptation) or stay pinned (collapse).

Reads three signals per FADC layer:

  c_att -- channel attention. shape (B, C_in, 1, 1, 1).
      v1 was stuck at ~0.5 (bias=0 -> sigmoid(0)=0.5 -> identity after *2).
      v2 warm init: bias=0.5 -> sigmoid(0.5)~0.622 -> non-identity start.
      COLLAPSE test: std across inputs < 0.005 -> not adapting per input.

  f_att -- filter attention. shape (B, C_out, 1, 1, 1). Same test.

  k_att -- kernel/dilation attention.
      v1 shape (B, num_branches, 1, 1, 1) -- one weight per image.
      v2 shape (B, num_branches, D, H, W) -- one softmax per voxel.
      v1 collapsed to one-hot (std ~0.0005).
      v2's promise is spatial variation WITHIN an input.
      SPATIAL test: std of k_att branch-0 across voxels within one input.
      PER-INPUT test: how much does the spatial-mean of k_att branch-0
      change between different inputs.

NOTE ON TEMPERATURE
    At ep25 out of 100, cosine anneal gives T ~ 3.58 (still very soft).
    Soft softmax pulls k_att values toward 0.5 regardless of adaptation.
    So a SMALL k_att spatial std at ep25 doesn't necessarily mean collapse
    -- it can also mean "still-soft softmax." The definitive k_att read
    is at ep100 when T=1.0. But c_att/f_att are NOT temperature-gated
    -- they can be read directly for adaptation evidence today.

Run:
    conda activate fadc3d
    python diag_v2_attention.py
"""
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))
from models.unet_3d_fadc_v2 import UNet3DFADC_V2
from fadc_3d_v2.omni_attention_3d_spatial import OmniAttention3DSpatial


CKPT = r"C:\Users\bhara\Downloads\check\fadcenc0der_attentions_v2_2\40\best_model.pth"
N_INPUTS = 8
INPUT_SHAPE = (1, 2, 32, 32, 16)   # small — CPU-friendly


def find_fadc_layers(model):
    """Yield (dotted_name, OmniAttention3DSpatial_module) for every FADC layer."""
    for name, mod in model.named_modules():
        if isinstance(mod, OmniAttention3DSpatial):
            yield name, mod


def register_hooks(model, capture):
    """Hook every OmniAttention3DSpatial.forward — writes into capture[name]."""
    handles = []
    for name, mod in find_fadc_layers(model):
        capture[name] = {"c": [], "f": [], "s": [], "k": []}

        def make_hook(nm):
            def hook(module, inp, out):
                c_att, f_att, s_att, k_att = out
                capture[nm]["c"].append(c_att.detach().cpu())
                capture[nm]["f"].append(f_att.detach().cpu())
                # s_att and k_att may be a scalar 1.0 (skip path) on older configs.
                if torch.is_tensor(s_att):
                    capture[nm]["s"].append(s_att.detach().cpu())
                if torch.is_tensor(k_att):
                    capture[nm]["k"].append(k_att.detach().cpu())
            return hook

        handles.append(mod.register_forward_hook(make_hook(name)))
    return handles


def _batched_and_input_std(stack):
    """stack shape (N, B, ...) where each hook call captured a batch of size B.
    Returns (input_std, batched_std): input_std is std across N (per-image batch
    means), batched_std is mean std across the batch dim inside each hook call."""
    # Per-hook mean over all non-batch dims -> (N, B)
    per_batch_mean = stack.flatten(2).mean(dim=2)
    input_std = per_batch_mean.std().item()

    # Batch std inside each hook call, averaged over hooks. Requires B>=2.
    if stack.shape[1] < 2:
        batched_std = float("nan")
    else:
        # std across B within each hook, mean over the remaining dims, then avg over N.
        std_within = stack.float().std(dim=1, unbiased=False)   # (N, ...)
        batched_std = std_within.flatten(1).mean(dim=1).mean().item()
    return input_std, batched_std


def summarize(capture, temperature_estimate):
    """Report per-layer adaptation metrics for c/f/s/k."""
    print(f"\nEstimated k_att temperature at this checkpoint's epoch: ~{temperature_estimate:.2f}")
    print("(T=1.0 is 'normal sharp softmax'; T=4.0 is very soft; higher T pulls k_att toward uniform)")
    print()
    header = (f"{'Layer':<28} "
              f"{'c mean':>8} {'c inp-std':>10} {'c bat-std':>10} "
              f"{'f mean':>8} {'f inp-std':>10} {'f bat-std':>10} "
              f"{'s inp-std':>10} {'s bat-std':>10} "
              f"{'k spat-std':>11} {'k inp-std':>10}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    layer_names = sorted(capture.keys())
    for name in layer_names:
        c_stack = torch.stack(capture[name]["c"], dim=0)
        f_stack = torch.stack(capture[name]["f"], dim=0)

        c_mean = c_stack.mean().item()
        f_mean = f_stack.mean().item()
        c_inp, c_bat = _batched_and_input_std(c_stack)
        f_inp, f_bat = _batched_and_input_std(f_stack)

        if capture[name]["s"]:
            s_stack = torch.stack(capture[name]["s"], dim=0)
            s_inp, s_bat = _batched_and_input_std(s_stack)
            s_inp_s = f"{s_inp:.4f}"
            s_bat_s = f"{s_bat:.4f}" if s_bat == s_bat else "  nan "
        else:
            s_inp_s = "SKIP"
            s_bat_s = "SKIP"

        if capture[name]["k"]:
            k_stack = torch.stack(capture[name]["k"], dim=0)   # (N, B, n_br, D, H, W)
            # Spatial std WITHIN one (image, branch), averaged over N, B, branches.
            k_flat = k_stack.float().flatten(3)                # (N, B, n_br, D*H*W)
            spatial_std = k_flat.std(dim=3, unbiased=False).mean().item()
            # PER-INPUT std of the spatial-mean of branch-0, across the N*B samples.
            per_input_mean = k_stack[:, :, 0].flatten(2).mean(dim=2).flatten()
            per_input_std = per_input_mean.std().item()
            k_spat_s = f"{spatial_std:.4f}"
            k_inp_s = f"{per_input_std:.4f}"
        else:
            k_spat_s = "SKIP"
            k_inp_s = "SKIP"

        print(f"{name:<28} "
              f"{c_mean:>8.4f} {c_inp:>10.4f} {c_bat:>10.4f} "
              f"{f_mean:>8.4f} {f_inp:>10.4f} {f_bat:>10.4f} "
              f"{s_inp_s:>10} {s_bat_s:>10} "
              f"{k_spat_s:>11} {k_inp_s:>10}")

    print("=" * len(header))
    print()
    print("INTERPRETATION GUIDE")
    print("--------------------")
    print("USED IN FORWARD:  a column shows a number (not 'SKIP') -> attention path is live.")
    print("INPUT-ADAPTIVE :  'inp-std' or 'k inp-std' > ~0.010 -> attention varies across inputs.")
    print("BATCH-ADAPTIVE :  'bat-std' > ~0.010 -> attention varies across samples within a mini-batch.")
    print()
    print("c/f: mean ~ 0.622 + std < 0.005  -> warm init kept, attention NOT adapting.")
    print("c/f: std > 0.010                 -> attention IS varying (working).")
    print()
    print("s : inp-std > 0.005 -> spatial kernel-position attention adapts to input (v2.2 goal).")
    print()
    print("k : spat-std > 0.010 -> per-voxel spatial variation (v2 promise).")
    print("k : inp-std  > 0.005 -> per-input branch preference (v2.2 aux target).")
    print("    at HIGH T (>= 3.0) softmax is still soft, so small values are expected until T -> 1.")


def estimate_temperature(epoch, anneal_epochs=60, t_start=4.0, t_end=0.8):
    """Defaults match v2.2 launch args (t_end=0.8, anneal 60ep).
    For v2.1 checkpoints, override with anneal_epochs=100, t_end=0.8."""
    import math
    if anneal_epochs <= 1:
        return t_end
    e = min(max(epoch, 0), anneal_epochs - 1)
    cos_term = 0.5 * (1.0 + math.cos(math.pi * e / (anneal_epochs - 1)))
    return t_end + (t_start - t_end) * cos_term


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"ckpt  : {CKPT}")

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    epoch = ckpt.get("epoch", 0)   # 0-indexed
    print(f"epoch : {epoch} (0-indexed; ep25 in 1-indexed terms)")
    print(f"best_dice: {ckpt.get('best_dice'):.4f}")

    model = UNet3DFADC_V2(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_filters=cfg["model"]["base_filters"],
        fadc_placement="encoder",
        deep_supervision=cfg["model"].get("deep_supervision", False),
    ).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)

    # Set the temperature the training loop would have set for this epoch
    T_est = estimate_temperature(epoch)
    if hasattr(model, "set_temperature"):
        model.set_temperature(T_est)

    # Register hooks
    capture = {}
    handles = register_hooks(model, capture)

    # Feed N different random inputs (seeded for reproducibility)
    torch.manual_seed(0)
    print(f"\nfeeding {N_INPUTS} inputs of shape {INPUT_SHAPE}...")
    with torch.no_grad():
        for i in range(N_INPUTS):
            x = torch.randn(*INPUT_SHAPE, device=device)
            _ = model(x)

    for h in handles:
        h.remove()

    summarize(capture, T_est)


if __name__ == "__main__":
    main()
