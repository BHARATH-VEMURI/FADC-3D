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


CKPT = r"C:\Users\bhara\Downloads\check\fadcencoderV2fixed\70\best_model.pth"
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
        capture[name] = {"c": [], "f": [], "k": []}

        def make_hook(nm):
            def hook(module, inp, out):
                c_att, f_att, _s_att, k_att = out
                capture[nm]["c"].append(c_att.detach().cpu())
                capture[nm]["f"].append(f_att.detach().cpu())
                capture[nm]["k"].append(k_att.detach().cpu())
            return hook

        handles.append(mod.register_forward_hook(make_hook(name)))
    return handles


def summarize(capture, temperature_estimate):
    """Report per-layer adaptation metrics."""
    print(f"\nEstimated k_att temperature at this checkpoint's epoch: ~{temperature_estimate:.2f}")
    print("(T=1.0 is 'normal sharp softmax'; T=4.0 is very soft; higher T pulls k_att toward uniform)")
    print()
    print("=" * 108)
    print(f"{'Layer':<28} {'c_att mean':>11} {'c_att std':>11} "
          f"{'f_att mean':>11} {'f_att std':>11} "
          f"{'k spatial-std':>15} {'k per-input-std':>17}")
    print("-" * 108)

    layer_names = sorted(capture.keys())
    for name in layer_names:
        c_stack = torch.stack(capture[name]["c"], dim=0)    # (N, 1, C, 1, 1, 1)
        f_stack = torch.stack(capture[name]["f"], dim=0)    # (N, 1, C, 1, 1, 1)
        k_stack = torch.stack(capture[name]["k"], dim=0)    # (N, 1, n_br, D, H, W)

        c_mean = c_stack.mean().item()
        c_std_across_inputs = c_stack.mean(dim=(2, 3, 4, 5)).std().item()

        f_mean = f_stack.mean().item()
        f_std_across_inputs = f_stack.mean(dim=(2, 3, 4, 5)).std().item()

        # k_att branch-0 spatial map per input: shape (N, D, H, W)
        k0 = k_stack[:, 0, 0]

        # (a) SPATIAL std within one input, averaged across inputs
        spatial_std = k0.flatten(1).std(dim=1).mean().item()

        # (b) PER-INPUT variation: spatial-mean per input, then std across inputs
        per_input_mean = k0.mean(dim=(1, 2, 3))
        per_input_std = per_input_mean.std().item()

        print(f"{name:<28} {c_mean:>11.4f} {c_std_across_inputs:>11.4f} "
              f"{f_mean:>11.4f} {f_std_across_inputs:>11.4f} "
              f"{spatial_std:>15.4f} {per_input_std:>17.4f}")

    print("=" * 108)
    print()
    print("INTERPRETATION GUIDE")
    print("--------------------")
    print("c_att / f_att (per-image, temperature-independent):")
    print("  mean ~ 0.500 + std < 0.005  -> V1 identity collapse (bias=0, sigmoid(0)=0.5)")
    print("  mean ~ 0.622 + std < 0.005  -> V2 warm init took, but attention NOT adapting per input")
    print("  mean anywhere + std > 0.010 -> attention IS varying across inputs (working)")
    print()
    print("k_att (per-voxel spatial, temperature-modulated):")
    print("  spatial-std < 0.005 -> uniform value across space (v1-style collapse)")
    print("  spatial-std > 0.010 -> spatial variation within one input (v2 promise fulfilled)")
    print("  BUT AT HIGH T (>= 3.0), even a working v2 gives small spatial-std")
    print("  because the softmax is still soft. Re-run at ep100 for definitive read.")
    print()
    print("per-input-std (how much k_att's average changes between inputs):")
    print("  > 0.005 -> model is learning per-input dilation preference (any T)")


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
