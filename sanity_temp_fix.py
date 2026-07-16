"""Sanity check for the omni_attention_3d_spatial temperature-scope fix.

The bug: /self.temperature was applied inside the sigmoids for c_att / f_att /
s_att. At T=4.0 with warm bias 0.5, sigmoid(0.5 / 4.0) = 0.531 -- warm init
neutralized. Temperature belongs only in the k_att softmax.

This script instantiates a fresh v2 block (no training), then:
  1. sets T=4.0 (start-of-training v2 condition)
  2. runs one forward
  3. asserts c_att / f_att mean ~ 0.62 (sigmoid(0.5), warm init preserved)
  4. changes T to 1.0
  5. re-runs forward on same input
  6. asserts k_att distribution DID change (temperature still affects softmax)
  7. asserts c_att / f_att values did NOT change (sigmoid no longer T-gated)

Run:
    conda activate fadc3d
    python sanity_temp_fix.py
"""
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))
from fadc_3d_v2.adaptive_dilated_conv_3d_v2 import AdaptiveDilatedConv3DV2
from fadc_3d_v2.omni_attention_3d_spatial import OmniAttention3DSpatial


def capture_attention(block, x, T):
    """Set temperature, run forward, capture attention tuple."""
    block.omni_att.set_temperature(T)
    captured = {}
    def hook(_mod, _inp, out):
        c_att, f_att, s_att, k_att = out
        captured["c"] = c_att.detach().clone()
        captured["f"] = f_att.detach().clone()
        captured["s"] = s_att
        captured["k"] = k_att.detach().clone()
    h = block.omni_att.register_forward_hook(hook)
    with torch.no_grad():
        _ = block(x)
    h.remove()
    return captured


def main():
    torch.manual_seed(0)

    # Fresh v2 block, no training. Warm bias defaults to 0.5.
    block = AdaptiveDilatedConv3DV2(
        in_channels=8, out_channels=16,
        kernel_size=3,
        bias_init=0.5,     # explicit for clarity
    ).eval()

    # Confirm biases match warm-init as designed
    b_c = block.omni_att.channel_fc.bias.data
    b_f = block.omni_att.filter_fc.bias.data
    print(f"channel_fc.bias : mean={b_c.mean():.4f} std={b_c.std():.4f}  "
          f"(warm init 0.5 -> all should be 0.5)")
    print(f"filter_fc.bias  : mean={b_f.mean():.4f} std={b_f.std():.4f}")
    assert abs(b_c.mean().item() - 0.5) < 1e-6, "warm init not applied to channel_fc"
    assert abs(b_f.mean().item() - 0.5) < 1e-6, "warm init not applied to filter_fc"
    print()

    # One deterministic input
    x = torch.randn(1, 8, 8, 16, 16)

    # ---- Pass 1: T=4.0 (start-of-training v2 condition) ----
    T1 = 4.0
    a1 = capture_attention(block, x, T1)
    c_mean_T4 = a1["c"].mean().item()
    f_mean_T4 = a1["f"].mean().item()
    k_branch0_mean_T4 = a1["k"][:, 0].mean().item()
    k_branch0_std_T4  = a1["k"][:, 0].std().item()
    print(f"T = {T1}")
    print(f"  c_att mean = {c_mean_T4:.4f}  (expect ~0.622 = sigmoid(0.5) if fix is correct)")
    print(f"  f_att mean = {f_mean_T4:.4f}  (expect ~0.622)")
    print(f"  k_att branch-0: mean={k_branch0_mean_T4:.4f} std={k_branch0_std_T4:.4f}")
    print()

    # ---- Pass 2: T=1.0 (end-of-training v2 condition) ----
    T2 = 1.0
    a2 = capture_attention(block, x, T2)
    c_mean_T1 = a2["c"].mean().item()
    f_mean_T1 = a2["f"].mean().item()
    k_branch0_mean_T1 = a2["k"][:, 0].mean().item()
    k_branch0_std_T1  = a2["k"][:, 0].std().item()
    print(f"T = {T2}")
    print(f"  c_att mean = {c_mean_T1:.4f}")
    print(f"  f_att mean = {f_mean_T1:.4f}")
    print(f"  k_att branch-0: mean={k_branch0_mean_T1:.4f} std={k_branch0_std_T1:.4f}")
    print()

    # ---- Assertions ----
    print("checks:")

    # 1. c_att / f_att mean at T=4 should be ~0.62 (not ~0.53 -- old bug value)
    ok_c = abs(c_mean_T4 - 0.622) < 0.05
    ok_f = abs(f_mean_T4 - 0.622) < 0.05
    print(f"  [{'PASS' if ok_c else 'FAIL'}] c_att mean at T=4.0 close to 0.622 "
          f"(warm init preserved) -- got {c_mean_T4:.4f}")
    print(f"  [{'PASS' if ok_f else 'FAIL'}] f_att mean at T=4.0 close to 0.622 -- "
          f"got {f_mean_T4:.4f}")

    # 2. c_att / f_att should NOT change when temperature changes (sigmoid is T-free now)
    c_delta = abs(c_mean_T4 - c_mean_T1)
    f_delta = abs(f_mean_T4 - f_mean_T1)
    ok_c_flat = c_delta < 1e-5
    ok_f_flat = f_delta < 1e-5
    print(f"  [{'PASS' if ok_c_flat else 'FAIL'}] c_att invariant to T (|delta|={c_delta:.2e})")
    print(f"  [{'PASS' if ok_f_flat else 'FAIL'}] f_att invariant to T (|delta|={f_delta:.2e})")

    # 3. k_att SHOULD change with temperature (softmax is T-gated)
    k_std_ratio = k_branch0_std_T1 / max(k_branch0_std_T4, 1e-6)
    ok_k = k_std_ratio > 1.5   # T=1.0 sharpens -> higher std
    print(f"  [{'PASS' if ok_k else 'FAIL'}] k_att sharpens as T drops "
          f"(std ratio T1/T4 = {k_std_ratio:.2f}, expect > 1.5)")

    all_ok = all([ok_c, ok_f, ok_c_flat, ok_f_flat, ok_k])
    print()
    print("=== SANITY", "PASSED" if all_ok else "FAILED", "===")

    # Sanity math for the paper/report
    import math
    print()
    print("Reference math:")
    print(f"  BEFORE fix, T=4.0: sigmoid(0.5/4.0) = {1/(1+math.exp(-0.5/4.0)):.4f}  "
          f"(this was the observed 0.53 on the ep25 ckpt)")
    print(f"  AFTER  fix, T=any: sigmoid(0.5)      = {1/(1+math.exp(-0.5)):.4f}  "
          f"(warm init preserved, temperature-independent)")


if __name__ == "__main__":
    main()
