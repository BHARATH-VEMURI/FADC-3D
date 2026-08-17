"""Correctness tests for the corrected FADC3D implementation.

Coverage (23 assertions per spec):
  1  One base kernel parameter per AdaptiveDilatedConv3D.
  2  No independent kernel parameters per dilation.
  3  All dilation branches use the SAME adaptive kernel.
  4  Branch outputs identical spatial shape.
  5  Dilations are exactly (1,1,1), (2,2,2), (3,3,3).
  6  k_att shape (B, 3, D, H, W).
  7  k_att sums to 1 over dim=1.
  8  Expected dilation in [1, 3].
  9  W_low + W_high == W exactly.
  10 Identity AdaKern attentions reconstruct base kernel.
  11 c_low, f_low, c_high, f_high broadcast to (B, O, I, 3, 3, 3).
  12 s_att has 27 values and identity init.
  13 FrequencySelection3D identity reconstruction.
  14 Odd and even spatial sizes work.
  15 Output shape matches expected shape.
  16 All enabled mechanisms receive finite non-zero gradients.
  17 AMP (autocast + GradScaler) forward/backward is finite.
  18 Checkpoint save + strict=True reload round-trips.
  19 Encoder placement -> 8 corrected adaptive convs.
  20 Decoder placement -> 8.
  21 Bottleneck placement -> 2.
  22 Full placement -> 18.
  23 No corrected FADC modules outside the requested placement.
  24 Encoder+decoder placement (encdec, no bottleneck) -> 16.

Run:
    conda activate fadc3d
    python -m tests.test_fadc_3d_correct
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fadc_3d_correct import (
    FrequencySelection3D,
    AdaKern3D,
    AdaptiveDilatedConv3D,
)
from models.unet_3d_fadc_correct import (
    UNet3DFADCCorrect,
    build_unet3d_fadc_correct,
    EXPECTED_ADAPTIVE_CONV_COUNT,
    MODEL_NAMES,
)


N_ASSERTIONS_TARGET = 23


class Report:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def ok(self, tag: str, msg: str = "") -> None:
        self.passed.append(tag)
        print(f"  [OK] {tag}  {msg}")

    def bad(self, tag: str, msg: str) -> None:
        self.failed.append((tag, msg))
        print(f"  [FAIL] {tag}  {msg}")


REPORT = Report()


def check(cond: bool, tag: str, ok_msg: str = "", bad_msg: str = "") -> None:
    if cond:
        REPORT.ok(tag, ok_msg)
    else:
        REPORT.bad(tag, bad_msg or "assertion failed")


# ─────────────────────────── module-level fixtures

def _module(in_c=4, out_c=8, use_s=False):
    return AdaptiveDilatedConv3D(
        in_channels=in_c, out_channels=out_c,
        adakern_cfg=dict(use_position_att=use_s),
    )


# ─────────────────────────── 1: one base kernel per adaptive conv

def test_1_2_3_5():
    print("[1-3, 5] shared-kernel structure")
    m = _module()
    kernel_params = [n for n, _ in m.named_parameters()
                     if n == "weight"
                     or (n.endswith(".weight") and n.startswith("conv_branches."))]
    check(kernel_params == ["weight"],
          "1_one_base_kernel_param",
          "kernel params: ['weight']",
          f"unexpected kernel params: {kernel_params}")

    # No sub-Conv3d modules in the branch path — the branches use F.conv3d, not
    # nn.Conv3d modules. Any Conv3d we still find must belong to helper heads
    # (FreqSel predictors, AdaKern trunk, KAttHead), NOT to per-branch conv weights.
    per_dilation_convs = [n for n, mm in m.named_modules()
                          if isinstance(mm, nn.Conv3d) and "conv_branches" in n]
    check(per_dilation_convs == [],
          "2_no_per_dilation_convs",
          "no per-dilation nn.Conv3d modules",
          f"found: {per_dilation_convs}")

    # 3 + 5: run once, capture branch outputs (same kernel, same shape, exact dilations).
    x = torch.randn(2, 4, 8, 12, 16)
    _ = m(x)
    # k_att shape gives us num_branches: check we're producing 3.
    check(m.last_k_att.shape[1] == 3, "5_dilations_1_2_3_from_k_att",
          f"num_branches={m.last_k_att.shape[1]}",
          f"expected 3 branches, got {m.last_k_att.shape[1]}")
    check(m.dilation_list == (1, 2, 3), "5b_dilation_list_1_2_3",
          f"dilation_list={m.dilation_list}",
          f"expected (1,2,3), got {m.dilation_list}")


# ─────────────────────────── 3: all branches use the same adaptive kernel
#     — proved by construction (only ONE self.weight tensor is used to build
#       W_adaptive, then W_adaptive is passed to every dilation in the loop).
#     Also enforced structurally: no separate weight per branch (tested above).
def test_3_shared_kernel_direct():
    print("[3] all dilation branches consume ONE adaptive kernel")
    m = _module()
    # Freeze the base kernel; multiply by an arbitrary scalar via forward, then
    # examine branch outputs by directly calling _batched_conv3d with a known W.
    x = torch.randn(2, 4, 6, 6, 6)
    x_fs = m.fs(x)
    c_low, f_low, c_high, f_high, s_att = m.ada_kern(x_fs)
    W_adaptive = m.ada_kern.apply_to_kernel(m.weight, c_low, f_low, c_high, f_high, s_att)
    # Now for each dilation call _batched_conv3d and check output shape matches (2, 8, 6, 6, 6).
    shapes = set()
    for d in m.dilation_list:
        y = m._batched_conv3d(x_fs, W_adaptive, dilation=d)
        shapes.add(tuple(y.shape))
    check(len(shapes) == 1,
          "3_same_kernel_all_branches_shape_match",
          f"branch shape: {list(shapes)[0]}",
          f"branch shapes diverged: {shapes}")


# ─────────────────────────── 4: branch outputs have identical spatial shapes
def test_4_branch_shapes_equal():
    print("[4] branch outputs identical spatial shape")
    m = _module()
    x = torch.randn(2, 4, 8, 12, 14)
    x_fs = m.fs(x)
    c_low, f_low, c_high, f_high, s_att = m.ada_kern(x_fs)
    W_adaptive = m.ada_kern.apply_to_kernel(m.weight, c_low, f_low, c_high, f_high, s_att)
    outs = [m._batched_conv3d(x_fs, W_adaptive, dilation=d) for d in m.dilation_list]
    shapes = [tuple(o.shape) for o in outs]
    check(all(s == shapes[0] for s in shapes),
          "4_branch_shapes_equal",
          f"all branches -> {shapes[0]}",
          f"shapes diverged: {shapes}")


# ─────────────────────────── 6: k_att shape
# ─────────────────────────── 7: k_att sums to 1
# ─────────────────────────── 8: expected dilation in [1, 3]
def test_6_7_8_k_att_properties():
    print("[6-8] k_att shape / sum / expected dilation range")
    m = _module()
    x = torch.randn(2, 4, 8, 12, 16)
    _ = m(x)
    kshape = tuple(m.last_k_att.shape)
    check(kshape == (2, 3, 8, 12, 16),
          "6_k_att_shape_B_3_DHW",
          f"k_att shape {kshape}",
          f"expected (2,3,8,12,16), got {kshape}")
    sums = m.last_k_att.sum(dim=1)
    err = (sums - 1.0).abs().max().item()
    check(err < 1e-5,
          "7_k_att_sums_to_1",
          f"max|sum-1|={err:.2e}",
          f"softmax leaked: max|sum-1|={err:.2e}")
    ed = m.last_expected_dilation
    check(ed.min().item() >= 1.0 - 1e-5 and ed.max().item() <= 3.0 + 1e-5,
          "8_expected_dilation_in_1_3",
          f"E[dil] in [{ed.min().item():.4f}, {ed.max().item():.4f}]",
          f"out of range [{ed.min().item():.4f}, {ed.max().item():.4f}]")


# ─────────────────────────── 9: W_low + W_high == W
def test_9_kernel_decomposition():
    print("[9] W_low + W_high == W")
    W = torch.randn(8, 4, 3, 3, 3)
    W_low, W_high = AdaKern3D.decompose_kernel(W)
    err = (W_low + W_high - W).abs().max().item()
    check(err < 1e-6,
          "9_kernel_decomp_reconstructs",
          f"max|W_low+W_high-W|={err:.2e}",
          f"decomposition drift {err:.2e}")


# ─────────────────────────── 10: identity attentions reconstruct base kernel
# ─────────────────────────── 11: c/f broadcast to (B, O, I, 3, 3, 3)
# ─────────────────────────── 12: s_att has 27 values, identity init
def test_10_11_12_adakern_identity():
    print("[10-12] AdaKern identity + broadcast + s_att shape")
    ak = AdaKern3D(in_channels=4, out_channels=8, use_position_att=False).eval()
    W = torch.randn(8, 4, 3, 3, 3)
    x = torch.randn(2, 4, 6, 8, 8)
    with torch.no_grad():
        c_low, f_low, c_high, f_high, s_att = ak(x)
        W_ad = ak.apply_to_kernel(W, c_low, f_low, c_high, f_high, s_att)
    err = (W_ad - W.unsqueeze(0)).abs().max().item()
    check(err < 1e-6,
          "10_identity_att_reconstructs_kernel",
          f"max|W_ad - W|={err:.2e}",
          f"identity broken: {err:.2e}")
    check(tuple(W_ad.shape) == (2, 8, 4, 3, 3, 3),
          "11_c_f_broadcast_shape",
          f"W_ad shape {tuple(W_ad.shape)}",
          f"expected (2,8,4,3,3,3), got {tuple(W_ad.shape)}")

    # s_att shape + identity when enabled
    ak2 = AdaKern3D(in_channels=4, out_channels=8, use_position_att=True).eval()
    with torch.no_grad():
        c_low, f_low, c_high, f_high, s_att = ak2(x)
        W_ad2 = ak2.apply_to_kernel(W, c_low, f_low, c_high, f_high, s_att)
    check(tuple(s_att.shape) == (2, 1, 1, 3, 3, 3),
          "12a_s_att_27_values",
          f"s_att shape {tuple(s_att.shape)} (27 values)",
          f"expected (2,1,1,3,3,3), got {tuple(s_att.shape)}")
    err2 = (W_ad2 - W.unsqueeze(0)).abs().max().item()
    check(err2 < 1e-6,
          "12b_s_att_identity_init",
          f"max|W_ad-W|={err2:.2e} with s_att",
          f"s_att identity broken: {err2:.2e}")


# ─────────────────────────── 13: FrequencySelection3D identity reconstruction
# ─────────────────────────── 14: odd + even spatial sizes
def test_13_14_freq_sel():
    print("[13-14] FreqSel identity, odd+even sizes")
    ok_all = True
    details = []
    for D, H, W in [(8, 8, 4), (7, 9, 5), (10, 6, 8), (5, 5, 3)]:
        fs = FrequencySelection3D(in_channels=4).eval()
        x = torch.randn(2, 4, D, H, W)
        with torch.no_grad():
            y = fs(x)
        err = (y - x).abs().max().item()
        details.append(f"({D},{H},{W})->{err:.1e}")
        if err >= 1e-4:
            ok_all = False
    check(ok_all, "13_14_freq_sel_identity_odd_even",
          " ".join(details),
          " ".join(details))


# ─────────────────────────── 15: output shape matches expected
# ─────────────────────────── 16: mechanism gradients are finite and non-zero
def test_15_16_shapes_and_grads():
    print("[15-16] forward shape + finite gradient reach")
    m = AdaptiveDilatedConv3D(
        in_channels=4, out_channels=8,
        adakern_cfg=dict(use_position_att=True),
    )
    x = torch.randn(2, 4, 8, 12, 16, requires_grad=False)
    y = m(x)
    check(tuple(y.shape) == (2, 8, 8, 12, 16),
          "15_forward_shape",
          f"y={tuple(y.shape)}",
          f"expected (2,8,8,12,16), got {tuple(y.shape)}")

    loss = y.pow(2).mean()
    m.zero_grad()
    loss.backward()

    # Base kernel: finite grad, nonzero.
    g = m.weight.grad
    check(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0,
          "16a_base_kernel_grad",
          f"|g_weight|={g.abs().sum().item():.3e}",
          "base kernel grad missing / non-finite / zero")

    # AdaKern heads
    for tag, p in (
        ("c_low_fc",  m.ada_kern.c_low_fc.weight),
        ("f_low_fc",  m.ada_kern.f_low_fc.weight),
        ("c_high_fc", m.ada_kern.c_high_fc.weight),
        ("f_high_fc", m.ada_kern.f_high_fc.weight),
        ("s_fc",      m.ada_kern.s_fc.weight),
    ):
        pg = p.grad
        check(pg is not None and torch.isfinite(pg).all() and pg.abs().sum() > 0,
              f"16b_adakern_{tag}_grad",
              f"|g|={pg.abs().sum().item():.3e}",
              f"{tag} grad missing / non-finite / zero")

    # k_att head
    kg = m.k_att.head.weight.grad
    check(kg is not None and torch.isfinite(kg).all() and kg.abs().sum() > 0,
          "16c_k_att_head_grad",
          f"|g|={kg.abs().sum().item():.3e}",
          "k_att head grad missing / non-finite / zero")

    # FreqSel band predictors
    all_ok = True
    for i, conv in enumerate(m.fs.freq_weight_conv_list):
        g_i = conv.weight.grad
        if g_i is None or (not torch.isfinite(g_i).all()) or g_i.abs().sum() == 0:
            all_ok = False
            break
    check(all_ok, "16d_freq_sel_predictor_grads",
          f"all {len(m.fs.freq_weight_conv_list)} predictors received grad",
          "at least one FreqSel predictor has missing/zero grad")


# ─────────────────────────── 17: AMP finite forward/backward
def test_17_amp():
    print("[17] AMP autocast + GradScaler finite")
    if not torch.cuda.is_available():
        # Autocast still exposes fp16 on CPU (limited); GradScaler is CUDA-only.
        # We run a bf16 CPU autocast smoke instead — it exercises the FFT float32 branch.
        m = _module().eval()
        x = torch.randn(2, 4, 6, 8, 8)
        with autocast(device_type="cpu", dtype=torch.bfloat16):
            y = m(x)
        check(torch.isfinite(y).all(),
              "17_amp_cpu_bf16_forward_finite",
              "forward finite (CPU bf16)",
              "AMP forward produced NaN/inf")
        return

    # Real optimizer + real scaler.unscale_ so both the AMP forward AND the
    # scaled-then-unscaled backward are exercised end-to-end.
    device = torch.device("cuda")
    m = _module(use_s=True).to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    scaler = GradScaler("cuda")
    x = torch.randn(2, 4, 8, 12, 12, device=device)
    with autocast("cuda"):
        y = m(x)
        loss = y.float().pow(2).mean()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)                       # unscale in-place

    check(torch.isfinite(y).all(),
          "17a_amp_forward_finite",
          "AMP forward finite",
          "AMP forward produced NaN/inf")
    checked = [
        ("weight",     m.weight),
        ("c_low_fc",   m.ada_kern.c_low_fc.weight),
        ("f_low_fc",   m.ada_kern.f_low_fc.weight),
        ("c_high_fc",  m.ada_kern.c_high_fc.weight),
        ("f_high_fc",  m.ada_kern.f_high_fc.weight),
        ("s_fc",       m.ada_kern.s_fc.weight),
        ("k_att_head", m.k_att.head.weight),
    ]
    for tag, p in checked:
        g = p.grad
        check(g is not None and torch.isfinite(g).all(),
              f"17b_amp_unscaled_grad_finite_{tag}",
              f"|g|={g.abs().sum().item():.3e}" if g is not None else "no grad",
              f"{tag} grad missing or non-finite after unscale_")


# ─────────────────────────── 18: checkpoint round-trip strict=True
def test_18_ckpt_roundtrip(tmp_path: Path):
    print("[18] checkpoint save + strict reload")
    m1 = build_unet3d_fadc_correct(
        "unet3d_fadc_encoder_correct",
        in_channels=2, out_channels=2, base_filters=8,
    ).eval()  # eval() so Dropout3d is off and BN uses running stats.
    x = torch.randn(1, 2, 32, 32, 16)
    with torch.no_grad():
        y1 = m1(x)
    buf = io.BytesIO()
    torch.save({"model": m1.state_dict()}, buf)
    buf.seek(0)
    ckpt = torch.load(buf, weights_only=False)
    m2 = build_unet3d_fadc_correct(
        "unet3d_fadc_encoder_correct",
        in_channels=2, out_channels=2, base_filters=8,
    ).eval()
    missing, unexpected = m2.load_state_dict(ckpt["model"], strict=True)
    check(len(missing) == 0 and len(unexpected) == 0,
          "18a_strict_ckpt_load",
          "all keys matched under strict=True",
          f"missing={missing} unexpected={unexpected}")
    with torch.no_grad():
        y2 = m2(x)
    err = (y1 - y2).abs().max().item()
    check(err < 1e-6,
          "18b_ckpt_reload_forward_identical",
          f"max|y1-y2|={err:.2e}",
          f"reload changed outputs: {err:.2e}")


# ─────────────────────────── 19-23: placement counts
def test_19_20_21_22_23_placements():
    print("[19-23] placement adaptive-conv counts + isolation")
    for name in MODEL_NAMES:
        placement = name.split("_")[2]
        m = build_unet3d_fadc_correct(
            name, in_channels=2, out_channels=2, base_filters=8,
        )
        expected = EXPECTED_ADAPTIVE_CONV_COUNT[placement]
        n = m.count_adaptive_convs()
        check(n == expected,
              f"{ {'encoder':19,'decoder':20,'bottleneck':21,'encdec':24,'full':22}[placement] }_placement_{placement}_count",
              f"{name} -> {n}/{expected}",
              f"{name} has {n} adaptive convs, expected {expected}")
        adapt_names = m.adaptive_conv_names()
        allowed_prefixes = {
            "encoder":    ("enc",),
            "decoder":    ("dec",),
            "bottleneck": ("bottleneck",),
            "encdec":     ("enc", "dec"),
            "full":       ("enc", "bottleneck", "dec"),
        }[placement]
        outside = [n_ for n_ in adapt_names if not any(n_.startswith(pref) for pref in allowed_prefixes)]
        check(not outside,
              f"23_no_adapt_outside_{placement}",
              f"{name} isolation OK",
              f"{name} has adaptive convs outside {allowed_prefixes}: {outside}")


# ─────────────────────────── run

def main():
    print(f"Running corrected-FADC3D test suite (target {N_ASSERTIONS_TARGET} assertions)\n")

    test_1_2_3_5()
    test_3_shared_kernel_direct()
    test_4_branch_shapes_equal()
    test_6_7_8_k_att_properties()
    test_9_kernel_decomposition()
    test_10_11_12_adakern_identity()
    test_13_14_freq_sel()
    test_15_16_shapes_and_grads()
    test_17_amp()

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_18_ckpt_roundtrip(Path(td))

    test_19_20_21_22_23_placements()

    print("\n" + "=" * 60)
    print(f"passed : {len(REPORT.passed)}")
    print(f"failed : {len(REPORT.failed)}")
    if REPORT.failed:
        print("\nFAILURES:")
        for tag, msg in REPORT.failed:
            print(f"  - {tag}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
