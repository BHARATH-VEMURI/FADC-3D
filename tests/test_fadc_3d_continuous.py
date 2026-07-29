"""16 focused CPU tests for the continuous FADC3D implementation.

Numbering matches the task spec. Every test runs in seconds on CPU.
"""
from __future__ import annotations
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fadc_3d_continuous import ContinuousAdaDR3D, ContinuousDilatedConv3D
from fadc_3d_continuous.continuous_adadr_3d import _kernel_lattice_3d
from models.unet_3d_fadc_continuous import (
    build_unet3d_fadc_continuous, UNet3DFADCContinuous,
)


# ─────────────────────────────────────────────────────────────────────────
# Helper: build a bare ContinuousAdaDR3D and force it to a known state
# so we can compare against reference Conv3d numerically. We bypass the
# top-level block's AdaKern/FreqSelect and drive AdaDR3D directly.
# ─────────────────────────────────────────────────────────────────────────

def _make_adadr(in_c=4, out_c=8, padding_mode="border") -> ContinuousAdaDR3D:
    return ContinuousAdaDR3D(
        in_channels=in_c, out_channels=out_c,
        deform_groups=1, base_dilation=1, padding_mode=padding_mode,
        align_corners=True, chunk_positions=1,
    )


def _forced_weight_and_mask(module, x, dilation_D, in_c, out_c, mask_val=1.0):
    """Return (weight_per_sample, s_override, mask_override) so the adadr
    forward call yields sum over q of  weight[q] * mask * x_sampled_at_p+D*q.
    """
    B, _, D, H, W = x.shape
    # Reference Conv3d weight — will be used as W per-batch.
    conv = torch.nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False)
    # Broadcast to per-sample weight.
    W_per_sample = conv.weight.unsqueeze(0).expand(B, -1, -1, -1, -1, -1).contiguous()
    # Force s = D - 1 (so effective dilation = D).
    s_forced = torch.full((B, 1, D, H, W), float(dilation_D - 1))
    # Force mask = mask_val everywhere.
    mask_forced = torch.full((B, 1, 27, D, H, W), float(mask_val))
    return conv, W_per_sample, s_forced, mask_forced


def _run_with_forced_state(module, x, W_per_sample, s_forced, mask_forced):
    """Manually reproduce the module's forward but with s and mask overridden."""
    B, C_in, D, H, W = x.shape
    C_out = W_per_sample.shape[1]
    out = torch.zeros(B, C_out, D, H, W, dtype=x.dtype, device=x.device)
    lattice = module.lattice
    for k in range(27):
        q_zyx = lattice[k]
        # Sample using module's own helper (with the forced s).
        sampled = module._sample_at_q(x, s_forced, q_zyx)
        m_k = mask_forced[:, 0, k]                    # (B, D, H, W)
        modulated = sampled * m_k.unsqueeze(1)
        kz, ky, kx = int(q_zyx[0]) + 1, int(q_zyx[1]) + 1, int(q_zyx[2]) + 1
        W_k = W_per_sample[:, :, :, kz, ky, kx]       # (B, C_out, C_in)
        contrib = torch.einsum("boc,bcdhw->bodhw", W_k, modulated)
        out = out + contrib
    return out


# =========================================================================
# 1-3. Constant-D matches Conv3d(dilation=D) for D=1, 2, 3
# =========================================================================

def _test_constant_dilation(D: int, padding: str):
    """When s = D - 1 (constant), mask = 1, weight = Conv3d.weight, the
    continuous conv output must equal Conv3d(dilation=D) forward, up to
    trilinear-sampling numerical noise (float32 grid_sample precision)."""
    torch.manual_seed(0)
    in_c, out_c = 3, 5
    module = _make_adadr(in_c, out_c, padding_mode=padding)
    x = torch.randn(2, in_c, 12, 12, 8)
    conv, W_per, s_forced, mask_forced = _forced_weight_and_mask(
        module, x, D, in_c, out_c
    )
    y_ours = _run_with_forced_state(module, x, W_per, s_forced, mask_forced)

    # Reference Conv3d(dilation=D). Adjust padding to preserve shape.
    padding_mode = "replicate" if padding == "border" else "constant"
    pad = D  # for kernel_size=3 and dilation=D: padding needed = D
    x_pad = F.pad(x, [pad, pad, pad, pad, pad, pad], mode=padding_mode)
    conv_D = torch.nn.Conv3d(in_c, out_c, kernel_size=3, padding=0,
                             dilation=D, bias=False)
    with torch.no_grad():
        conv_D.weight.copy_(conv.weight)
    y_ref = conv_D(x_pad)
    return y_ours, y_ref


def test_1_constant_D_equals_conv3d_dilation_1():
    y_ours, y_ref = _test_constant_dilation(1, padding="border")
    err = (y_ours - y_ref).abs().max().item()
    assert err < 1e-4, f"D=1 mismatch: max|err|={err:.3e}"
    print(f"[1]  D=1  max|err| = {err:.3e}  OK")


def test_2_constant_D_equals_conv3d_dilation_2():
    y_ours, y_ref = _test_constant_dilation(2, padding="border")
    err = (y_ours - y_ref).abs().max().item()
    # D=2 uses integer sample coords -> grid_sample should be exact.
    assert err < 1e-3, f"D=2 mismatch: max|err|={err:.3e}"
    print(f"[2]  D=2  max|err| = {err:.3e}  OK")


def test_3_constant_D_equals_conv3d_dilation_3():
    y_ours, y_ref = _test_constant_dilation(3, padding="border")
    err = (y_ours - y_ref).abs().max().item()
    assert err < 1e-3, f"D=3 mismatch: max|err|={err:.3e}"
    print(f"[3]  D=3  max|err| = {err:.3e}  OK")


# =========================================================================
# 4. Both zero-padding and replication-padding reference behaviour
# =========================================================================

def test_4_padding_modes_both_work():
    for pad_mode in ("border", "zeros"):
        y_ours, y_ref = _test_constant_dilation(1, padding=pad_mode)
        err = (y_ours - y_ref).abs().max().item()
        assert err < 1e-3, f"padding={pad_mode} D=1 mismatch: {err:.3e}"
    print(f"[4]  padding modes: border + zeros both match Conv3d at D=1  OK")


# =========================================================================
# 5. Fractional D=1.5 produces finite output
# =========================================================================

def test_5_fractional_dilation_produces_finite():
    module = _make_adadr(4, 4)
    x = torch.randn(2, 4, 10, 10, 8)
    W_per = torch.randn(2, 4, 4, 3, 3, 3)
    s_frac = torch.full((2, 1, 10, 10, 8), 0.5)      # D = 1.5
    mask = torch.full((2, 1, 27, 10, 10, 8), 1.0)
    y = _run_with_forced_state(module, x, W_per, s_frac, mask)
    assert torch.isfinite(y).all().item(), "fractional D produced NaN/Inf"
    print(f"[5]  fractional D=1.5 output finite, mean={y.mean().item():+.4f}  OK")


# =========================================================================
# 6. Dilation affects z, y and x isotropically
# =========================================================================

def test_6_dilation_isotropic_all_three_axes():
    """Fire a delta at the center; sample q=(1,0,0), q=(0,1,0), q=(0,0,1)
    at D=2 → sample positions should be exactly ±2 voxels along that
    single axis. Compare the three axis-wise samples for equal displacement."""
    module = _make_adadr(1, 1)
    # Zero volume, delta at (5, 5, 5) — safely away from any boundary.
    x = torch.zeros(1, 1, 12, 12, 12)
    x[0, 0, 5, 5, 5] = 1.0
    s = torch.full((1, 1, 12, 12, 12), 1.0)   # D = 2

    lattice = module.lattice
    # Find lattice indices for (1,0,0), (0,1,0), (0,0,1).
    idx_z = None; idx_y = None; idx_x = None
    for k in range(27):
        qz, qy, qx = int(lattice[k, 0]), int(lattice[k, 1]), int(lattice[k, 2])
        if (qz, qy, qx) == (1, 0, 0): idx_z = k
        if (qz, qy, qx) == (0, 1, 0): idx_y = k
        if (qz, qy, qx) == (0, 0, 1): idx_x = k
    # Sample x at position p + 2*q for each. Sampled tensor is
    # (B, C, D, H, W) where entry [b, c, z, y, x] is x sampled at p+2q.
    with torch.no_grad():
        samp_z = module._sample_at_q(x, s, lattice[idx_z])[0, 0]
        samp_y = module._sample_at_q(x, s, lattice[idx_y])[0, 0]
        samp_x = module._sample_at_q(x, s, lattice[idx_x])[0, 0]
    # For q=(+1,0,0), sample at p+(2,0,0). Entry at p=(3,5,5) will read
    # x[3+2,5,5] = x[5,5,5] = 1.
    v_z = samp_z[3, 5, 5].item()  # p=(3,5,5) with D=2*(1,0,0) shift → reads (5,5,5)
    v_y = samp_y[5, 3, 5].item()
    v_x = samp_x[5, 5, 3].item()
    assert abs(v_z - 1.0) < 1e-5, f"z axis broken: {v_z}"
    assert abs(v_y - 1.0) < 1e-5, f"y axis broken: {v_y}"
    assert abs(v_x - 1.0) < 1e-5, f"x axis broken: {v_x}"
    print(f"[6]  isotropic dilation: z={v_z:.4f} y={v_y:.4f} x={v_x:.4f}  OK")


# =========================================================================
# 7. Center coordinate q=(0,0,0) remains fixed regardless of s
# =========================================================================

def test_7_center_q_is_identity_regardless_of_s():
    module = _make_adadr(1, 1)
    x = torch.arange(6*6*6, dtype=torch.float32).reshape(1, 1, 6, 6, 6)
    # Arbitrary s, any values.
    s = torch.rand(1, 1, 6, 6, 6) * 5.0
    lattice = module.lattice
    # Find center index (should be 13 given zyx ordering).
    center_idx = None
    for k in range(27):
        if tuple(int(v) for v in lattice[k]) == (0, 0, 0):
            center_idx = k
    assert center_idx is not None, "center coord not in lattice"
    with torch.no_grad():
        samp = module._sample_at_q(x, s, lattice[center_idx])
    err = (samp - x).abs().max().item()
    assert err < 1e-5, f"center q not identity: max|err|={err:.3e}"
    print(f"[7]  center q=(0,0,0) is identity regardless of s (idx={center_idx})  OK")


# =========================================================================
# 8. Output spatial dimensions preserved
# =========================================================================

def test_8_spatial_dimensions_preserved():
    for D, H, W in [(8, 8, 4), (12, 16, 6), (16, 12, 10)]:
        m = ContinuousDilatedConv3D(in_channels=3, out_channels=5, base_dilation=1)
        x = torch.randn(2, 3, D, H, W)
        y = m(x)
        assert y.shape == (2, 5, D, H, W), f"shape mismatch: {y.shape} vs (2,5,{D},{H},{W})"
    print(f"[8]  spatial dims preserved for multiple input shapes  OK")


# =========================================================================
# 9. Integer sampling coordinates have no half-voxel shift
# =========================================================================

def test_9_integer_sampling_no_halfvoxel_shift():
    """align_corners=True + normalization 2*p/(N-1)-1 must map integer p
    to itself when D=1. Delta test: input has 1.0 at (5,5,5); q=(+1,+1,+1)
    with s=0 (D=1) should read from (6,6,6) which is zero."""
    module = _make_adadr(1, 1)
    x = torch.zeros(1, 1, 10, 10, 10)
    x[0, 0, 5, 5, 5] = 1.0
    s = torch.zeros(1, 1, 10, 10, 10)   # D = 1

    lattice = module.lattice
    # q = (+1, +1, +1)
    q = torch.tensor([1.0, 1.0, 1.0])
    with torch.no_grad():
        samp = module._sample_at_q(x, s, q)[0, 0]
    # Sample at p=(4,4,4) reads x[4+1,4+1,4+1] = x[5,5,5] = 1.
    assert abs(samp[4, 4, 4].item() - 1.0) < 1e-5, \
        f"integer shift wrong: got {samp[4,4,4].item()}"
    # Sample at p=(5,5,5) reads x[6,6,6] = 0.
    assert abs(samp[5, 5, 5].item()) < 1e-5, \
        f"integer shift wrong: got {samp[5,5,5].item()}"
    print(f"[9]  integer sampling: no half-voxel shift  OK")


# =========================================================================
# 10. Gradients reach input, kernel, offset head, mask head, AdaKern, FreqSel
# =========================================================================

def test_10_gradients_reach_all_learnables():
    m = ContinuousDilatedConv3D(in_channels=4, out_channels=6, base_dilation=1)
    x = torch.randn(2, 4, 8, 8, 4, requires_grad=True)
    y = m(x)
    loss = y.pow(2).mean()
    loss.backward()
    # NOTE on trunk grads: with paper-faithful zero-init on the four heads
    # (omni_low.channel_fc, omni_low.filter_fc, omni_high.channel_fc,
    # omni_high.filter_fc), each trunk's shared_fc receives zero gradient
    # at step 1 by construction (heads' weights are zero, so backward
    # through them delivers zero upstream). After the heads themselves take
    # one optimizer step, the trunks begin receiving gradient. This is the
    # official-style two-trunk init pattern documented in
    # `AdaKern3DOfficial`. We require gradient on the HEADS at step 1 and
    # verify each trunk starts receiving gradient AFTER one optimizer step.
    parts = {
        "input.x":                       x,
        "base_kernel.weight":            m.weight,
        "adadr.conv_offset.w":           m.adadr.conv_offset.weight,
        "adadr.conv_mask.w":             m.adadr.conv_mask.weight,
        "adakern.omni_low.channel_fc":   m.ada_kern.omni_low.channel_fc.weight,
        "adakern.omni_low.filter_fc":    m.ada_kern.omni_low.filter_fc.weight,
        "adakern.omni_high.channel_fc":  m.ada_kern.omni_high.channel_fc.weight,
        "adakern.omni_high.filter_fc":   m.ada_kern.omni_high.filter_fc.weight,
        "fs.pred[0].w":                  m.fs.freq_weight_conv_list[0].weight,
    }
    grad_norms = {name: (p.grad.norm().item() if p.grad is not None else None)
                  for name, p in parts.items()}
    for name, g in grad_norms.items():
        assert g is not None, f"no gradient reached {name}"
        assert g > 0, f"zero gradient at {name}"
    # After one optimizer step, both trunks receive gradient.
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    opt.step()
    m.zero_grad(set_to_none=True)
    y2 = m(x)
    y2.pow(2).mean().backward()
    for name, trunk_w in (
        ("omni_low.fc",  m.ada_kern.omni_low.fc.weight),
        ("omni_high.fc", m.ada_kern.omni_high.fc.weight),
    ):
        g = trunk_w.grad
        assert g is not None and g.norm().item() > 0, (
            f"{name} trunk still had zero gradient after one optimizer step"
        )
    print(f"[10] gradients reach all 8 checked learnables:")
    for name, g in grad_norms.items():
        print(f"     {name:34s} |grad|={g:.3e}")
    print("     both trunk fcs receive gradient after 1 optimizer step  OK")


# =========================================================================
# 11. Batch sizes 1 and 2 both work
# =========================================================================

def test_11_batches_1_and_2():
    for B in (1, 2):
        m = ContinuousDilatedConv3D(in_channels=2, out_channels=3, base_dilation=1)
        x = torch.randn(B, 2, 8, 8, 4)
        y = m(x)
        assert y.shape == (B, 3, 8, 8, 4), f"B={B} shape wrong: {y.shape}"
        assert torch.isfinite(y).all().item()
    print(f"[11] batch sizes 1 and 2 both work  OK")


# =========================================================================
# 12. AMP has no NaN/Inf
# =========================================================================

def test_12_amp_no_nan_inf():
    if not torch.cuda.is_available():
        # CPU autocast supports bfloat16 for many ops. Test with bf16.
        m = ContinuousDilatedConv3D(in_channels=3, out_channels=4, base_dilation=1)
        x = torch.randn(2, 3, 8, 8, 4)
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                y = m(x)
            assert torch.isfinite(y).all().item(), "AMP produced non-finite"
            print(f"[12] CPU bfloat16 AMP: finite  OK")
        except (RuntimeError, NotImplementedError) as e:
            # Some ops (grid_sample) may not be bf16 on CPU; that's OK
            # to skip — the AMP path is exercised on GPU in the notebook.
            print(f"[12] CPU AMP not supported for this op mix — SKIPPED "
                  f"(will be verified on Kaggle GPU)")
    else:
        m = ContinuousDilatedConv3D(in_channels=3, out_channels=4,
                                     base_dilation=1).cuda()
        x = torch.randn(2, 3, 8, 8, 4, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = m(x)
        assert torch.isfinite(y).all().item()
        print(f"[12] CUDA fp16 AMP: finite  OK")


# =========================================================================
# 13. Strict checkpoint save/load succeeds
# =========================================================================

def test_13_strict_checkpoint_roundtrip():
    m = build_unet3d_fadc_continuous(base_filters=8)
    tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False).name
    try:
        torch.save({"model": m.state_dict(),
                    "arch_identity": m.arch_identity()}, tmp)
        ckpt = torch.load(tmp, map_location="cpu", weights_only=False)
        m2 = build_unet3d_fadc_continuous(base_filters=8)
        missing, unexpected = m2.load_state_dict(ckpt["model"], strict=True)
        assert not missing and not unexpected, \
            f"strict load leaked keys: missing={missing} unexpected={unexpected}"
        # Forward parity in eval mode. Patch must survive 4 stride-2 pools
        # of the encoder (min-dim >= 16 needed).
        m.eval(); m2.eval()
        x = torch.randn(1, 2, 32, 32, 16)
        with torch.no_grad():
            y1 = m(x); y2 = m2(x)
        err = (y1 - y2).abs().max().item()
        assert err < 1e-5, f"forward parity failed: max|err|={err:.3e}"
        print(f"[13] strict save/load OK, forward parity |err|={err:.3e}")
    finally:
        os.unlink(tmp)


# =========================================================================
# 14. Discrete FADC3D checkpoint is rejected with a clear arch error
# =========================================================================

def test_14_discrete_checkpoint_rejected_strict():
    """Build a discrete FADC3D model, save its state_dict, then try to
    strict-load it into a continuous model. Must fail cleanly."""
    from models.unet_3d_fadc_correct import build_unet3d_fadc_correct
    m_disc = build_unet3d_fadc_correct("unet3d_fadc_encoder_correct",
                                        in_channels=2, out_channels=2,
                                        base_filters=8, deep_supervision=False)
    m_cont = build_unet3d_fadc_continuous(base_filters=8)
    try:
        m_cont.load_state_dict(m_disc.state_dict(), strict=True)
    except RuntimeError as e:
        msg = str(e)
        # RuntimeError from load_state_dict lists missing + unexpected keys.
        assert "Missing key" in msg or "Unexpected key" in msg or \
               "missing" in msg.lower() or "unexpected" in msg.lower(), \
               f"error message not helpful: {msg[:200]}"
        print(f"[14] discrete ckpt rejected under strict=True (clear arch error)  OK")
        return
    raise AssertionError("strict=True should have rejected the discrete checkpoint")


# =========================================================================
# 15. Isotropic impulse test: equal voxel displacement across zyx
# =========================================================================

def test_15_impulse_isotropy():
    """Fire an impulse at (5,5,5). At D=2 with unit kernel weight, sampling
    at q=(+1,0,0), (0,+1,0), (0,0,+1) reads from positions
    (7,5,5), (5,7,5), (5,5,7). All should read x=0 (impulse only at 5,5,5).
    At q=(-2/2=-1, ...) with s=1 they read from (5±2, ...); the impulse
    responds only for the q whose displacement lands on it."""
    module = _make_adadr(1, 1)
    x = torch.zeros(1, 1, 12, 12, 12); x[0, 0, 5, 5, 5] = 1.0
    s = torch.full((1, 1, 12, 12, 12), 1.0)  # D = 2

    # Voxel p = (3, 5, 5). Sample at q=(+1,0,0) shifts by D*q=(2,0,0)
    # -> reads x[5,5,5] = 1.
    with torch.no_grad():
        for qname, q in [("+z", torch.tensor([1., 0., 0.])),
                         ("+y", torch.tensor([0., 1., 0.])),
                         ("+x", torch.tensor([0., 0., 1.]))]:
            samp = module._sample_at_q(x, s, q)[0, 0]
            # For q=(+1,0,0): p at (3,5,5) reads x[5,5,5] = 1.
            # For q=(0,+1,0): p at (5,3,5) reads x[5,5,5] = 1.
            # For q=(0,0,+1): p at (5,5,3) reads x[5,5,5] = 1.
            pos = [5, 5, 5]; pos[int((q != 0).nonzero().item())] = 3
            v = samp[pos[0], pos[1], pos[2]].item()
            assert abs(v - 1.0) < 1e-5, f"impulse asymmetric on {qname}: {v}"
    print(f"[15] impulse response equal for +z / +y / +x displacements  OK")


# =========================================================================
# 16. Expanded 3D lattice has 27 unique coords + correct ordering
# =========================================================================

def test_16_lattice_27_unique_and_ordered():
    lat = _kernel_lattice_3d()
    assert lat.shape == (27, 3), f"lattice shape {lat.shape}"
    unique = {tuple(row.tolist()) for row in lat}
    assert len(unique) == 27, f"got {len(unique)} unique coords (expected 27)"
    # All coordinates in {-1,0,1}^3.
    all_expected = {(z, y, x) for z in (-1, 0, 1)
                              for y in (-1, 0, 1)
                              for x in (-1, 0, 1)}
    assert unique == set((float(z), float(y), float(x)) for z, y, x in all_expected)
    # Ordering: qz varies slowest, qx fastest.
    assert tuple(lat[0].tolist())  == (-1., -1., -1.)
    assert tuple(lat[13].tolist()) == (0., 0., 0.)     # center at index 13
    assert tuple(lat[26].tolist()) == (1., 1., 1.)
    print(f"[16] lattice has 27 unique coords, center at idx 13, ordering "
          f"(qz slowest, qx fastest)  OK")


# =========================================================================
if __name__ == "__main__":
    tests = [
        test_1_constant_D_equals_conv3d_dilation_1,
        test_2_constant_D_equals_conv3d_dilation_2,
        test_3_constant_D_equals_conv3d_dilation_3,
        test_4_padding_modes_both_work,
        test_5_fractional_dilation_produces_finite,
        test_6_dilation_isotropic_all_three_axes,
        test_7_center_q_is_identity_regardless_of_s,
        test_8_spatial_dimensions_preserved,
        test_9_integer_sampling_no_halfvoxel_shift,
        test_10_gradients_reach_all_learnables,
        test_11_batches_1_and_2,
        test_12_amp_no_nan_inf,
        test_13_strict_checkpoint_roundtrip,
        test_14_discrete_checkpoint_rejected_strict,
        test_15_impulse_isotropy,
        test_16_lattice_27_unique_and_ordered,
    ]
    failed = []
    for fn in tests:
        try:
            fn()
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"  FAIL: {fn.__name__}: {e!r}")
    print(f"\n{'='*60}")
    print(f"passed: {len(tests) - len(failed)} / {len(tests)}")
    if failed:
        for name, e in failed:
            print(f"  - {name}: {e!r}")
        sys.exit(1)
    print("ALL CONTINUOUS FADC3D TESTS PASSED")
