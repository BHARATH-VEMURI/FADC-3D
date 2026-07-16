"""Smoke tests for v2.2 attention wiring (separated c/f trunks + residual gate).

Checks:
  * Forward shape of AdaptiveDilatedConv3DV2(k=3) is unchanged.
  * s_att is a real tensor whose trailing dims are (K, K, K).
  * k_att shape stays (B, num_branches, D, H, W).
  * FADCConvBlockV2 stores last_filter_attention AND last_filter_attention_mul
    on both inner convs (proving f_att is applied post-BN via the ready mul).
  * Residual gate mode is the default and its multiplier stays finite.
  * A backward pass produces non-zero gradients for every learnable head:
      shared_fc, c_trunk_fc, f_trunk_fc  (new v2.2 descriptor trunks)
      channel_fc  (c_att)
      filter_fc   (f_att)
      spatial_fc  (s_att)
      kernel_spatial_head (k_att)

Run:
    conda activate fadc3d
    python -m tests.test_fadc_v2_attention
"""
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from fadc_3d_v2.adaptive_dilated_conv_3d_v2 import AdaptiveDilatedConv3DV2
from fadc_3d_v2.omni_attention_3d_spatial import OmniAttention3DSpatial
from models.unet_3d_fadc_v2 import UNet3DFADC_V2, FADCConvBlockV2


def _grad_ok(param, tag):
    if param.grad is None:
        raise AssertionError(f"{tag}: grad is None")
    g = param.grad.abs().sum().item()
    if g == 0.0:
        raise AssertionError(f"{tag}: grad is all zeros")
    print(f"  {tag:<30s}  |grad|={g:.4e}  OK")


def test_forward_and_shapes():
    print("[1] forward + shape checks")
    torch.manual_seed(0)
    in_ch, out_ch, K = 4, 8, 3
    x = torch.randn(2, in_ch, 8, 12, 12)
    m = AdaptiveDilatedConv3DV2(in_ch, out_ch, kernel_size=K)

    captured = {}
    def hook(_mod, _inp, out):
        captured["c"], captured["f"], captured["s"], captured["k"] = out
    h = m.omni_att.register_forward_hook(hook)
    y = m(x)
    h.remove()

    assert y.shape == (2, out_ch, 8, 12, 12), f"unexpected y.shape={y.shape}"
    assert torch.is_tensor(captured["s"]), "s_att is not a tensor (spatial_fc disabled?)"
    assert captured["s"].shape[-3:] == (K, K, K), \
        f"s_att trailing dims {captured['s'].shape[-3:]} != (K,K,K)"
    assert captured["k"].shape == (2, len(m.dilation_list), 8, 12, 12), \
        f"k_att shape {captured['k'].shape} unexpected"
    print(f"  y.shape={tuple(y.shape)}  s_att={tuple(captured['s'].shape)}  "
          f"k_att={tuple(captured['k'].shape)}  OK")


def test_last_filter_attention_stored():
    print("[2] FADCConvBlockV2 caches last_filter_attention (+ mul) on inner convs")
    torch.manual_seed(0)
    blk = FADCConvBlockV2(in_ch=4, out_ch=8)
    x = torch.randn(2, 4, 8, 12, 12)
    _ = blk(x)
    for tag, conv in (("conv1", blk.conv1), ("conv2", blk.conv2)):
        assert torch.is_tensor(conv.last_filter_attention), \
            f"{tag}.last_filter_attention was not populated"
        assert torch.is_tensor(conv.last_filter_attention_mul), \
            f"{tag}.last_filter_attention_mul was not populated"
        assert conv.apply_filter_attention is False, \
            f"{tag}.apply_filter_attention must be False so BN sees the un-scaled conv output"
        # Residual mode is default in v2.2 → multiplier centred on 1.0.
        mul = conv.last_filter_attention_mul
        assert torch.isfinite(mul).all(), f"{tag}.last_filter_attention_mul has NaN/inf"
        assert (mul >= 0).all() and (mul <= 2).all(), \
            f"{tag}.last_filter_attention_mul out of expected [0, 2] range: min={mul.min().item()} max={mul.max().item()}"
    print("  both convs expose non-None last_filter_attention + finite mul in [0, 2]  OK")


def test_gate_modes():
    print("[2b] residual gate is default; multiply mode still works")
    torch.manual_seed(0)
    m_res = AdaptiveDilatedConv3DV2(4, 8, kernel_size=3)
    assert m_res.filter_gate_mode == 'residual', "v2.2 default filter_gate_mode should be 'residual'"
    assert m_res.channel_gate_mode == 'residual', "v2.2 default channel_gate_mode should be 'residual'"
    m_mul = AdaptiveDilatedConv3DV2(4, 8, kernel_size=3,
                                    channel_gate_mode='multiply', filter_gate_mode='multiply')
    x = torch.randn(2, 4, 8, 12, 12)
    y_res = m_res(x)
    y_mul = m_mul(x)
    assert y_res.shape == y_mul.shape == (2, 8, 8, 12, 12)
    print(f"  residual mode ready-mul range [{m_res.last_filter_attention_mul.min().item():.3f}, "
          f"{m_res.last_filter_attention_mul.max().item():.3f}]  |  "
          f"multiply mode ready-mul range [{m_mul.last_filter_attention_mul.min().item():.3f}, "
          f"{m_mul.last_filter_attention_mul.max().item():.3f}]  OK")


def test_backward_grads_all_heads():
    print("[3] backward produces gradients for shared_fc + c/f trunks + c/f/s/k heads")
    torch.manual_seed(0)
    m = AdaptiveDilatedConv3DV2(4, 8, kernel_size=3)
    x = torch.randn(2, 4, 8, 12, 12, requires_grad=False)
    y = m(x)
    y.sum().backward()

    omni = m.omni_att
    _grad_ok(omni.shared_fc.weight,   "shared_fc.weight (s_att trunk)")
    _grad_ok(omni.c_trunk_fc.weight,  "c_trunk_fc.weight (c_att trunk)")
    _grad_ok(omni.f_trunk_fc.weight,  "f_trunk_fc.weight (f_att trunk)")
    _grad_ok(omni.channel_fc.weight,  "channel_fc.weight (c_att head)")
    _grad_ok(omni.filter_fc.weight,   "filter_fc.weight  (f_att head)")
    assert omni.spatial_fc is not None, "spatial_fc missing — kernel_size=1 was passed?"
    _grad_ok(omni.spatial_fc.weight, "spatial_fc.weight (s_att head)")
    _grad_ok(omni.kernel_spatial_head.weight, "kernel_spatial_head.weight (k_att)")


def test_unet_forward_smoke():
    print("[4] UNet3DFADC_V2 forward smoke (bottleneck placement)")
    torch.manual_seed(0)
    m = UNet3DFADC_V2(in_channels=2, out_channels=2, base_filters=8,
                      fadc_placement="bottleneck")
    x = torch.randn(1, 2, 32, 32, 16)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 2, 32, 32, 16), f"unexpected y.shape={y.shape}"
    print(f"  y.shape={tuple(y.shape)}  OK")


def test_weighted_aux_loss_end_to_end():
    """Wire hooks + run the new weighted aux loss end-to-end on synthetic input."""
    print("[5] weighted attention_diversity_loss returns per-head breakdown")
    from training.train_centralized_v2 import (
        register_attention_hooks, attention_diversity_loss,
    )
    torch.manual_seed(0)
    m = UNet3DFADC_V2(in_channels=2, out_channels=2, base_filters=8,
                      fadc_placement="bottleneck")
    store, handles, hooked = register_attention_hooks(m, layer_prefixes=None)
    assert hooked, "no FADC layers were hooked — layer walker broken?"
    store.enabled = True
    x = torch.randn(2, 2, 32, 32, 16)   # batch=2 so batch-std is defined
    _ = m(x)
    store.enabled = False

    weights = {'c': 2.0, 'f': 4.0, 's': 0.75, 'k': 0.25}
    device = torch.device('cpu')
    total, per_head = attention_diversity_loss(store, target_std=0.03,
                                               device=device, weights=weights)
    for h in handles:
        h.remove()

    assert isinstance(per_head, dict) and set(per_head.keys()) == {'c', 'f', 's', 'k'}
    assert total.dim() == 0, "aux total must be scalar"
    for k, v in per_head.items():
        assert torch.is_tensor(v) and v.dim() == 0, f"aux_{k} not a scalar tensor"
    # Sanity: weighted total should equal the linear combination.
    lin = sum(weights[k] * per_head[k] for k in ('c', 'f', 's', 'k'))
    assert torch.allclose(total, lin, atol=1e-6), \
        f"weighted total {total.item():.6f} != sum of w*head {lin.item():.6f}"
    print(f"  hooked={len(hooked)} layers | "
          f"raw c={per_head['c'].item():.4f} f={per_head['f'].item():.4f} "
          f"s={per_head['s'].item():.4f} k={per_head['k'].item():.4f} | "
          f"weighted_total={total.item():.4f}  OK")


def main():
    test_forward_and_shapes()
    test_last_filter_attention_stored()
    test_gate_modes()
    test_backward_grads_all_heads()
    test_unet_forward_smoke()
    test_weighted_aux_loss_end_to_end()
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
