"""Smoke tests for the v2.2 attention wiring in AdaptiveDilatedConv3DV2 and
FADCConvBlockV2.

Checks:
  * Forward shape of AdaptiveDilatedConv3DV2(k=3) is unchanged.
  * s_att is a real tensor whose trailing dims are (K, K, K).
  * k_att shape stays (B, num_branches, D, H, W).
  * FADCConvBlockV2 stores last_filter_attention on both inner convs
    (proving f_att is being applied post-BN by the block).
  * A backward pass produces non-zero gradients for:
      channel_fc  (c_att)
      filter_fc   (f_att)
      spatial_fc  (s_att — the whole point of the v2.2 change)
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
    print("[2] FADCConvBlockV2 caches last_filter_attention on inner convs")
    torch.manual_seed(0)
    blk = FADCConvBlockV2(in_ch=4, out_ch=8)
    x = torch.randn(2, 4, 8, 12, 12)
    _ = blk(x)
    for tag, conv in (("conv1", blk.conv1), ("conv2", blk.conv2)):
        assert torch.is_tensor(conv.last_filter_attention), \
            f"{tag}.last_filter_attention was not populated"
        assert conv.apply_filter_attention is False, \
            f"{tag}.apply_filter_attention must be False so BN sees the un-scaled conv output"
    print("  both convs expose non-None last_filter_attention  OK")


def test_backward_grads_all_heads():
    print("[3] backward produces gradients for c/f/s/k heads")
    torch.manual_seed(0)
    m = AdaptiveDilatedConv3DV2(4, 8, kernel_size=3)
    x = torch.randn(2, 4, 8, 12, 12, requires_grad=False)
    y = m(x)
    y.sum().backward()

    omni = m.omni_att
    _grad_ok(omni.channel_fc.weight, "channel_fc.weight (c_att)")
    _grad_ok(omni.filter_fc.weight,  "filter_fc.weight (f_att)")
    assert omni.spatial_fc is not None, "spatial_fc missing — kernel_size=1 was passed?"
    _grad_ok(omni.spatial_fc.weight, "spatial_fc.weight (s_att)")
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


def main():
    test_forward_and_shapes()
    test_last_filter_attention_stored()
    test_backward_grads_all_heads()
    test_unet_forward_smoke()
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
