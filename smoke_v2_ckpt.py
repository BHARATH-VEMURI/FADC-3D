"""Runtime smoke test for the Encoder v2 ep25 checkpoint.

Standalone -- no MONAI dependency. Verifies:
  1. torch.load succeeds
  2. UNet3DFADC_V2 constructs correctly with the ckpt's config
  3. state_dict loads with strict=True (no missing/unexpected keys)
  4. Forward pass produces the expected output shape
  5. (bonus) infer_with_tta.build_model dispatches to UNet3DFADC_V2 for
     the model name unet3d_fadc_encoder_v2 -- validates today's Q21 fix
     on a real checkpoint if MONAI is available in the env.

Run:
    conda activate fadc3d
    python smoke_v2_ckpt.py
"""
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))
from models.unet_3d_fadc_v2 import UNet3DFADC_V2


CKPT = r"C:\Users\bhara\Downloads\check\fadcencoder_V2\seed42_epoch_25\best_model.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}")
    print(f"ckpt   : {CKPT}\n")

    # 1. Load
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    print("--- checkpoint metadata ---")
    print(f"epoch (0-indexed): {ckpt.get('epoch')}")
    print(f"best_dice        : {ckpt.get('best_dice'):.4f}")
    print(f"model_name stamp : {ckpt.get('model_name', '<none -- pre-fix run, expected>')}")
    cfg = ckpt["config"]
    print(f"fadc_version     : {cfg['model'].get('fadc_version')}")
    print(f"patch_size       : {cfg['data']['patch_size']}")
    print(f"k_att anneal     : {cfg['training'].get('k_att_temp_start')} -> "
          f"{cfg['training'].get('k_att_temp_end')}")

    # 2. Build model directly (bypasses infer_with_tta to isolate dependencies)
    print("\n--- direct build + load ---")
    model = UNet3DFADC_V2(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_filters=cfg["model"]["base_filters"],
        fadc_placement="encoder",
        deep_supervision=cfg["model"].get("deep_supervision", False),
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet3DFADC_V2(fadc_placement='encoder') built -- {n_params:,} params")

    # 3. strict=True load -- the real test of state_dict compatibility
    missing_unexpected = model.load_state_dict(ckpt["model"], strict=True)
    print("state_dict loaded OK (strict=True) -- no missing/unexpected keys")

    # 4. Forward pass (small patch so CPU is fine)
    print("\n--- forward pass ---")
    small = (1, 2, 32, 32, 16)
    x = torch.randn(*small, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == small, f"unexpected output shape {y.shape}"
    print(f"input  {tuple(x.shape)} -> output {tuple(y.shape)} OK")

    # 5. Bonus: verify infer_with_tta.build_model dispatches correctly on this ckpt
    print("\n--- bonus: infer_with_tta.build_model dispatch ---")
    try:
        from infer_with_tta import build_model as tta_build
        m2 = tta_build("unet3d_fadc_encoder_v2", cfg, device)
        assert type(m2).__name__ == "UNet3DFADC_V2", \
            f"Q21 fix broken -- got {type(m2).__name__}"
        m2.load_state_dict(ckpt["model"], strict=True)
        print("build_model('unet3d_fadc_encoder_v2') -> UNet3DFADC_V2 OK")
        print("state_dict loads via inference-script path OK")
    except ImportError as e:
        print(f"skipped (MONAI not in env): {e}")

    print("\n=== SMOKE PASSED ===")


if __name__ == "__main__":
    main()
