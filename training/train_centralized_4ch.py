"""
4-channel training script — sibling of training/train_centralized.py.

Only differences from the 2ch version:
  - Imports build_centralized_loaders_4ch from data.mama_mia_dataset_4ch
  - Supports --phase_aug / --p_phase_aug flags for runtime control
  - Default --config points to configs/config_4ch.yaml (in_channels=4)
  - Default --output_dir uses 4ch naming
  - Only baseline UNet3D supported here for the first 4ch probe; add FADC
    variants later by extending the model-selection block if 4ch probe succeeds.
"""

import os
import sys
import time
import json
import random
import argparse
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

sys.path.append(str(Path(__file__).parent.parent))
from data.mama_mia_dataset_4ch import build_centralized_loaders_4ch, DATA_ROOT
from models.unet_3d import UNet3D
from training.losses import DiceCELoss


def parse_args():
    parser = argparse.ArgumentParser()
    PROJECT_ROOT = str(Path(__file__).parent.parent)
    parser.add_argument("--config",      type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "config_4ch.yaml"))
    parser.add_argument("--data_root",   type=str, default=DATA_ROOT)
    parser.add_argument("--output_dir",  type=str, default="outputs/4ch_unet3d")
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--batch_size",  type=int, default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--preprocessed_cache_dir", type=str, default=None,
                        help="Path to 4ch .npz cache from scripts/4ch_preprocess_to_cache.py")
    parser.add_argument("--resume",      type=str, default=None)
    parser.add_argument("--smoke_test",  action="store_true")
    parser.add_argument("--model",       type=str, default="unet3d", choices=["unet3d"],
                        help="Only baseline UNet3D for the first 4ch probe")
    parser.add_argument("--patch_size",  type=int, nargs=3, default=None,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--seed",        type=int, default=None)
    parser.add_argument("--phase_aug",   type=int, default=None,
                        help="Override config: 1=enable phase aug, 0=disable")
    parser.add_argument("--p_phase_aug", type=float, default=None,
                        help="Override config: probability per sample of phase permutation")
    return parser.parse_args()


def load_config(config_path: str, args) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if args.epochs      is not None: cfg["training"]["epochs"]      = args.epochs
    if args.batch_size  is not None: cfg["training"]["batch_size"]  = args.batch_size
    if args.lr          is not None: cfg["training"]["lr"]          = args.lr
    if args.num_workers is not None: cfg["data"]["num_workers"]     = args.num_workers
    if args.patch_size  is not None: cfg["data"]["patch_size"]      = args.patch_size
    if args.phase_aug   is not None: cfg["data"]["phase_aug"]       = bool(args.phase_aug)
    if args.p_phase_aug is not None: cfg["data"]["p_phase_aug"]     = args.p_phase_aug
    return cfg


def validate(model, val_loader, dice_metric, post_pred, post_label, patch_size, device):
    model.eval()
    dice_metric.reset()
    iou_list, sens_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            with autocast("cuda", enabled=device.type == "cuda"):
                preds = sliding_window_inference(
                    inputs=images, roi_size=patch_size, sw_batch_size=4,
                    predictor=model, overlap=0.0,
                )
            preds_bin  = [post_pred(i)         for i in decollate_batch(preds)]
            labels_bin = [post_label(i.long()) for i in decollate_batch(labels)]
            dice_metric(y_pred=preds_bin, y=labels_bin)
            pred_fg  = preds_bin[0][1].float()
            label_fg = labels_bin[0][1].float()
            tp = (pred_fg * label_fg).sum().item()
            fp = (pred_fg * (1 - label_fg)).sum().item()
            fn = ((1 - pred_fg) * label_fg).sum().item()
            iou_list.append(tp / (tp + fp + fn + 1e-6))
            sens_list.append(tp / (tp + fn + 1e-6))
    mean_dice = dice_metric.aggregate().item()
    mean_iou  = float(np.mean(iou_list))  if iou_list  else 0.0
    mean_sens = float(np.mean(sens_list)) if sens_list else 0.0
    dice_metric.reset()
    return mean_dice, mean_iou, mean_sens


def train(cfg, args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[seed] Seeded with seed={args.seed} (deterministic CuDNN)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_size  = tuple(cfg["data"]["patch_size"])
    epochs      = cfg["training"]["epochs"]
    lr          = cfg["training"]["lr"]
    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    val_every   = cfg["training"]["val_every"]
    phase_aug   = cfg["data"].get("phase_aug", True)
    p_phase_aug = cfg["data"].get("p_phase_aug", 0.5)

    split_csv = os.path.join(args.data_root, "train_test_splits.csv")
    if not os.path.exists(split_csv):
        split_csv = None

    if args.smoke_test:
        print("SMOKE TEST MODE — 4 cases, 2 epochs")
        epochs, num_workers, val_every = 2, 0, 1

    train_loader, val_loader = build_centralized_loaders_4ch(
        preprocessed_cache_dir=args.preprocessed_cache_dir or "",
        split_csv=split_csv,
        num_workers=num_workers,
        batch_size=batch_size,
        max_cases=4 if args.smoke_test else None,
        patch_size=patch_size,
        seed=args.seed,
        phase_aug=phase_aug,
        p_phase_aug=p_phase_aug,
    )

    assert cfg["model"]["in_channels"] == 4, "Expected in_channels=4 for 4ch run"
    model = UNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_filters=cfg["model"]["base_filters"],
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: 4ch UNet3D | Parameters: {total_params:,}")

    criterion = DiceCELoss(
        dice_weight=cfg["training"]["dice_weight"],
        ce_weight=cfg["training"]["ce_weight"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    warmup_epochs = args.warmup_epochs
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)

    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred   = AsDiscrete(argmax=True, to_onehot=2)
    post_label  = AsDiscrete(to_onehot=2)

    start_epoch = 0
    best_dice   = 0.0
    train_log   = []

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        print(f"Resumed from epoch {start_epoch} | Best Dice so far: {best_dice:.4f}")

    print(f"\nStarting 4ch training: {epochs} epochs | LR: {lr} | Batch: {batch_size}")
    print(f"Phase augmentation: {'ON (p=%.2f)' % p_phase_aug if phase_aug else 'OFF'}")
    print("=" * 70)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = epoch_dice = epoch_ce = 0.0
        num_batches = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{epochs}",
                    leave=False, ncols=110, unit="batch", file=sys.stdout)

        for batch in pbar:
            if isinstance(batch["image"], list):
                imgs = [torch.from_numpy(x.copy()) if isinstance(x, np.ndarray) else x
                        for x in batch["image"]]
                lbls = [torch.from_numpy(x.copy()) if isinstance(x, np.ndarray) else x
                        for x in batch["label"]]
                images = torch.cat(imgs, dim=0).to(device)
                labels = torch.cat(lbls, dim=0).to(device)
            else:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

            optimizer.zero_grad()
            with autocast("cuda", enabled=device.type == "cuda"):
                preds = model(images)
                total_loss, dice_loss, ce_loss = criterion(preds, labels)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            epoch_dice += dice_loss.item()
            epoch_ce   += ce_loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}",
                              "dice": f"{dice_loss.item():.4f}",
                              "ce":   f"{ce_loss.item():.4f}"}, refresh=False)
        pbar.close()
        scheduler.step()

        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        avg_ce   = epoch_ce   / num_batches
        elapsed  = time.time() - t0
        lr_now   = scheduler.get_last_lr()[0]
        mins, secs = divmod(int(elapsed), 60)
        print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Dice loss: {avg_dice:.4f} | CE loss: {avg_ce:.4f} | "
              f"LR: {lr_now:.2e} | Time: {mins}m {secs}s")

        log_entry = {"epoch": epoch + 1, "loss": avg_loss, "dice_loss": avg_dice,
                     "ce_loss": avg_ce, "lr": lr_now, "time_s": elapsed}

        if (epoch + 1) % val_every == 0:
            val_dice, val_iou, val_sens = validate(
                model, val_loader, dice_metric, post_pred, post_label, patch_size, device)
            marker = " <-- BEST" if val_dice > best_dice else ""
            print(f"  Val  Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | "
                  f"Sensitivity: {val_sens:.4f}  (best: {best_dice:.4f}){marker}")
            log_entry["val_dice"] = val_dice
            log_entry["val_iou"]  = val_iou
            log_entry["val_sensitivity"] = val_sens

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_dice": best_dice, "config": cfg},
                           output_dir / "best_model.pth")
                print(f"  *** NEW BEST  Dice {best_dice:.4f} -> {output_dir}/best_model.pth ***")

        train_log.append(log_entry)
        if (epoch + 1) % 5 == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_dice": best_dice, "config": cfg},
                       output_dir / "latest_checkpoint.pth")

    with open(output_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)
    with open(output_dir / "meta.json", "w") as f:
        json.dump({"seed": args.seed, "model": "4ch_unet3d",
                   "best_dice": best_dice, "epochs": epochs,
                   "phase_aug": phase_aug, "p_phase_aug": p_phase_aug,
                   "in_channels": 4}, f, indent=2)

    print(f"\nTraining complete. Best Val Dice: {best_dice:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config, args)
    train(cfg, args)
