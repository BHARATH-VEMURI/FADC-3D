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
from data.mama_mia_dataset import build_centralized_loaders, DATA_ROOT
from models.unet_3d import UNet3D
from models.unet_3d_fadc import UNet3DFADC
from training.losses import DiceCELoss


# ─────────────────────────────────────────────
# 1. ARGUMENT PARSING
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    PROJECT_ROOT = str(Path(__file__).parent.parent)
    parser.add_argument("--config",      type=str, default=os.path.join(PROJECT_ROOT, "configs", "config.yaml"))
    parser.add_argument("--data_root",   type=str, default=DATA_ROOT)
    parser.add_argument("--output_dir",  type=str, default="outputs/unet3d")
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--batch_size",  type=int, default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--cache_rate",  type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--persistent_cache_dir",    type=str, default=None)
    parser.add_argument("--preprocessed_cache_dir",  type=str, default=None,
                        help="Path to .npz cache from scripts/preprocess_to_cache.py (fastest)")
    parser.add_argument("--resume",      type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--smoke_test",  action="store_true",
                        help="Run 2 epochs on 4 cases to verify pipeline end-to-end")
    parser.add_argument("--model",       type=str, default="unet3d",
                        choices=["unet3d", "unet3d_fadc",
                                 "unet3d_fadc_encoder", "unet3d_fadc_bottleneck",
                                 "unet3d_fadc_mid", "unet3d_fadc_deep",
                                 "unet3d_fadc_deep_lite"],
                        help="Model architecture to train")
    parser.add_argument("--patch_size",  type=int, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Override patch size, e.g. --patch_size 96 96 48")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear LR warmup epochs (0 to disable)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. If set, enables "
                             "cudnn.deterministic=True and seeds torch/numpy/random + DataLoader workers.")
    parser.add_argument("--deep_supervision", action="store_true",
                        help="Enable nnU-Net style deep supervision (aux heads at dec2/dec3/dec4). "
                             "Only the main head is used at inference.")
    return parser.parse_args()


# nnU-Net deep-supervision weights at (main, dec2, dec3, dec4), normalized to sum to 1.
_DS_WEIGHTS = [1.0, 0.5, 0.25, 0.125]
_DS_WEIGHTS = [w / sum(_DS_WEIGHTS) for w in _DS_WEIGHTS]
_DS_SCALES  = [1.0, 0.5, 0.25, 0.125]  # spatial scale of each head relative to main


def deep_supervision_loss(preds_tuple, label, criterion):
    """Weighted Dice+CE across DS scales. Returns (total, main_dice, main_ce)."""
    total = 0.0
    main_dice = None
    main_ce   = None
    for i, (p, w, s) in enumerate(zip(preds_tuple, _DS_WEIGHTS, _DS_SCALES)):
        if s == 1.0:
            lbl = label
        else:
            lbl = F.interpolate(label.float(), scale_factor=s, mode="nearest").to(label.dtype)
        sub_total, sub_dice, sub_ce = criterion(p, lbl)
        total = total + w * sub_total
        if i == 0:
            main_dice = sub_dice
            main_ce   = sub_ce
    return total, main_dice, main_ce


# ─────────────────────────────────────────────
# 2. CONFIG LOADER
# ─────────────────────────────────────────────

def load_config(config_path: str, args) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.epochs      is not None: cfg["training"]["epochs"]      = args.epochs
    if args.batch_size  is not None: cfg["training"]["batch_size"]  = args.batch_size
    if args.lr          is not None: cfg["training"]["lr"]          = args.lr
    if args.cache_rate  is not None: cfg["data"]["cache_rate"]      = args.cache_rate
    if args.num_workers is not None: cfg["data"]["num_workers"]     = args.num_workers
    if args.patch_size  is not None: cfg["data"]["patch_size"]      = args.patch_size

    # Stamp DS flag into cfg so it's saved with the checkpoint and the viz cell
    # knows whether to instantiate the model with aux heads.
    cfg["model"]["deep_supervision"] = bool(args.deep_supervision)

    return cfg


# ─────────────────────────────────────────────
# 3. VALIDATION — SLIDING WINDOW + METRICS
# ─────────────────────────────────────────────

def validate(model, val_loader, dice_metric, post_pred, post_label, patch_size, device):
    """Returns (mean_dice, mean_iou, mean_sensitivity) over all validation cases."""
    model.eval()
    dice_metric.reset()
    iou_list  = []
    sens_list = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast("cuda", enabled=device.type == "cuda"):
                preds = sliding_window_inference(
                    inputs=images,
                    roi_size=patch_size,
                    sw_batch_size=4,
                    predictor=model,
                    overlap=0.0,
                )

            preds_bin  = [post_pred(i)          for i in decollate_batch(preds)]
            labels_bin = [post_label(i.long())  for i in decollate_batch(labels)]

            dice_metric(y_pred=preds_bin, y=labels_bin)

            # Per-case IoU (Jaccard) and Sensitivity on the foreground channel
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


# ─────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────

def train(cfg, args):
    # Apply reproducibility seeding BEFORE building model / loaders / optimizer.
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[seed] Seeded torch/numpy/random/cudnn with seed={args.seed} "
              f"(cudnn.deterministic=True, benchmark=False)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_size  = tuple(cfg["data"]["patch_size"])
    epochs      = cfg["training"]["epochs"]
    lr          = cfg["training"]["lr"]
    batch_size  = cfg["training"]["batch_size"]
    cache_rate  = cfg["data"]["cache_rate"]
    num_workers = cfg["data"]["num_workers"]
    val_every   = cfg["training"]["val_every"]

    # ── Data ──────────────────────────────────
    split_csv = os.path.join(args.data_root, "train_test_splits.csv")
    if not os.path.exists(split_csv):
        split_csv = None
        print("Warning: train_test_splits.csv not found — using all data for train")

    if args.smoke_test:
        print("SMOKE TEST MODE — 4 cases, 2 epochs")
        epochs      = 2
        cache_rate  = 0.0
        num_workers = 0
        val_every   = 1

    train_loader, val_loader = build_centralized_loaders(
        data_root=args.data_root,
        split_csv=split_csv,
        cache_rate=cache_rate,
        num_workers=num_workers,
        batch_size=batch_size,
        max_cases=4 if args.smoke_test else None,
        persistent_cache_dir=args.persistent_cache_dir or "",
        preprocessed_cache_dir=args.preprocessed_cache_dir or "",
        patch_size=patch_size,
        seed=args.seed,
    )

    # ── Model ─────────────────────────────────
    model_kwargs = dict(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_filters=cfg["model"]["base_filters"],
    )
    fadc_kwargs = dict(deep_supervision=args.deep_supervision)
    if args.model == "unet3d_fadc":
        model = UNet3DFADC(**model_kwargs, fadc_placement='full', **fadc_kwargs).to(device)
    elif args.model == "unet3d_fadc_encoder":
        model = UNet3DFADC(**model_kwargs, fadc_placement='encoder', **fadc_kwargs).to(device)
    elif args.model == "unet3d_fadc_bottleneck":
        model = UNet3DFADC(**model_kwargs, fadc_placement='bottleneck', **fadc_kwargs).to(device)
    elif args.model == "unet3d_fadc_mid":
        model = UNet3DFADC(**model_kwargs, fadc_placement='mid', **fadc_kwargs).to(device)
    elif args.model == "unet3d_fadc_deep":
        model = UNet3DFADC(**model_kwargs, fadc_placement='deep', **fadc_kwargs).to(device)
    elif args.model == "unet3d_fadc_deep_lite":
        model = UNet3DFADC(**model_kwargs, fadc_placement='deep_lite', **fadc_kwargs).to(device)
    else:
        if args.deep_supervision:
            raise ValueError("--deep_supervision is only wired for FADC variants. "
                             "Add DS heads to UNet3D first if you want it on the baseline.")
        model = UNet3D(**model_kwargs).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Parameters: {total_params:,}"
          f"{' | DEEP SUPERVISION ON (aux heads at dec2/dec3/dec4)' if args.deep_supervision else ''}")

    # ── Loss, Optimizer, Scheduler ────────────
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

    # ── Metrics ───────────────────────────────
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred   = AsDiscrete(argmax=True, to_onehot=2)
    post_label  = AsDiscrete(to_onehot=2)

    # ── Resume from checkpoint ─────────────────
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

    # ── Training ──────────────────────────────
    print(f"\nStarting training: {epochs} epochs | LR: {lr} | Batch: {batch_size}")
    print("=" * 70)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss  = 0.0
        epoch_dice  = 0.0
        epoch_ce    = 0.0
        num_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{epochs}",
                    leave=False, ncols=110, unit="batch", file=sys.stdout)

        for batch in pbar:
            if isinstance(batch["image"], list):
                # PersistentDataset returns numpy arrays after disk reload — convert first
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
                if isinstance(preds, tuple):
                    total_loss, dice_loss, ce_loss = deep_supervision_loss(preds, labels, criterion)
                else:
                    total_loss, dice_loss, ce_loss = criterion(preds, labels)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += total_loss.item()
            epoch_dice  += dice_loss.item()
            epoch_ce    += ce_loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "dice": f"{dice_loss.item():.4f}",
                "ce":   f"{ce_loss.item():.4f}",
            }, refresh=False)

        pbar.close()
        scheduler.step()

        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        avg_ce   = epoch_ce   / num_batches
        elapsed  = time.time() - t0
        lr_now   = scheduler.get_last_lr()[0]

        mins, secs = divmod(int(elapsed), 60)
        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | Dice loss: {avg_dice:.4f} | CE loss: {avg_ce:.4f} | "
              f"LR: {lr_now:.2e} | Time: {mins}m {secs}s")

        log_entry = {
            "epoch":     epoch + 1,
            "loss":      avg_loss,
            "dice_loss": avg_dice,
            "ce_loss":   avg_ce,
            "lr":        lr_now,
            "time_s":    elapsed,
        }

        # ── Validation ────────────────────────
        if (epoch + 1) % val_every == 0:
            val_dice, val_iou, val_sens = validate(
                model, val_loader, dice_metric,
                post_pred, post_label, patch_size, device
            )
            marker = " <-- BEST" if val_dice > best_dice else ""
            print(f"  Val  Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | Sensitivity: {val_sens:.4f}"
                  f"  (best so far: {best_dice:.4f}){marker}")

            log_entry["val_dice"]        = val_dice
            log_entry["val_iou"]         = val_iou
            log_entry["val_sensitivity"] = val_sens

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_dice": best_dice,
                    "config":    cfg,
                }, output_dir / "best_model.pth")
                print(f"  {'*' * 60}")
                print(f"  *** NEW BEST  Epoch {epoch+1:03d} | "
                      f"Dice {best_dice:.4f} | IoU {val_iou:.4f} | Sensitivity {val_sens:.4f} ***")
                print(f"  *** Saved -> {output_dir}/best_model.pth")
                print(f"  {'*' * 60}")

        train_log.append(log_entry)

        # Save latest checkpoint every 5 epochs for resuming
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_dice": best_dice,
                "config":    cfg,
            }, output_dir / "latest_checkpoint.pth")

    # ── Save training log ─────────────────────
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    # Stamp seed + model in a separate meta.json (keeps train_log.json shape stable).
    with open(output_dir / "meta.json", "w") as f:
        json.dump({"seed": args.seed, "model": args.model,
                   "best_dice": best_dice, "epochs": epochs}, f, indent=2)

    print(f"\nTraining complete. Best Val Dice: {best_dice:.4f}")
    print(f"Outputs saved to: {output_dir}")


# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config, args)
    train(cfg, args)
