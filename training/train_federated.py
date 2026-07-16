"""
Federated training orchestrator (single-node simulation).

All clients run sequentially in one Python process — fits Kaggle's single-GPU
constraint while preserving the standard FL algorithmic structure (round /
local epoch / aggregate / global eval).

Architecture-agnostic:
  - model_fn: () -> nn.Module      (built per-run from the CLI)
  - loss_fn: any nn.Module returning (total, dice, ce)
  - partition: pre-computed {client_id: list[case_dict]}

This file does NOT know about UNet3D, UNet3DFADC, DiceCELoss, or argparse —
those wire-ups live in scripts/fl_simulate.py.
"""
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from monai.data import DataLoader as MonaiLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

sys.path.append(str(Path(__file__).parent.parent))
from data.mama_mia_dataset import (
    PreprocessedDataset,
    _safe_collate,
    _seed_worker,
)
from training.fl_algorithms import (
    fedavg_aggregate,
    fedbn_aggregate,
    get_bn_state_dict_keys,
    split_bn_state,
)


# ─────────────────────────────────────────────
# Loader builders
# ─────────────────────────────────────────────

def _make_loader(cache_dir, cases, *, is_train, batch_size, num_workers,
                 patch_size, seed):
    """Wrap a case list in a DataLoader using the existing PreprocessedDataset.

    Train loaders shuffle and apply random crops + augmentations; val loaders
    iterate the full volume in deterministic order so per-case metrics line up
    with the val_cases list.
    """
    ds = PreprocessedDataset(cache_dir, cases, is_train=is_train,
                             patch_size=patch_size)
    loader_kwargs = dict(collate_fn=_safe_collate, pin_memory=True,
                         persistent_workers=num_workers > 0)
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
        loader_kwargs["worker_init_fn"] = _seed_worker
        loader_kwargs["generator"] = gen

    return MonaiLoader(
        ds,
        batch_size=batch_size if is_train else 1,
        shuffle=is_train,
        num_workers=num_workers,
        **loader_kwargs,
    )


def build_client_train_loaders(partition, cache_dir, batch_size, num_workers,
                               patch_size, seed):
    """Per-client train loaders. The cache_dir here is the .../train subdir."""
    return {
        cid: _make_loader(cache_dir, cases, is_train=True,
                          batch_size=batch_size, num_workers=num_workers,
                          patch_size=patch_size, seed=seed)
        for cid, cases in partition.items()
        if len(cases) > 0
    }


def build_global_val_loader(val_cases, cache_dir, num_workers, patch_size, seed):
    """One val loader over all val cases. batch_size=1, shuffle=False so the
    iteration order matches val_cases for per-case bookkeeping.
    """
    return _make_loader(cache_dir, val_cases, is_train=False,
                        batch_size=1, num_workers=num_workers,
                        patch_size=patch_size, seed=seed)


# ─────────────────────────────────────────────
# Validation — single pass, per-case bookkeeping
# ─────────────────────────────────────────────

def validate_per_case(model, val_loader, val_cases, patch_size, device):
    """Sliding-window inference on every val case; record per-case Dice/IoU/Sens.

    The per-client breakdown is computed by grouping these per-case records by
    client_id, so global and per-client validation share a single inference pass.
    """
    model.eval()
    post_pred  = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)
    dice_metric = DiceMetric(include_background=False, reduction="none")

    per_case = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
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

            preds_bin  = [post_pred(p)          for p in decollate_batch(preds)]
            labels_bin = [post_label(l.long())  for l in decollate_batch(labels)]

            dice_metric.reset()
            dice_metric(y_pred=preds_bin, y=labels_bin)
            d = float(dice_metric.aggregate().item())

            pf = preds_bin[0][1].float()
            lf = labels_bin[0][1].float()
            tp = (pf * lf).sum().item()
            fp = (pf * (1 - lf)).sum().item()
            fn = ((1 - pf) * lf).sum().item()
            iou  = tp / (tp + fp + fn + 1e-6)
            sens = tp / (tp + fn + 1e-6)

            meta = val_cases[batch_idx]
            per_case.append({
                "patient_id": meta["patient_id"],
                "collection": meta["collection"],
                "client_id":  meta["client_id"],
                "dice": d, "iou": iou, "sens": sens,
            })
    return per_case


def _aggregate(records):
    if not records:
        return {"dice": 0.0, "iou": 0.0, "sens": 0.0, "n": 0}
    return {
        "dice": float(np.mean([r["dice"] for r in records])),
        "iou":  float(np.mean([r["iou"]  for r in records])),
        "sens": float(np.mean([r["sens"] for r in records])),
        "n":    len(records),
    }


def summarize(per_case):
    """Return {'global': {...}, 'per_client': {client_id: {...}}}."""
    by_client = {}
    for r in per_case:
        by_client.setdefault(r["client_id"], []).append(r)
    return {
        "global":     _aggregate(per_case),
        "per_client": {cid: _aggregate(rs) for cid, rs in by_client.items()},
    }


# ─────────────────────────────────────────────
# Local training step (one client, E epochs)
# ─────────────────────────────────────────────

def local_train(model, loader, loss_fn, local_epochs, lr, device, scaler):
    """Train `model` in place for `local_epochs` epochs over `loader`.

    Returns the dict of average losses across the local epochs (for logging).
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    sum_loss = sum_dice = sum_ce = 0.0
    num_batches = 0

    for _ in range(local_epochs):
        for batch in loader:
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
                # FL v1 does not wire deep supervision; the model's forward()
                # returns a tuple only when model.training AND ds is enabled.
                # If a DS-trained model is ever passed in, drop the aux heads.
                if isinstance(preds, tuple):
                    preds = preds[0]
                total_loss, dice_loss, ce_loss = loss_fn(preds, labels)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            sum_loss += total_loss.item()
            sum_dice += dice_loss.item()
            sum_ce   += ce_loss.item()
            num_batches += 1

    return {
        "loss":      sum_loss / max(num_batches, 1),
        "dice_loss": sum_dice / max(num_batches, 1),
        "ce_loss":   sum_ce   / max(num_batches, 1),
        "batches":   num_batches,
    }


# ─────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────

def _seed_everything(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sample_clients(client_ids, fraction, round_rng):
    """Random sample of at least one client per round."""
    n = max(1, int(round(len(client_ids) * fraction)))
    return round_rng.sample(list(client_ids), k=n)


def run_fl(
    *,
    model_fn,
    loss_fn,
    partition,           # {client_id: [case_dict, ...]} — train partition
    val_cases,           # list of dicts — global val (in deterministic order)
    train_cache_dir,     # str — path to preprocessed .npz train cache
    val_cache_dir,       # str — path to preprocessed .npz val cache
    algorithm,           # "fedavg" | "fedbn"
    rounds,
    local_epochs,
    lr,
    client_fraction=1.0,
    batch_size=2,
    num_workers=2,
    patch_size=(128, 128, 64),
    output_dir="outputs/fl",
    seed=None,
    save_every=5,
):
    """Run federated training. Returns the path to the best-checkpoint file."""
    assert algorithm in ("fedavg", "fedbn"), f"Unknown algorithm: {algorithm}"
    _seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[FL] Device: {device}")
    print(f"[FL] Algorithm: {algorithm} | Rounds: {rounds} | Local epochs: {local_epochs} | "
          f"LR: {lr} | Client fraction: {client_fraction} | Seed: {seed}")

    # ── Data ─────────────────────────────────
    client_loaders = build_client_train_loaders(
        partition, train_cache_dir,
        batch_size=batch_size, num_workers=num_workers,
        patch_size=patch_size, seed=seed,
    )
    val_loader = build_global_val_loader(
        val_cases, val_cache_dir,
        num_workers=num_workers, patch_size=patch_size, seed=seed,
    )
    print(f"[FL] Active clients: {sorted(client_loaders.keys())}")
    print(f"[FL] Global val cases: {len(val_cases)}")

    client_ids   = sorted(client_loaders.keys())
    n_samples    = {cid: len(partition[cid]) for cid in client_ids}

    # ── Models ───────────────────────────────
    global_model = model_fn().to(device)
    local_model  = model_fn().to(device)
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"[FL] Model parameters: {total_params:,}")

    # ── BN bookkeeping (FedBN only) ──────────
    bn_keys = get_bn_state_dict_keys(global_model) if algorithm == "fedbn" else set()
    # Persist each client's BN state between rounds (CPU to save VRAM).
    bn_state_per_client = {
        cid: split_bn_state({k: v.detach().cpu().clone()
                             for k, v in global_model.state_dict().items()},
                            bn_keys)[1]
        for cid in client_ids
    } if algorithm == "fedbn" else {}

    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    # ── Round loop ───────────────────────────
    # -inf so even a Dice=0 first round saves best_model.pth — guarantees a
    # checkpoint exists after every run, including smoke tests on synthetic data.
    best_dice = float("-inf")
    fl_log = []
    round_rng = random.Random(seed if seed is not None else 0)

    for round_idx in range(1, rounds + 1):
        t_round = time.time()
        participants = _sample_clients(client_ids, client_fraction, round_rng)

        client_states  = []
        client_weights = []
        client_losses  = {}

        prior_global_state = {k: v.detach().clone()
                              for k, v in global_model.state_dict().items()}

        for cid in participants:
            # Reset local_model to current global params
            local_model.load_state_dict(prior_global_state)

            # FedBN: overlay this client's persisted BN
            if algorithm == "fedbn":
                state = local_model.state_dict()
                for k, v in bn_state_per_client[cid].items():
                    state[k] = v.to(device)
                local_model.load_state_dict(state)

            losses = local_train(local_model, client_loaders[cid], loss_fn,
                                 local_epochs, lr, device, scaler)
            client_losses[cid] = losses

            # Snapshot state for aggregation (CPU copies to free GPU between clients)
            snapshot = {k: v.detach().cpu().clone()
                        for k, v in local_model.state_dict().items()}
            client_states.append(snapshot)
            client_weights.append(n_samples[cid])

            # FedBN: persist this client's new BN state
            if algorithm == "fedbn":
                _, bn_only = split_bn_state(snapshot, bn_keys)
                bn_state_per_client[cid] = bn_only

        # ── Aggregate ────────────────────────
        if algorithm == "fedavg":
            agg = fedavg_aggregate(client_states, client_weights)
        else:  # fedbn
            prior_cpu = {k: v.detach().cpu().clone()
                         for k, v in prior_global_state.items()}
            agg = fedbn_aggregate(client_states, client_weights, bn_keys, prior_cpu)

        # Move aggregated state back onto device, install in global_model
        global_model.load_state_dict({k: v.to(device) for k, v in agg.items()})

        # ── Validate ─────────────────────────
        t_val = time.time()
        per_case = validate_per_case(global_model, val_loader, val_cases,
                                     patch_size, device)
        eval_summary = summarize(per_case)
        val_time = time.time() - t_val

        round_time = time.time() - t_round
        glob = eval_summary["global"]
        print(f"[Round {round_idx:03d}/{rounds}] "
              f"clients={participants} | "
              f"Global Dice {glob['dice']:.4f} | IoU {glob['iou']:.4f} | "
              f"Sens {glob['sens']:.4f} | "
              f"val {val_time:.0f}s | total {round_time:.0f}s")
        for cid in sorted(eval_summary["per_client"].keys()):
            pc = eval_summary["per_client"][cid]
            print(f"           client {cid}: Dice {pc['dice']:.4f} | "
                  f"IoU {pc['iou']:.4f} | Sens {pc['sens']:.4f} | n={pc['n']}")

        # ── Log ──────────────────────────────
        round_log = {
            "round":         round_idx,
            "participants":  participants,
            "client_losses": client_losses,
            "global":        glob,
            "per_client":    eval_summary["per_client"],
            "val_time_s":    val_time,
            "round_time_s":  round_time,
        }
        fl_log.append(round_log)

        with open(output_dir / "fl_log.json", "w") as f:
            json.dump(fl_log, f, indent=2)

        # ── Save best & latest ───────────────
        if glob["dice"] > best_dice:
            best_dice = glob["dice"]
            torch.save({
                "round":      round_idx,
                "model":      global_model.state_dict(),
                "bn_per_client": bn_state_per_client if algorithm == "fedbn" else {},
                "best_dice":  best_dice,
                "algorithm":  algorithm,
            }, output_dir / "best_model.pth")
            print(f"           *** NEW BEST  Dice {best_dice:.4f} — saved best_model.pth")

        if round_idx % save_every == 0:
            torch.save({
                "round":      round_idx,
                "model":      global_model.state_dict(),
                "bn_per_client": bn_state_per_client if algorithm == "fedbn" else {},
                "best_dice":  best_dice,
                "algorithm":  algorithm,
            }, output_dir / "latest_round.pth")

    # ── Final meta ───────────────────────────
    with open(output_dir / "meta.json", "w") as f:
        json.dump({
            "seed":            seed,
            "algorithm":       algorithm,
            "rounds":          rounds,
            "local_epochs":    local_epochs,
            "lr":              lr,
            "client_fraction": client_fraction,
            "n_clients":       len(client_ids),
            "n_samples":       n_samples,
            "best_dice":       best_dice,
        }, f, indent=2)

    print(f"\n[FL] Done. Best global Dice: {best_dice:.4f}")
    print(f"[FL] Outputs: {output_dir}")
    return str(output_dir / "best_model.pth")
