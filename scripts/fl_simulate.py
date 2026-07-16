"""
CLI entrypoint for federated training.

This file owns the wiring (argparse + model factory + loss + partition build).
The orchestrator (training/train_federated.py) is architecture-agnostic — to
plug in a different model later, only this file changes.

Example:
  python scripts/fl_simulate.py \
      --model unet3d \
      --algorithm fedavg \
      --partition natural \
      --preprocessed_cache_dir /kaggle/input/.../mama-mia-cache \
      --rounds 30 --local_epochs 2 --client_fraction 1.0 \
      --seed 42 --output_dir outputs/fl/unet3d_fedavg_natural_s42
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).parent.parent))
from data.mama_mia_dataset import DATA_ROOT, discover_cases_from_cache
from models.unet_3d import UNet3D
from models.unet_3d_fadc import UNet3DFADC
from training.fl_partitions import partition_factory, partition_summary
from training.train_federated import run_fl
from training.losses import DiceCELoss


_FADC_PLACEMENT = {
    "unet3d_fadc":             "full",
    "unet3d_fadc_encoder":     "encoder",
    "unet3d_fadc_bottleneck":  "bottleneck",
    "unet3d_fadc_mid":         "mid",
    "unet3d_fadc_deep":        "deep",
    "unet3d_fadc_deep_lite":   "deep_lite",
}


def make_model_fn(model_name: str, cfg: dict):
    """Return a no-arg builder that instantiates a fresh model on CPU."""
    model_kwargs = dict(
        in_channels  = cfg["model"]["in_channels"],
        out_channels = cfg["model"]["out_channels"],
        base_filters = cfg["model"]["base_filters"],
    )

    if model_name == "unet3d":
        return lambda: UNet3D(**model_kwargs)
    if model_name in _FADC_PLACEMENT:
        placement = _FADC_PLACEMENT[model_name]
        # deep_supervision is intentionally False — FL v1 does not wire DS.
        return lambda: UNet3DFADC(**model_kwargs,
                                  fadc_placement=placement,
                                  deep_supervision=False)
    raise ValueError(f"Unknown model: {model_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    project_root = str(Path(__file__).parent.parent)

    parser.add_argument("--config", type=str,
                        default=os.path.join(project_root, "configs", "config.yaml"))
    parser.add_argument("--model", type=str, default="unet3d",
                        choices=["unet3d"] + list(_FADC_PLACEMENT.keys()),
                        help="Model architecture. Swap to the final winner after seed sweep.")

    # FL config
    parser.add_argument("--algorithm",   type=str, default="fedavg",
                        choices=["fedavg", "fedbn"])
    parser.add_argument("--partition",   type=str, default="natural",
                        choices=["natural", "iid"])
    parser.add_argument("--n_clients",   type=int, default=4,
                        help="Only used when --partition iid. Natural is always 4.")
    parser.add_argument("--rounds",      type=int, default=30)
    parser.add_argument("--local_epochs",   type=int, default=2)
    parser.add_argument("--client_fraction", type=float, default=1.0,
                        help="Fraction of clients sampled per round, in (0, 1].")

    # Optimization
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--patch_size",  type=int, nargs=3, default=None,
                        metavar=("X", "Y", "Z"))

    # Data
    parser.add_argument("--data_root",   type=str, default=DATA_ROOT)
    parser.add_argument("--preprocessed_cache_dir", type=str, required=True,
                        help="Path to .npz cache root (must contain train/ and val/ subdirs)")

    # Bookkeeping
    parser.add_argument("--seed",        type=int, default=None)
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--save_every",  type=int, default=5,
                        help="Save latest_round.pth every N rounds")
    parser.add_argument("--smoke_test",  action="store_true",
                        help="2 rounds x 1 local epoch x 4 cases per client. Skips real validation cost.")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if args.patch_size is not None:
        cfg["data"]["patch_size"] = args.patch_size
    patch_size = tuple(cfg["data"]["patch_size"])

    train_cache_dir = os.path.join(args.preprocessed_cache_dir, "train")
    val_cache_dir   = os.path.join(args.preprocessed_cache_dir, "val")

    split_csv = os.path.join(args.data_root, "train_test_splits.csv")
    if not os.path.exists(split_csv):
        split_csv = None
        print("[FL] Warning: train_test_splits.csv not found — using all cached cases")

    train_cases = discover_cases_from_cache(train_cache_dir, split_csv, split="train")
    val_cases   = discover_cases_from_cache(val_cache_dir,   split_csv, split="test")
    print(f"[FL] Train cases (all): {len(train_cases)}  | Val cases (global): {len(val_cases)}")

    if args.smoke_test:
        # 4 train cases/client, 4 val total — for CLI sanity only; the real
        # smoke test is tests/test_fl_loop.py (pure synthetic, no I/O).
        print("[FL] SMOKE TEST MODE — 2 rounds, 1 local epoch, 4 cases/client")
        args.rounds = 2
        args.local_epochs = 1
        val_cases = val_cases[:4]

    partition_fn = partition_factory(args.partition, args.n_clients, args.seed or 0)
    partition    = partition_fn(train_cases)
    if args.smoke_test:
        partition = {cid: cases[:4] for cid, cases in partition.items()}

    print(f"[FL] Partition ({args.partition}):")
    print(partition_summary(partition))

    model_fn = make_model_fn(args.model, cfg)
    loss_fn  = DiceCELoss(
        dice_weight=cfg["training"]["dice_weight"],
        ce_weight=cfg["training"]["ce_weight"],
    )

    run_fl(
        model_fn         = model_fn,
        loss_fn          = loss_fn,
        partition        = partition,
        val_cases        = val_cases,
        train_cache_dir  = train_cache_dir,
        val_cache_dir    = val_cache_dir,
        algorithm        = args.algorithm,
        rounds           = args.rounds,
        local_epochs     = args.local_epochs,
        lr               = args.lr,
        client_fraction  = args.client_fraction,
        batch_size       = args.batch_size,
        num_workers      = args.num_workers,
        patch_size       = patch_size,
        output_dir       = args.output_dir,
        seed             = args.seed,
        save_every       = args.save_every,
    )


if __name__ == "__main__":
    main()
