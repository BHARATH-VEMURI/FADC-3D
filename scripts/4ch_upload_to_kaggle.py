"""
Upload the 4ch preprocessed cache to Kaggle as a private dataset.

Uses the Kaggle API key at C:\\Users\\bhara\\.kaggle\\kaggle.json.

Creates dataset under user 'vbk1999' with slug '4ch-mama-mia-cache'.
Resulting Kaggle path will be:
  /kaggle/input/datasets/vbk1999/4ch-mama-mia-cache/

The upload is a single zip per directory (train/ and val/) — this is the
fastest path for Kaggle's dataset API; per-file uploads are throttled.

Usage:
    # First-time create:
    python scripts/4ch_upload_to_kaggle.py --create \\
        --cache_dir "C:/Users/bhara/Desktop/4ch_mama_mia_cache"

    # Subsequent updates (after fixing some cases):
    python scripts/4ch_upload_to_kaggle.py --update \\
        --cache_dir "C:/Users/bhara/Desktop/4ch_mama_mia_cache" \\
        --message "Fixed corrupted ISPY1 cases"

Pre-flight checks:
  - kaggle CLI installed (`pip install kaggle`)
  - kaggle.json at C:\\Users\\bhara\\.kaggle\\kaggle.json with vbk1999 credentials
  - 4ch cache has train/ and val/ subdirs
  - Enough disk for temporary zip files (~42 GB free)

Upload time estimate: 4-12 hours depending on home upload speed.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


KAGGLE_USER = "vbk1999"
DATASET_SLUG = "4ch-mama-mia-cache"
DATASET_TITLE = "MAMA-MIA 4-channel preprocessed cache (pre, post1, post2, sub)"
DATASET_FULL = f"{KAGGLE_USER}/{DATASET_SLUG}"

KAGGLE_JSON = Path(r"C:\Users\bhara\.kaggle\kaggle.json")


def _check_kaggle_cli():
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: kaggle CLI not found. Install with: pip install kaggle")
        sys.exit(1)


def _check_kaggle_json():
    if not KAGGLE_JSON.exists():
        print(f"ERROR: {KAGGLE_JSON} not found.")
        sys.exit(1)
    with open(KAGGLE_JSON) as f:
        cfg = json.load(f)
    if cfg.get("username") != KAGGLE_USER:
        print(f"WARN: kaggle.json username is '{cfg.get('username')}', expected '{KAGGLE_USER}'.")
        print(f"      Make sure you're using the API key from the vbk1999 account.")
        resp = input("Continue anyway? [y/N]: ").strip().lower()
        if resp != "y":
            sys.exit(1)


def _check_cache(cache_dir: Path):
    train_dir = cache_dir / "train"
    val_dir   = cache_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        print(f"ERROR: cache_dir must have train/ and val/ subdirs at {cache_dir}")
        sys.exit(1)
    n_train = len(list(train_dir.glob("*.npz")))
    n_val   = len(list(val_dir.glob("*.npz")))
    if n_train < 1000 or n_val < 200:
        print(f"WARN: low file counts (train={n_train}, val={n_val}).")
        print(f"      Expected ~1200 train + ~306 val. Run preprocessing first.")
        resp = input("Continue anyway? [y/N]: ").strip().lower()
        if resp != "y":
            sys.exit(1)
    return n_train, n_val


def _prepare_upload_dir(cache_dir: Path, staging_dir: Path):
    """Copy/move cache files into a flat staging dir for kaggle upload.
    Kaggle datasets accept folders directly, so we just point at cache_dir.
    But Kaggle's metadata file (dataset-metadata.json) needs to be at the
    root of the upload, so we stage there.

    We create symlinks (or copies on Windows fallback) to avoid duplicating 42GB.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)

    # dataset-metadata.json — required by kaggle CLI
    metadata = {
        "title": DATASET_TITLE,
        "id": DATASET_FULL,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(staging_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Use the cache_dir directly. Kaggle API accepts the parent dir containing
    # dataset-metadata.json + the actual data files/subdirs.
    # We need the metadata file IN cache_dir, so copy it there.
    shutil.copy(staging_dir / "dataset-metadata.json", cache_dir / "dataset-metadata.json")
    print(f"Wrote dataset-metadata.json to {cache_dir}")


def cmd_create(args):
    cache_dir = Path(args.cache_dir)
    _check_kaggle_cli()
    _check_kaggle_json()
    n_train, n_val = _check_cache(cache_dir)
    print(f"Cache OK: {n_train} train, {n_val} val .npz files")

    # Write dataset-metadata.json into cache_dir
    staging_dir = Path(args.staging_dir) if args.staging_dir else cache_dir / ".staging"
    _prepare_upload_dir(cache_dir, staging_dir)

    print(f"\nReady to create Kaggle dataset: {DATASET_FULL}")
    print(f"Visibility: PRIVATE")
    print(f"Estimated upload size: ~40-42 GB")
    print(f"Estimated upload time: 4-12 hours")
    print()
    resp = input("Proceed with create? [y/N]: ").strip().lower()
    if resp != "y":
        print("Aborted.")
        sys.exit(0)

    cmd = [
        "kaggle", "datasets", "create",
        "-p", str(cache_dir),
        "--dir-mode", "zip",   # bundle train/ and val/ folders into zips
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def cmd_update(args):
    cache_dir = Path(args.cache_dir)
    _check_kaggle_cli()
    _check_kaggle_json()
    n_train, n_val = _check_cache(cache_dir)
    print(f"Cache OK: {n_train} train, {n_val} val .npz files")

    # Ensure metadata is present (may have been updated)
    staging_dir = Path(args.staging_dir) if args.staging_dir else cache_dir / ".staging"
    _prepare_upload_dir(cache_dir, staging_dir)

    print(f"\nReady to update Kaggle dataset: {DATASET_FULL}")
    print(f"Message: {args.message}")
    resp = input("Proceed with update? [y/N]: ").strip().lower()
    if resp != "y":
        print("Aborted.")
        sys.exit(0)

    cmd = [
        "kaggle", "datasets", "version",
        "-p", str(cache_dir),
        "-m", args.message,
        "--dir-mode", "zip",
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="First-time dataset creation")
    p_create.add_argument("--cache_dir", type=str,
                          default=r"C:\Users\bhara\Desktop\4ch_mama_mia_cache")
    p_create.add_argument("--staging_dir", type=str, default=None)
    p_create.set_defaults(func=cmd_create)

    p_update = sub.add_parser("update", help="Push a new version of existing dataset")
    p_update.add_argument("--cache_dir", type=str,
                          default=r"C:\Users\bhara\Desktop\4ch_mama_mia_cache")
    p_update.add_argument("--staging_dir", type=str, default=None)
    p_update.add_argument("-m", "--message", type=str, required=True,
                          help="Update message (required by kaggle CLI)")
    p_update.set_defaults(func=cmd_update)

    # Convenience: support legacy --create / --update flags as well
    # via subparser dispatch (cmd="create" or cmd="update")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
