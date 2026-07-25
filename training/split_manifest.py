"""Deterministic patient-level split manifest for the corrected FADC3D
70/10/20 experiment.

Contract:
- Patient-level only (never slice/patch). All data for one patient stays in
  one split.
- Stratified by collection (DUKE / ISPY1 / ISPY2 / NACT) so ratios are
  preserved WITHIN each collection.
- Deterministic under a given seed. Reproducing the same manifest twice from
  the same on-disk cache and seed yields byte-identical CSV rows.
- The manifest carries the absolute .npz path per patient so downstream
  loaders don't have to guess which of the old `train/` or `val/` subfolders
  a patient came from.

Persistence:
- CSV columns: patient_id, collection, split, relative_npz_path.
  `relative_npz_path` is written relative to `cache_root` (the parent that
  contains the old `train/` and `val/` subfolders) so the manifest is
  portable across machines whose mount points differ.
- Metadata JSON records seed, ratios, per-collection/per-split counts, and
  the SHA256 of the CSV file.

This module never copies or moves .npz files. It only enumerates and
categorizes them.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

COLLECTIONS = ("DUKE", "ISPY1", "ISPY2", "NACT")

# ─────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────

def enumerate_patients(cache_root: str,
                       subdirs: Iterable[str] = ("train", "val")) -> list[dict]:
    """Walk the on-disk cache, return one dict per .npz file.

    `cache_root` is the parent containing the old `train/` and `val/`
    subfolders. `subdirs` lets a test override the enumeration to a
    non-standard layout.

    Fields per case:
      - patient_id       : lower-cased .npz stem (e.g. 'duke_001')
      - collection       : uppercased leading token (e.g. 'DUKE')
      - npz_path         : absolute path
      - relative_npz_path: path relative to cache_root (portable)
      - source_subdir    : which enumerated subdir contained the file

    Raises RuntimeError if a patient_id appears in more than one subdir
    (the old train/val cache should never have duplicates — an overlap
    means the cache is stale or the layout was hand-edited).
    """
    cache_root_p = Path(cache_root)
    if not cache_root_p.exists():
        raise FileNotFoundError(f"cache_root does not exist: {cache_root}")

    seen: dict[str, str] = {}
    out: list[dict] = []
    for sub in subdirs:
        d = cache_root_p / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("*.npz")):
            pid = p.stem.lower()
            coll = pid.split("_")[0].upper()
            if coll not in COLLECTIONS:
                # ignore stray files; the old cache has always been strict
                continue
            if pid in seen:
                raise RuntimeError(
                    f"Duplicate patient_id {pid!r} across cache subdirs "
                    f"({seen[pid]!r} and {sub!r}). Refusing to build a manifest."
                )
            seen[pid] = sub
            out.append({
                "patient_id":        pid,
                "collection":        coll,
                "npz_path":          str(p),
                "relative_npz_path": os.path.relpath(str(p), str(cache_root_p)).replace("\\", "/"),
                "source_subdir":     sub,
            })
    return out


# ─────────────────────────────────────────────────────────────────────────
# Split assignment
# ─────────────────────────────────────────────────────────────────────────

def assign_splits(patients: list[dict],
                  seed: int,
                  ratios: tuple[float, float, float] = (0.70, 0.10, 0.20)
                  ) -> list[dict]:
    """Assign each patient to 'train' / 'val' / 'test'.

    Stratified by collection: within each collection the ratios are applied
    to that collection's patients only, so overall ratios only differ from
    the target by rounding.

    Deterministic under `seed`. Uses `random.Random(seed).shuffle` per
    collection (never touches the global RNG). Rounding rule:
      n_train = round(n * r_train)
      n_val   = round(n * r_val)
      n_test  = n - n_train - n_val
    which pushes ALL rounding drift into the test bucket (kept the largest,
    so a one-patient rounding wobble doesn't starve val).
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}: {ratios}")
    if any(r < 0.0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios}")

    r_train, r_val, r_test = ratios
    rng = random.Random(seed)

    by_coll: dict[str, list[dict]] = defaultdict(list)
    for c in patients:
        by_coll[c["collection"]].append(c)

    assigned: list[dict] = []
    for coll in sorted(by_coll):
        cases = sorted(by_coll[coll], key=lambda c: c["patient_id"])
        # Per-collection deterministic shuffle so different seeds actually
        # produce different splits AND the same seed reproduces byte-for-byte.
        local_rng = random.Random(f"{seed}::{coll}")
        local_rng.shuffle(cases)
        n = len(cases)
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        # Clamp so no split is negative when n is tiny.
        n_train = max(0, min(n, n_train))
        n_val = max(0, min(n - n_train, n_val))
        n_test = n - n_train - n_val
        for c in cases[:n_train]:
            c = dict(c); c["split"] = "train"; assigned.append(c)
        for c in cases[n_train : n_train + n_val]:
            c = dict(c); c["split"] = "val"; assigned.append(c)
        for c in cases[n_train + n_val :]:
            c = dict(c); c["split"] = "test"; assigned.append(c)
    # Return sorted by patient_id so the CSV is deterministic across runs.
    assigned.sort(key=lambda c: c["patient_id"])
    return assigned


# ─────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────

_CSV_COLUMNS = ("patient_id", "collection", "split", "relative_npz_path")


def write_manifest(assigned: list[dict], csv_path: str) -> None:
    """Write the CSV manifest with LF line endings so the SHA256 checksum
    is portable across Windows / Linux."""
    csv_path = str(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    # Use newline='' so csv writes '\n' verbatim across platforms.
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS,
                           extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for c in assigned:
            w.writerow({k: c[k] for k in _CSV_COLUMNS})


def manifest_sha256(csv_path: str) -> str:
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_metadata(assigned: list[dict],
                   seed: int,
                   ratios: tuple[float, float, float],
                   csv_path: str,
                   meta_path: str) -> dict:
    """Write the metadata JSON companion. Returns the metadata dict."""
    meta_path = str(meta_path)
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)

    by_split = Counter(c["split"] for c in assigned)
    per_collection_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c in assigned:
        per_collection_split[c["collection"]][c["split"]] += 1

    meta = {
        "seed":                 int(seed),
        "ratios":               {"train": float(ratios[0]),
                                 "val":   float(ratios[1]),
                                 "test":  float(ratios[2])},
        "n_patients_total":     int(len(assigned)),
        "n_per_split":          {k: int(by_split[k]) for k in ("train", "val", "test")},
        "n_per_collection_split": {coll: {sp: int(per_collection_split[coll].get(sp, 0))
                                          for sp in ("train", "val", "test")}
                                   for coll in COLLECTIONS},
        "csv_path":             os.path.abspath(csv_path),
        "csv_sha256":           manifest_sha256(csv_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return meta


# ─────────────────────────────────────────────────────────────────────────
# Load-side
# ─────────────────────────────────────────────────────────────────────────

CLIENT_MAP = {"DUKE": 0, "ISPY1": 1, "ISPY2": 2, "NACT": 3}


def load_manifest(csv_path: str,
                  split: str,
                  cache_root: str,
                  require_exists: bool = True) -> list[dict]:
    """Read a manifest CSV, return the case list for one split.

    Each case carries `patient_id`, `collection`, `client_id`, and an
    absolute `npz_path` resolved against `cache_root` (the parent of the
    old train/ and val/ subfolders).

    When `require_exists=True` (the default) asserts every referenced .npz
    exists — cheap and prevents silent skips at training time.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train|val|test, got {split!r}")
    cache_root_p = Path(cache_root)
    if not cache_root_p.exists():
        raise FileNotFoundError(f"cache_root does not exist: {cache_root}")

    cases: list[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["split"] != split:
                continue
            rel = row["relative_npz_path"].replace("\\", "/")
            npz_abs = (cache_root_p / rel).resolve()
            if require_exists and not npz_abs.exists():
                raise FileNotFoundError(
                    f"Manifest references missing file: {npz_abs} "
                    f"(patient_id={row['patient_id']}, split={split})"
                )
            coll = row["collection"]
            cases.append({
                "patient_id": row["patient_id"],
                "collection": coll,
                "client_id":  CLIENT_MAP.get(coll, -1),
                "npz_path":   str(npz_abs),
            })
    # Sorted by patient_id for determinism.
    cases.sort(key=lambda c: c["patient_id"])
    return cases


def verify_manifest_partitions(csv_path: str) -> dict:
    """Cross-check: every patient exactly once, no duplicates, splits pairwise
    disjoint, union = full patient set. Returns a small summary dict.
    Raises AssertionError on any violation.
    """
    seen: dict[str, str] = {}
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid = row["patient_id"]
            sp = row["split"]
            if pid in seen:
                raise AssertionError(
                    f"duplicate patient_id in manifest: {pid} "
                    f"(splits {seen[pid]!r} and {sp!r})"
                )
            if sp not in by_split:
                raise AssertionError(f"unknown split {sp!r} for patient {pid}")
            seen[pid] = sp
            by_split[sp].add(pid)
    # Pairwise disjointness is guaranteed by the seen-dict above.
    total = sum(len(s) for s in by_split.values())
    if total != len(seen):
        raise AssertionError(f"split-sum {total} != unique-patients {len(seen)}")
    return {
        "n_train": len(by_split["train"]),
        "n_val":   len(by_split["val"]),
        "n_test":  len(by_split["test"]),
        "n_total": len(seen),
    }


# ─────────────────────────────────────────────────────────────────────────
# End-to-end convenience
# ─────────────────────────────────────────────────────────────────────────

def generate_and_write(cache_root: str,
                       csv_path: str,
                       meta_path: str,
                       seed: int = 42,
                       ratios: tuple[float, float, float] = (0.70, 0.10, 0.20),
                       ) -> dict:
    """One-shot: enumerate → assign → write CSV + JSON, return metadata."""
    patients = enumerate_patients(cache_root)
    assigned = assign_splits(patients, seed=seed, ratios=ratios)
    write_manifest(assigned, csv_path)
    meta = write_metadata(assigned, seed=seed, ratios=ratios,
                          csv_path=csv_path, meta_path=meta_path)
    return meta
