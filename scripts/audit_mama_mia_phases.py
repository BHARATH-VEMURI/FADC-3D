"""
Audit which DCE-MRI phases are available across all 1506 MAMA-MIA cases.

Scans MAMA_MIA_COMPLETE/images and for each patient folder counts how many
phase files exist (_0000, _0001, _0002, _0003, ...). Reports:
  - distribution of phase counts (how many cases have exactly 2, 3, 4, ... phases)
  - per-collection availability (DUKE / ISPY1 / ISPY2 / NACT)
  - which specific phases are universally available
  - cases missing _0003 (the most likely gap)

Used to decide what 4 channels to put in the 4-ch preprocessing cache.

Usage:
    python scripts/audit_mama_mia_phases.py \
        --data_root "C:/Users/bhara/Desktop/MAMA_MIA_COMPLETE"

Runs in ~1 minute on a CPU. Read-only.
"""

import argparse
import re
from pathlib import Path
from collections import Counter, defaultdict


PHASE_PATTERN = re.compile(r"_(\d{4})\.nii(\.gz)?$")
COLLECTIONS = ["DUKE", "ISPY1", "ISPY2", "NACT"]


def discover_phases(patient_folder: Path) -> set[int]:
    """Return the set of phase numbers present in this patient folder."""
    phases = set()
    for f in patient_folder.iterdir():
        m = PHASE_PATTERN.search(f.name)
        if m:
            phases.add(int(m.group(1)))
    return phases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default=r"C:\Users\bhara\Desktop\MAMA_MIA_COMPLETE")
    args = parser.parse_args()

    images_dir = Path(args.data_root) / "images"
    if not images_dir.exists():
        print(f"ERROR: {images_dir} does not exist.")
        return

    # phase_set per patient
    per_patient_phases = {}
    per_collection_phases = defaultdict(list)  # collection -> list[set[int]]

    for patient_folder in sorted(images_dir.iterdir()):
        if not patient_folder.is_dir():
            continue
        name = patient_folder.name
        collection = name.split("_")[0]
        if collection not in COLLECTIONS:
            continue
        phases = discover_phases(patient_folder)
        if not phases:
            continue
        per_patient_phases[name] = phases
        per_collection_phases[collection].append(phases)

    total = len(per_patient_phases)
    print(f"{'='*60}")
    print(f"MAMA-MIA Phase Audit")
    print(f"{'='*60}")
    print(f"Data root      : {args.data_root}")
    print(f"Total patients : {total}")
    print()

    # ── 1. Distribution of phase COUNT per patient ────────────────────────
    count_dist = Counter(len(p) for p in per_patient_phases.values())
    print("Distribution of phase counts per patient:")
    print(f"  {'#phases':>10} | {'count':>6} | {'%':>6}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*6}")
    for n in sorted(count_dist.keys()):
        c = count_dist[n]
        pct = 100.0 * c / total
        print(f"  {n:>10} | {c:>6} | {pct:>5.1f}%")
    print()

    # ── 2. Per-collection availability ────────────────────────────────────
    print("Per-collection phase availability:")
    print(f"  {'collection':>11} | {'n':>5} | {'phases per case (mean/min/max)':>32}")
    print(f"  {'-'*11}-+-{'-'*5}-+-{'-'*32}")
    for col in COLLECTIONS:
        if col not in per_collection_phases:
            continue
        phases_list = per_collection_phases[col]
        counts = [len(p) for p in phases_list]
        mean = sum(counts) / len(counts)
        print(f"  {col:>11} | {len(phases_list):>5} | "
              f"mean={mean:>4.1f}, min={min(counts)}, max={max(counts)}")
    print()

    # ── 3. Universal availability per phase number ───────────────────────
    print("Per-phase availability (how many of the 1506 cases have each phase):")
    print(f"  {'phase':>6} | {'count':>6} | {'%':>6} | universal? | per-collection (DUKE/ISPY1/ISPY2/NACT)")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*44}")
    max_phase = max(max(p) for p in per_patient_phases.values())
    for phase in range(0, max_phase + 1):
        c = sum(1 for p in per_patient_phases.values() if phase in p)
        if c == 0:
            continue
        pct = 100.0 * c / total
        universal = "YES" if c == total else "no"
        # per-collection breakdown
        per_col = []
        for col in COLLECTIONS:
            n_total_col = len(per_collection_phases[col])
            n_with_phase = sum(1 for p in per_collection_phases[col] if phase in p)
            per_col.append(f"{n_with_phase}/{n_total_col}")
        per_col_str = "  ".join(per_col)
        print(f"  _{phase:04d} | {c:>6} | {pct:>5.1f}% | {universal:>10} | {per_col_str}")
    print()

    # ── 4. How many have ALL of {0,1,2,3}? ────────────────────────────────
    needed_4ch = {0, 1, 2, 3}
    have_all_4 = [name for name, phases in per_patient_phases.items() if needed_4ch.issubset(phases)]
    print(f"Cases with all of _0000, _0001, _0002, _0003: "
          f"{len(have_all_4)} / {total} ({100*len(have_all_4)/total:.1f}%)")

    # Per-collection breakdown of 4-phase availability
    print(f"  per-collection:")
    for col in COLLECTIONS:
        col_cases = [n for n in have_all_4 if n.startswith(col)]
        col_total = len(per_collection_phases[col])
        print(f"    {col:>6}: {len(col_cases)}/{col_total} "
              f"({100*len(col_cases)/col_total:.1f}%)")
    print()

    # ── 5. List the cases MISSING any of the 4 phases (for diagnostics) ───
    missing_4ch = [(name, sorted(needed_4ch - phases))
                   for name, phases in per_patient_phases.items()
                   if not needed_4ch.issubset(phases)]
    if missing_4ch:
        print(f"Cases missing one or more of _0000-_0003: {len(missing_4ch)}")
        # Show first 20 to keep output sane
        print(f"  first 20:")
        for name, missing in missing_4ch[:20]:
            print(f"    {name}: missing {missing}")
        if len(missing_4ch) > 20:
            print(f"  ... and {len(missing_4ch) - 20} more")
    print()

    # ── 6. Recommendation ────────────────────────────────────────────────
    print(f"{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    pct_all_4 = 100 * len(have_all_4) / total
    if pct_all_4 >= 95:
        print(f"OK: {pct_all_4:.1f}% of cases have all 4 phases (_0000-_0003).")
        print(f"    Build 4-ch cache using: [_0000, _0001, _0002, _0003]")
        print(f"    Skip {len(missing_4ch)} incomplete cases ({100-pct_all_4:.1f}%)")
    elif pct_all_4 >= 80:
        print(f"PARTIAL: {pct_all_4:.1f}% have all 4 phases. Trade-off:")
        print(f"    Option A: 4-ch [_0000,_0001,_0002,_0003], drop {len(missing_4ch)} cases")
        print(f"    Option B: 4-ch [_0000,_0001,_0002,subtraction(_0001-_0000)], keep all")
    else:
        print(f"WARN: only {pct_all_4:.1f}% have all 4 phases.")
        print(f"    Recommend 4-ch [_0000,_0001,_0002,subtraction] to keep all cases.")
        print(f"    Universal-phase 4-ch concat is NOT feasible without losing >20% of data.")


if __name__ == "__main__":
    main()
