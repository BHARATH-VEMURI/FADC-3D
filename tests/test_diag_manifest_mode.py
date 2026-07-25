"""Focused tests for the manifest-restricted diagnostic mode + notebook
contract assertions.

Coverage:
  D1  Manifest-mode selection is a strict subset of the requested partition.
  D2  Val diagnostics cannot see test patients even when both live in the
      same physical directory (data-leakage regression guard).
  D3  Legacy --preprocessed_cache directory scan still works and IS marked
      as unsafe with a warning banner.
  D4  Invalid --split_partition value raises before touching the filesystem.
  D5  Missing --split_manifest file raises clearly.
  D6  --split_manifest without --preprocessed_cache_dir raises with a
      helpful message.
  N1  Both DS and no-DS notebook cell-13 commands request split_partition=val.
  N2  Both notebooks pin the SAME exact EXPECTED_GIT_COMMIT.
  N3  The no-DS notebook's preflight refuses to launch training when
      EXPECTED_MANIFEST_SHA256 is empty.
  N4  The no-DS notebook's preflight refuses an obviously-invalid checksum
      (wrong length / non-hex).
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.split_manifest import (
    COLLECTIONS, generate_and_write, load_manifest, manifest_sha256,
)

DIAG = str(REPO / "diag_fadc_3d_correct.py")
NB_DS   = REPO / "kaggle_train_fadc3d_correct_encoder_ds_split701020_s42.ipynb"
NB_NODS = REPO / "kaggle_train_fadc3d_correct_encoder_nods_split701020_s42.ipynb"


# ─────────────────────────────────────────────────────────────────────────
# Synthetic .npz cache fixture
# ─────────────────────────────────────────────────────────────────────────

def _make_case(dir_path: Path, patient_id: str):
    dir_path.mkdir(parents=True, exist_ok=True)
    # Volume must survive 4 stride-2 pools of the UNet (min-dim >= 32).
    img = np.random.RandomState(hash(patient_id) & 0xFFFF).randn(2, 48, 48, 32).astype(np.float16)
    lbl = np.zeros((1, 48, 48, 32), dtype=np.uint8)
    np.savez(dir_path / f"{patient_id}.npz", image=img, label=lbl)


def _make_cache(root: Path, n_per_coll_train: int = 12, n_per_coll_val: int = 6):
    for coll in COLLECTIONS:
        for i in range(n_per_coll_train):
            _make_case(root / "train", f"{coll.lower()}_{i:03d}")
        for i in range(n_per_coll_val):
            _make_case(root / "val", f"{coll.lower()}_{100 + i:03d}")


def _fresh(prefix="fadc_diag_"):
    return Path(tempfile.mkdtemp(prefix=prefix))


def _run_diag(*args, cwd=None, timeout=90):
    """Invoke diag_fadc_3d_correct.py as a subprocess; return CompletedProcess."""
    cmd = [sys.executable, DIAG, *args]
    return subprocess.run(cmd, cwd=cwd or str(REPO),
                          capture_output=True, text=True, timeout=timeout)


# ─────────────────────────────────────────────────────────────────────────
# D-series: diagnostic script behaviour
# ─────────────────────────────────────────────────────────────────────────

def test_D1_manifest_mode_selects_only_requested_partition():
    root = _fresh()
    try:
        _make_cache(root, n_per_coll_train=8, n_per_coll_val=4)
        csv_path  = root / "manifest.csv"
        meta_path = root / "manifest_meta.json"
        generate_and_write(str(root), str(csv_path), str(meta_path),
                           seed=42, ratios=(0.70, 0.10, 0.20))

        val_ids = {c["patient_id"] for c in load_manifest(
            str(csv_path), split="val", cache_root=str(root))}
        train_ids = {c["patient_id"] for c in load_manifest(
            str(csv_path), split="train", cache_root=str(root))}
        test_ids  = {c["patient_id"] for c in load_manifest(
            str(csv_path), split="test",  cache_root=str(root))}
        assert val_ids and train_ids and test_ids, (len(val_ids), len(train_ids), len(test_ids))

        r = _run_diag(
            "--split_manifest", str(csv_path),
            "--split_partition", "val",
            "--preprocessed_cache_dir", str(root),
            "--n_patches", "3",
            "--patch_size", "32", "32", "16",
        )
        assert r.returncode == 0, f"diag failed:\n{r.stdout}\n{r.stderr}"
        printed = set(re.findall(r"patient_id=(\S+)", r.stdout))
        assert printed, f"diag printed no patient_ids:\n{r.stdout[-800:]}"
        assert printed.issubset(val_ids), \
            f"diag selected patients outside val: {printed - val_ids}"
        assert printed.isdisjoint(test_ids), "LEAK: diag touched test patients"
        assert printed.isdisjoint(train_ids), "LEAK: diag touched train patients"
        print(f"[D1] OK — {len(printed)} patient(s), all in val")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_D2_val_diag_cannot_see_test_patients_in_shared_dir():
    """The regression guard: if val and test both live in the same physical
    subfolder (as they may after the 70/10/20 re-partition), manifest mode
    must still refuse to sample test patients into a val-diagnostic run."""
    root = _fresh()
    try:
        # Put EVERY patient into a single physical subfolder — mimics the
        # worst case where val/test aren't physically separated.
        (root / "all").mkdir()
        pids = []
        for coll in COLLECTIONS:
            for i in range(15):
                pid = f"{coll.lower()}_{i:03d}"
                _make_case(root / "all", pid)
                pids.append(pid)
        # Fake a manifest whose train/val/test set is derived from `all/` but
        # where the CSV's relative_npz_path points into `all/`. Use the shared
        # generator with a custom subdir list.
        from training.split_manifest import (
            assign_splits, enumerate_patients, write_manifest, write_metadata,
        )
        patients = enumerate_patients(str(root), subdirs=("all",))
        assigned = assign_splits(patients, seed=42, ratios=(0.70, 0.10, 0.20))
        csv_path  = root / "manifest.csv"
        meta_path = root / "manifest_meta.json"
        write_manifest(assigned, str(csv_path))
        write_metadata(assigned, seed=42, ratios=(0.70, 0.10, 0.20),
                       csv_path=str(csv_path), meta_path=str(meta_path))

        by_split = {"train": set(), "val": set(), "test": set()}
        for c in assigned:
            by_split[c["split"]].add(c["patient_id"])
        # Precondition: val and test physically colocated.
        for split_name in ("val", "test"):
            for pid in by_split[split_name]:
                assert (root / "all" / f"{pid}.npz").exists(), pid

        r = _run_diag(
            "--split_manifest", str(csv_path),
            "--split_partition", "val",
            "--preprocessed_cache_dir", str(root),
            "--n_patches", "4",
            "--patch_size", "32", "32", "16",
        )
        assert r.returncode == 0, f"diag failed:\n{r.stdout}\n{r.stderr}"
        printed = set(re.findall(r"patient_id=(\S+)", r.stdout))
        assert printed, "no patient_ids printed"
        assert printed.issubset(by_split["val"]), \
            f"leaked non-val: {printed - by_split['val']}"
        assert printed.isdisjoint(by_split["test"]), \
            "REGRESSION: val diagnostic pulled a test patient from shared dir"
        print(f"[D2] OK — val/test shared dir, but diag stayed in val "
              f"({len(printed)} patient(s))")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_D3_legacy_directory_mode_still_works():
    root = _fresh()
    try:
        _make_cache(root, n_per_coll_train=4, n_per_coll_val=2)
        r = _run_diag(
            "--preprocessed_cache", str(root / "val"),
            "--n_patches", "2",
            "--patch_size", "32", "32", "16",
        )
        assert r.returncode == 0, f"legacy mode failed:\n{r.stdout}\n{r.stderr}"
        # Warning banner must appear.
        assert "legacy directory-scan mode" in r.stdout.lower(), \
            "expected the unsafe-mode warning banner"
        print("[D3] OK — legacy directory-scan mode still supported (with warning)")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_D4_invalid_partition_rejected_by_argparse():
    root = _fresh()
    try:
        _make_cache(root, n_per_coll_train=2, n_per_coll_val=1)
        csv_path  = root / "manifest.csv"
        meta_path = root / "manifest_meta.json"
        generate_and_write(str(root), str(csv_path), str(meta_path),
                           seed=42, ratios=(0.70, 0.10, 0.20))
        r = _run_diag(
            "--split_manifest", str(csv_path),
            "--split_partition", "training",   # not one of {train, val, test}
            "--preprocessed_cache_dir", str(root),
            "--n_patches", "1", "--patch_size", "32", "32", "16",
        )
        assert r.returncode != 0
        assert "invalid choice" in (r.stderr + r.stdout).lower(), \
            f"expected argparse choice error; got:\n{r.stdout}\n{r.stderr}"
        print("[D4] OK — invalid partition rejected")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_D5_missing_manifest_raises_clearly():
    r = _run_diag(
        "--split_manifest", "/nonexistent/manifest.csv",
        "--split_partition", "val",
        "--preprocessed_cache_dir", "/tmp",
        "--n_patches", "1", "--patch_size", "32", "32", "16",
    )
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "manifest.csv" in combined.lower() or "no such file" in combined.lower() \
        or "does not exist" in combined.lower(), \
        f"expected a clear missing-manifest error; got:\n{combined}"
    print("[D5] OK — missing manifest raises clearly")


def test_D6_manifest_without_cache_dir_raises():
    root = _fresh()
    try:
        _make_cache(root, n_per_coll_train=2, n_per_coll_val=1)
        csv_path  = root / "manifest.csv"
        meta_path = root / "manifest_meta.json"
        generate_and_write(str(root), str(csv_path), str(meta_path),
                           seed=42, ratios=(0.70, 0.10, 0.20))
        r = _run_diag(
            "--split_manifest", str(csv_path),
            "--split_partition", "val",
            # deliberately omit --preprocessed_cache_dir
            "--n_patches", "1", "--patch_size", "32", "32", "16",
        )
        assert r.returncode != 0, f"expected failure; got:\n{r.stdout}"
        combined = (r.stdout + r.stderr).lower()
        assert "preprocessed_cache_dir" in combined, \
            f"expected --preprocessed_cache_dir mention; got:\n{combined[-800:]}"
        print("[D6] OK — manifest w/o cache_dir raises")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# N-series: notebook contract assertions
# ─────────────────────────────────────────────────────────────────────────

def _nb_cell(nb_path, i):
    with io.open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    return "".join(nb["cells"][i]["source"])


def test_N1_both_notebooks_use_split_partition_val_in_cell13():
    for label, path in (("DS", NB_DS), ("nods", NB_NODS)):
        src = _nb_cell(path, 13)
        # AST-extract the cmd list.
        import ast
        cmd = None
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "cmd" \
                       and isinstance(node.value, ast.List):
                        cmd = [e.value if isinstance(e, ast.Constant) else "<non-const>"
                               for e in node.value.elts]
        assert cmd is not None, f"{label}: cell 13 has no cmd = [...]"
        assert "--split_manifest" in cmd, f"{label}: cell 13 missing --split_manifest"
        assert "--split_partition" in cmd, f"{label}: cell 13 missing --split_partition"
        i = cmd.index("--split_partition")
        assert cmd[i + 1] == "val", \
            f"{label}: cell 13 --split_partition must be 'val', got {cmd[i+1]!r}"
        # Old legacy --preprocessed_cache must NOT be used.
        assert "--preprocessed_cache" not in cmd, \
            f"{label}: cell 13 must not use legacy --preprocessed_cache (use --preprocessed_cache_dir)"
        # And --preprocessed_cache_dir must be there.
        assert "--preprocessed_cache_dir" in cmd, \
            f"{label}: cell 13 missing --preprocessed_cache_dir"
        # Return-code guard must be present.
        assert "raise SystemExit" in src and "Diagnostic subprocess failed" in src, \
            f"{label}: cell 13 does not SystemExit on nonzero diag exit"
    print("[N1] OK — both notebooks use split_partition=val, no legacy flag, SystemExit on failure")


def test_N2_both_notebooks_pin_same_exact_commit():
    ds   = _nb_cell(NB_DS, 1)
    nods = _nb_cell(NB_NODS, 1)
    ds_m   = re.search(r'EXPECTED_GIT_COMMIT\s*=\s*"([0-9a-fA-F]{40})"', ds)
    nods_m = re.search(r'EXPECTED_GIT_COMMIT\s*=\s*"([0-9a-fA-F]{40})"', nods)
    assert ds_m,   "DS CONFIG does not define EXPECTED_GIT_COMMIT as 40-hex string"
    assert nods_m, "nods CONFIG does not define EXPECTED_GIT_COMMIT as 40-hex string"
    assert ds_m.group(1) == nods_m.group(1), \
        f"DS pin={ds_m.group(1)!r} != nods pin={nods_m.group(1)!r}"
    # And both cell-4 checkouts must be detached-HEAD + verified.
    for label, path in (("DS", NB_DS), ("nods", NB_NODS)):
        c4 = _nb_cell(path, 4)
        assert "checkout" in c4 and "--detach" in c4 and "EXPECTED_GIT_COMMIT" in c4, \
            f"{label}: cell 4 does not do detached-HEAD checkout of EXPECTED_GIT_COMMIT"
        assert "GIT_COMMIT_HASH == EXPECTED_GIT_COMMIT" in c4, \
            f"{label}: cell 4 does not assert HEAD matches EXPECTED_GIT_COMMIT"
        # Never pull after pinning. Strip comments first — the pin-rationale
        # comment intentionally mentions "we never git pull after pinning".
        _code_only = "\n".join(re.sub(r"#.*$", "", line) for line in c4.splitlines())
        assert "git pull" not in _code_only and "pull --ff-only" not in _code_only, \
            f"{label}: cell 4 must not pull after pinning (executable code)"
        # Also confirm no pull operation is invoked as an actual argv.
        assert not re.search(r'\["git",\s*"-C",\s*CODE_DIR,\s*"pull"', c4), \
            f"{label}: cell 4 must not invoke git pull"
    print(f"[N2] OK — both notebooks pinned at {ds_m.group(1)}")


def test_N3_nods_preflight_rejects_empty_expected_sha():
    """Simulate: nods CONFIG with EXPECTED_MANIFEST_SHA256 = '' — preflight
    must raise SystemExit before any expensive cell runs."""
    cfg = _nb_cell(NB_NODS, 1)
    pre = _nb_cell(NB_NODS, 2)
    # Set the missing symbols the preflight cell reads (DATA_ROOT etc).
    ns = {"sys": sys, "os": os}
    exec(compile(cfg, "<cfg>", "exec"), ns, ns)
    # Force EXPECTED_MANIFEST_SHA256 to empty (default in CONFIG).
    ns["EXPECTED_MANIFEST_SHA256"] = ""
    try:
        exec(compile(pre, "<pre>", "exec"), ns, ns)
    except SystemExit as e:
        assert "EXPECTED_MANIFEST_SHA256" in str(e), f"unexpected SystemExit: {e}"
        print(f"[N3] OK — preflight aborts on empty EXPECTED_MANIFEST_SHA256: {e}")
        return
    except AssertionError as e:
        assert "EXPECTED_MANIFEST_SHA256" in str(e), f"unexpected AssertionError: {e}"
        print(f"[N3] OK — preflight asserts on empty EXPECTED_MANIFEST_SHA256: {e}")
        return
    raise AssertionError("nods preflight did not reject empty EXPECTED_MANIFEST_SHA256")


def test_N4_nods_preflight_rejects_malformed_expected_sha():
    cfg = _nb_cell(NB_NODS, 1)
    pre = _nb_cell(NB_NODS, 2)
    ns = {"sys": sys, "os": os}
    exec(compile(cfg, "<cfg>", "exec"), ns, ns)
    # Wrong-length hex.
    ns["EXPECTED_MANIFEST_SHA256"] = "abcd"
    try:
        exec(compile(pre, "<pre>", "exec"), ns, ns)
    except (SystemExit, AssertionError) as e:
        msg = str(e)
        assert "EXPECTED_MANIFEST_SHA256" in msg or "hex" in msg.lower() or "64" in msg, \
            f"unexpected error message: {msg}"
        print(f"[N4] OK — preflight rejects malformed EXPECTED_MANIFEST_SHA256: {msg[:120]}")
        return
    raise AssertionError("nods preflight accepted a malformed checksum")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fns = [
        test_D1_manifest_mode_selects_only_requested_partition,
        test_D2_val_diag_cannot_see_test_patients_in_shared_dir,
        test_D3_legacy_directory_mode_still_works,
        test_D4_invalid_partition_rejected_by_argparse,
        test_D5_missing_manifest_raises_clearly,
        test_D6_manifest_without_cache_dir_raises,
        test_N1_both_notebooks_use_split_partition_val_in_cell13,
        test_N2_both_notebooks_pin_same_exact_commit,
        test_N3_nods_preflight_rejects_empty_expected_sha,
        test_N4_nods_preflight_rejects_malformed_expected_sha,
    ]
    failures = []
    for fn in fns:
        print(f"\n---- {fn.__name__} ----")
        try:
            fn()
        except Exception as e:
            failures.append((fn.__name__, e))
            print(f"  FAIL: {e}")
    print(f"\n{'='*60}\npassed : {len(fns)-len(failures)} / {len(fns)}")
    if failures:
        for name, e in failures:
            print(f"  - {name}: {e}")
        sys.exit(1)
    print("ALL DIAG + NOTEBOOK CONTRACT TESTS PASSED")
