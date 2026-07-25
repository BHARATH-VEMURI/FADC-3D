"""Focused tests for cross-session Kaggle resume support in the DS + nods
70/10/20 notebooks. Runs offline — no GPU, no real MRI data.

Coverage:
  R1  Preflight rejects asymmetric CONFIG (one of RESUME_INPUT_DIR /
      RESUME_FROM populated, the other empty AND not shorthand-defaultable).
  R2  Preflight defaults RESUME_FROM to 'last_checkpoint.pth' when
      RESUME_INPUT_DIR is set but RESUME_FROM is empty (shorthand).
  R3  Both empty -> IS_RESUME_MODE False; the existing stale-.pth guard
      still fires on a dirty OUTPUT_DIR_FULL.
  R4  Both set + resume ckpt exists -> IS_RESUME_MODE True; preflight
      succeeds and the stale guard accepts the copied ckpt.
  R5  Resume mode with a non-existent RESUME_FROM -> preflight aborts.
  R6  Both notebooks: cell 11 (FULL training) appends --resume when
      IS_RESUME_MODE is True.
  R7  Both notebooks: cell 11 keeps the fresh-only no-`--resume` assertion
      when IS_RESUME_MODE is False.
  R8  Both notebooks: cells 8 + 9 (smoke train + smoke reload) skip when
      IS_RESUME_MODE is True.
  R9  Both notebooks add RESUME_INPUT_DIR + RESUME_FROM to CONFIG.
  R10 The split-cell resume block hard-checks arch_identity['model_name'],
      arch_identity['deep_supervision'], and split_identity['split_manifest_sha256'].
"""
from __future__ import annotations

import ast
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

NB_DS   = REPO / "kaggle_train_fadc3d_correct_encoder_ds_split701020_s42.ipynb"
NB_NODS = REPO / "kaggle_train_fadc3d_correct_encoder_nods_split701020_s42.ipynb"


def _cell(nb_path, i):
    with io.open(nb_path, encoding="utf-8") as f:
        return "".join(json.load(f)["cells"][i]["source"])


def _exec_cfg_and_preflight(nb_path, cfg_overrides, preflight_extra_env=None):
    """Run CONFIG then PREFLIGHT of a notebook in a fresh namespace, applying
    the given CONFIG overrides. Returns the resulting namespace or the
    raised exception."""
    cfg = _cell(nb_path, 1)
    pre = _cell(nb_path, 2)
    ns = {"sys": sys, "os": os}
    exec(compile(cfg, "<cfg>", "exec"), ns, ns)
    ns.update(cfg_overrides)
    if preflight_extra_env:
        ns.update(preflight_extra_env)
    try:
        exec(compile(pre, "<pre>", "exec"), ns, ns)
        return ns, None
    except (SystemExit, AssertionError) as e:
        return ns, e


# ─────────────────────────────────────────────────────────────────────────
# R9 — CONFIG defines both vars
# ─────────────────────────────────────────────────────────────────────────

def test_R9_config_defines_both_resume_vars():
    for label, nb in (("DS", NB_DS), ("nods", NB_NODS)):
        cfg = _cell(nb, 1)
        assert re.search(r'^RESUME_INPUT_DIR\s*=\s*""', cfg, flags=re.MULTILINE), \
            f"{label}: CONFIG missing RESUME_INPUT_DIR default"
        assert re.search(r'^RESUME_FROM\s*=\s*""', cfg, flags=re.MULTILINE), \
            f"{label}: CONFIG missing RESUME_FROM default"
    print("[R9] OK — both notebooks define RESUME_INPUT_DIR and RESUME_FROM in CONFIG")


# ─────────────────────────────────────────────────────────────────────────
# R1 — asymmetric config rejected (only RESUME_FROM populated is asymmetric)
# ─────────────────────────────────────────────────────────────────────────

def test_R1_preflight_rejects_asymmetric_config():
    for label, nb in (("DS", NB_DS), ("nods", NB_NODS)):
        # The "asymmetric-but-not-shorthand-defaultable" case: FROM is set
        # but INPUT_DIR is empty. That MUST hard-error.
        overrides = {"RESUME_INPUT_DIR": "", "RESUME_FROM": "last_checkpoint.pth"}
        # nods notebook also requires EXPECTED_MANIFEST_SHA256 to be set;
        # pre-set a valid 64-hex placeholder so we don't trip that other guard.
        if label == "nods":
            overrides["EXPECTED_MANIFEST_SHA256"] = "0" * 64
        _, err = _exec_cfg_and_preflight(nb, overrides)
        assert isinstance(err, SystemExit), \
            f"{label}: expected SystemExit on RESUME_FROM-only config, got {err!r}"
        assert "RESUME_INPUT_DIR" in str(err) or "resume" in str(err).lower(), \
            f"{label}: SystemExit message must mention the resume config: {err!r}"
        print(f"  [R1/{label}] asymmetric rejected: {str(err)[:100]}")
    print("[R1] OK — asymmetric CONFIG rejected in both notebooks")


# ─────────────────────────────────────────────────────────────────────────
# R2 — shorthand default (INPUT_DIR set, FROM empty -> FROM defaults)
# ─────────────────────────────────────────────────────────────────────────

def test_R2_preflight_shorthand_defaults_resume_from():
    tmp = Path(tempfile.mkdtemp(prefix="fadc_resume_"))
    (tmp / "last_checkpoint.pth").write_bytes(b"dummy")  # existence only
    try:
        for label, nb in (("DS", NB_DS), ("nods", NB_NODS)):
            overrides = {"RESUME_INPUT_DIR": str(tmp), "RESUME_FROM": ""}
            if label == "nods":
                overrides["EXPECTED_MANIFEST_SHA256"] = "0" * 64
            ns, err = _exec_cfg_and_preflight(nb, overrides)
            assert err is None, f"{label}: shorthand preflight should not raise: {err!r}"
            assert ns.get("RESUME_FROM") == "last_checkpoint.pth", \
                f"{label}: RESUME_FROM should default to 'last_checkpoint.pth', got {ns.get('RESUME_FROM')!r}"
            assert ns.get("IS_RESUME_MODE") is True, \
                f"{label}: IS_RESUME_MODE should be True; got {ns.get('IS_RESUME_MODE')!r}"
            print(f"  [R2/{label}] shorthand OK — RESUME_FROM defaulted to 'last_checkpoint.pth'")
        print("[R2] OK — shorthand default works in both notebooks")
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# R3 — both empty -> fresh mode + stale guard fires when OUTPUT_DIR_FULL dirty
# ─────────────────────────────────────────────────────────────────────────

def _make_notebook_output_dir(label: str, prefix: str) -> Path:
    """Build a tmp OUTPUT_DIR_FULL that passes each notebook's preflight
    substring/basename guards."""
    if label == "DS":
        basename = "fadc3d_correct_encoder_ds_split701020_test"
    else:
        basename = "fadc3d_correct_encoder_nods_split701020_test"
    d = Path(tempfile.mkdtemp(prefix=prefix)) / basename
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_R3_fresh_mode_still_refuses_stale_pth():
    try:
        for label, nb in (("DS", NB_DS), ("nods", NB_NODS)):
            tmp_out = _make_notebook_output_dir(label, "fadc_stale_")
            (tmp_out / "leftover.pth").write_bytes(b"stale")
            overrides = {
                "OUTPUT_DIR_FULL": str(tmp_out),
                "RESUME_INPUT_DIR": "",
                "RESUME_FROM": "",
            }
            if label == "nods":
                overrides["EXPECTED_MANIFEST_SHA256"] = "0" * 64
            ns, err = _exec_cfg_and_preflight(nb, overrides)
            assert isinstance(err, SystemExit), \
                f"{label}: fresh mode with stale .pth should SystemExit, got {err!r}"
            assert "fresh-only" in str(err).lower() or "already contains" in str(err).lower(), \
                f"{label}: stale-guard message wrong: {err!r}"
            print(f"  [R3/{label}] fresh mode still refuses stale .pth")
        print("[R3] OK — fresh mode preserves the stale-.pth guard")
    finally:
        import shutil; shutil.rmtree(tmp_out, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# R4 — resume mode accepts a resume ckpt in OUTPUT_DIR_FULL
# ─────────────────────────────────────────────────────────────────────────

def test_R4_resume_mode_accepts_copied_ckpt_in_output_dir():
    tmp_in  = Path(tempfile.mkdtemp(prefix="fadc_resume_in_"))
    (tmp_in  / "last_checkpoint.pth").write_bytes(b"src")
    try:
        for label, nb in (("DS", NB_DS), ("nods", NB_NODS)):
            tmp_out = _make_notebook_output_dir(label, "fadc_resume_out_")
            (tmp_out / "last_checkpoint.pth").write_bytes(b"copied")  # simulates cell-5 copy
            overrides = {
                "OUTPUT_DIR_FULL": str(tmp_out),
                "RESUME_INPUT_DIR": str(tmp_in),
                "RESUME_FROM": "last_checkpoint.pth",
            }
            if label == "nods":
                overrides["EXPECTED_MANIFEST_SHA256"] = "0" * 64
            ns, err = _exec_cfg_and_preflight(nb, overrides)
            assert err is None, \
                f"{label}: preflight should accept copied ckpt in resume mode: {err!r}"
            assert ns.get("IS_RESUME_MODE") is True
            print(f"  [R4/{label}] resume mode accepts existing copied ckpt")
        print("[R4] OK — resume mode preflight accepts the copied ckpt path")
    finally:
        import shutil
        shutil.rmtree(tmp_in,  ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# R5 — resume mode aborts on missing source ckpt
# ─────────────────────────────────────────────────────────────────────────

def test_R5_resume_mode_aborts_on_missing_source_ckpt():
    tmp_in = Path(tempfile.mkdtemp(prefix="fadc_resume_missing_"))
    # deliberately don't create the resume ckpt inside
    try:
        for label, nb in (("DS", NB_DS), ("nods", NB_NODS)):
            overrides = {
                "RESUME_INPUT_DIR": str(tmp_in),
                "RESUME_FROM": "nonexistent.pth",
            }
            if label == "nods":
                overrides["EXPECTED_MANIFEST_SHA256"] = "0" * 64
            ns, err = _exec_cfg_and_preflight(nb, overrides)
            assert isinstance(err, SystemExit), \
                f"{label}: missing resume ckpt should SystemExit, got {err!r}"
            assert "not found" in str(err).lower() or "not exist" in str(err).lower(), \
                f"{label}: message wrong: {err!r}"
            print(f"  [R5/{label}] missing source ckpt aborts: {str(err)[:80]}")
        print("[R5] OK — resume mode aborts on missing source ckpt")
    finally:
        import shutil; shutil.rmtree(tmp_in, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# R6 + R7 — cell 11 --resume behavior
# ─────────────────────────────────────────────────────────────────────────

def _extract_and_run_cell11(nb_path, is_resume_mode):
    """Run CONFIG through cell 11 in a minimal-monkeypatched namespace, so we
    can observe the FINAL cmd list including any --resume appended by the
    resume block. Stops execution just before subprocess.Popen fires by
    monkeypatching Popen to record cmd and raise a sentinel."""
    ns = {"sys": sys, "os": os, "IS_RESUME_MODE": is_resume_mode}
    # Preload variables that cell 11 reads from earlier cells.
    exec(compile(_cell(nb_path, 1), "<cfg>", "exec"), ns, ns)
    ns["IS_RESUME_MODE"] = is_resume_mode
    ns["OUTPUT_DIR_FULL"] = tempfile.mkdtemp(prefix="fadc_c11_")
    if is_resume_mode:
        (Path(ns["OUTPUT_DIR_FULL"]) / "last_checkpoint.pth").write_bytes(b"x")
    ns["CODE_DIR"] = "/fake/code"
    ns["MANIFEST_CSV"] = "/fake/manifest.csv"
    ns["MANIFEST_SHA256"] = "0" * 64
    ns["MANIFEST_SOURCE"] = "unit-test"

    # Monkey-patch subprocess.Popen to capture argv and abort.
    import subprocess as _sp
    captured = {}
    class _P:
        def __init__(self, cmd, *a, **kw):
            captured["cmd"] = list(cmd)
            raise RuntimeError("STOP_BEFORE_LAUNCH")  # sentinel
    ns["__patched_popen"] = _P
    body = _cell(nb_path, 11).replace("subprocess.Popen", "__patched_popen")
    ns["subprocess"] = _sp
    try:
        exec(compile(body, "<c11>", "exec"), ns, ns)
    except RuntimeError as e:
        if "STOP_BEFORE_LAUNCH" not in str(e):
            raise
    except SystemExit as e:
        return {"error": e, "cmd": captured.get("cmd")}
    return {"error": None, "cmd": captured.get("cmd")}


def test_R6_cell11_appends_resume_flag_when_resume_mode():
    for label, nb in (("DS", NB_DS), ("nods", NB_NODS)):
        out = _extract_and_run_cell11(nb, is_resume_mode=True)
        assert out["cmd"] is not None, \
            f"{label}: cell 11 did not reach subprocess.Popen in resume mode. err={out['error']!r}"
        assert "--resume" in out["cmd"], \
            f"{label}: cell 11 cmd missing --resume in resume mode: {out['cmd']}"
        i = out["cmd"].index("--resume")
        assert out["cmd"][i + 1].endswith("last_checkpoint.pth"), \
            f"{label}: --resume value should point at last_checkpoint.pth, got {out['cmd'][i+1]!r}"
        print(f"  [R6/{label}] cell 11 appends --resume {out['cmd'][i+1]}")
    print("[R6] OK — resume mode appends --resume in both notebooks")


def test_R7_cell11_keeps_no_resume_assertion_in_fresh_mode():
    for label, nb in (("DS", NB_DS), ("nods", NB_NODS)):
        out = _extract_and_run_cell11(nb, is_resume_mode=False)
        assert out["cmd"] is not None, \
            f"{label}: cell 11 did not reach subprocess.Popen in fresh mode. err={out['error']!r}"
        assert "--resume" not in out["cmd"], \
            f"{label}: fresh-mode cmd must not contain --resume: {out['cmd']}"
        print(f"  [R7/{label}] cell 11 fresh-mode cmd has no --resume")
    print("[R7] OK — fresh mode still refuses --resume in both notebooks")


# ─────────────────────────────────────────────────────────────────────────
# R8 — smoke cells 8 + 9 skip when IS_RESUME_MODE is True
# ─────────────────────────────────────────────────────────────────────────

def test_R8_smoke_and_smoke_reload_skip_in_resume_mode():
    for label, nb in (("DS", NB_DS), ("nods", NB_NODS)):
        for cell_i in (8, 9):
            src = _cell(nb, cell_i)
            # Must contain the resume-mode skip branch we injected.
            assert "if IS_RESUME_MODE:" in src, \
                f"{label}: cell {cell_i} missing `if IS_RESUME_MODE:` skip branch"
            assert "skipping" in src.lower(), \
                f"{label}: cell {cell_i} skip branch missing 'skipping' marker"
            # And the ORIGINAL body must live under an `else:` branch.
            assert re.search(r"^else:\s*$", src, flags=re.MULTILINE), \
                f"{label}: cell {cell_i} missing `else:` wrapper around original body"
            print(f"  [R8/{label}/c{cell_i}] resume-mode skip branch present")
    print("[R8] OK — smoke + smoke-reload skip in resume mode")


# ─────────────────────────────────────────────────────────────────────────
# R10 — split cell resume block hard-checks identity
# ─────────────────────────────────────────────────────────────────────────

def test_R10_split_cell_checks_identity_before_copy():
    for label, nb in (("DS", NB_DS), ("nods", NB_NODS)):
        src = _cell(nb, 5)
        assert "RESUME MODE: verify + copy" in src, \
            f"{label}: cell 5 missing resume-mode block"
        for needle in (
            'if _r_arch.get("model_name") != MODEL_NAME',
            'if bool(_r_arch.get("deep_supervision"',
            'if _r_ckpt_sha != MANIFEST_SHA256',
            'shutil.copy2(RESUME_SRC_PATH,',
        ):
            assert needle in src, \
                f"{label}: cell 5 resume block missing check `{needle}`"
        print(f"  [R10/{label}] cell 5 resume block hard-checks arch + DS + manifest SHA")
    print("[R10] OK — split-cell resume block enforces identity before copy")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_R9_config_defines_both_resume_vars,
        test_R1_preflight_rejects_asymmetric_config,
        test_R2_preflight_shorthand_defaults_resume_from,
        test_R3_fresh_mode_still_refuses_stale_pth,
        test_R4_resume_mode_accepts_copied_ckpt_in_output_dir,
        test_R5_resume_mode_aborts_on_missing_source_ckpt,
        test_R6_cell11_appends_resume_flag_when_resume_mode,
        test_R7_cell11_keeps_no_resume_assertion_in_fresh_mode,
        test_R8_smoke_and_smoke_reload_skip_in_resume_mode,
        test_R10_split_cell_checks_identity_before_copy,
    ]
    failed = []
    for fn in tests:
        print(f"\n---- {fn.__name__} ----")
        try:
            fn()
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"  FAIL: {e!r}")
    print(f"\n{'='*60}\npassed : {len(tests)-len(failed)} / {len(tests)}")
    if failed:
        for name, e in failed:
            print(f"  - {name}: {e!r}")
        sys.exit(1)
    print("ALL RESUME NOTEBOOK TESTS PASSED")
