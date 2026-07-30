"""Notebook contract tests.

Enforce that the continuous default-split notebook has the memory-safe
config (BATCH_SIZE=1, GRAD_ACCUM_STEPS=2, VAL_SW_BATCH_SIZE=1) while the
discrete siblings remain untouched at the historical values (BATCH_SIZE=2,
VAL_SW_BATCH_SIZE=4). Runs on CPU; parses raw .ipynb JSON only.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _config_cell_text(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "CONFIG" in src and "BATCH_SIZE" in src:
            return src
    raise AssertionError(f"no CONFIG cell found in {nb_path.name}")


def _int_assign(src: str, name: str) -> int:
    m = re.search(rf"^{name}\s*=\s*(\d+)", src, re.M)
    if not m:
        raise AssertionError(f"assignment for {name} not found")
    return int(m.group(1))


CONTINUOUS_NB = REPO / "kaggle_train_fadc3d_continuous_encoder_nods_defaultsplit_val5_s42.ipynb"
DISCRETE_NBS = [
    REPO / "kaggle_train_fadc3d_discrete_encoder_ds_defaultsplit_val5_s42.ipynb",
    REPO / "kaggle_train_fadc3d_discrete_encoder_nods_defaultsplit_val5_s42.ipynb",
]


def test_continuous_notebook_has_memory_safe_config():
    src = _config_cell_text(CONTINUOUS_NB)
    assert _int_assign(src, "BATCH_SIZE") == 1
    assert _int_assign(src, "GRAD_ACCUM_STEPS") == 2
    assert "EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS" in src
    assert _int_assign(src, "VAL_SW_BATCH_SIZE") == 1
    assert _int_assign(src, "EPOCHS") == 100
    assert "PATCH_SIZE     = [128, 128, 64]" in src
    assert "SMOKE_PATCH_SIZE = [48, 48, 32]" in src


def test_continuous_notebook_preflight_locks_config():
    nb = json.loads(CONTINUOUS_NB.read_text(encoding="utf-8"))
    preflight = None
    for c in nb["cells"]:
        s = "".join(c["source"])
        if "PREFLIGHT" in s and "VAL_SW_BATCH_SIZE" in s:
            preflight = s
            break
    assert preflight is not None
    assert "assert VAL_SW_BATCH_SIZE == 1" in preflight
    assert "assert BATCH_SIZE == 1" in preflight
    assert "assert GRAD_ACCUM_STEPS == 2" in preflight
    assert "EFFECTIVE_BATCH_SIZE == BATCH_SIZE * GRAD_ACCUM_STEPS" in preflight


def test_continuous_notebook_full_cell_wires_grad_accum():
    nb = json.loads(CONTINUOUS_NB.read_text(encoding="utf-8"))
    full = None
    for c in nb["cells"]:
        s = "".join(c["source"])
        if "FULL TRAINING command" in s:
            full = s
            break
    assert full is not None, "no FULL TRAINING cell found"
    assert "--grad_accum_steps" in full
    assert 'str(GRAD_ACCUM_STEPS)' in full
    # marker CONTENT validation (spec item 14)
    assert "marker mismatch" in full
    assert "continuous_smoke_ok.json" in full


def test_discrete_notebooks_unchanged():
    # Both discrete siblings retain the historical BATCH_SIZE=2 and
    # VAL_SW_BATCH_SIZE=4. If either has been silently touched, this test
    # fails so we know before running them.
    for nb_path in DISCRETE_NBS:
        if not nb_path.exists():
            # Some experiments have been deleted; skip missing ones.
            print(f"  (skipping missing notebook: {nb_path.name})")
            continue
        src = _config_cell_text(nb_path)
        assert _int_assign(src, "BATCH_SIZE") == 2, nb_path.name
        assert _int_assign(src, "VAL_SW_BATCH_SIZE") == 4, nb_path.name
        # Grad-accum is a continuous-only opt-in; discrete notebooks
        # should not touch it (otherwise their historical Dice numbers
        # become incomparable).
        assert "GRAD_ACCUM_STEPS" not in src, nb_path.name


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"\n{len(fns)} test(s) passed.")
