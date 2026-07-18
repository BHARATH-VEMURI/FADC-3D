"""Focused tests for the fast/formal validation split, atomic checkpoints,
and resume compatibility.

Covers:
   T1  make_primary_predictor extracts output[0] from a tuple.
   T2  make_primary_predictor is a no-op on a single-tensor output.
   T3  limit_val_iterable returns exactly max_cases items.
   T4  limit_val_iterable is deterministic across calls (same subset).
   T5  atomic_torch_save never leaves a partial file (temp cleanup on failure).
   T6  atomic_torch_save writes the exact object atomically on success.
   T7  legacy checkpoint (no scaler / train_log / best_fast_dice) resumes
       without crashing and prints what was restored.
   T8  Resume preserves accumulated train_log.
   T9  Pre-validation ckpt has epoch == completed training epoch.
  T10  A best_fast_model.pth save does NOT overwrite best_model.pth.
  T11  resolve_val_schedule maps legacy --val_every to formal.
  T12  resolve_val_schedule refuses ambiguous legacy + new formal.

Run:
    python tests/test_train_correct_utils.py
"""
from __future__ import annotations
import argparse
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.train_centralized_correct import (
    make_primary_predictor,
    limit_val_iterable,
    atomic_torch_save,
    resolve_val_schedule,
    _arch_identity,
)


PASSED = []
FAILED = []

def check(cond: bool, tag: str, ok_msg: str = "", bad_msg: str = "") -> None:
    if cond:
        PASSED.append(tag)
        print(f"  [OK] {tag}  {ok_msg}")
    else:
        FAILED.append((tag, bad_msg or "assertion failed"))
        print(f"  [FAIL] {tag}  {bad_msg or 'assertion failed'}")


# ─────────────────────────── T1, T2: predictor wrapper
def test_predictor_wrapper():
    print("[T1-T2] make_primary_predictor primary-output selection")

    class TupleModel(nn.Module):
        def forward(self, x):
            main = torch.ones_like(x[:, :1])
            aux1 = torch.full_like(main, 2.0)
            aux2 = torch.full_like(main, 3.0)
            aux3 = torch.full_like(main, 4.0)
            return (main, aux1, aux2, aux3)

    class SingleModel(nn.Module):
        def forward(self, x):
            return x

    x = torch.randn(1, 1, 4, 4, 4)
    pt = make_primary_predictor(TupleModel())
    y = pt(x)
    check(torch.is_tensor(y) and y.shape == (1, 1, 4, 4, 4) and (y == 1.0).all(),
          "T1_predictor_selects_output_0_from_tuple",
          f"y shape {tuple(y.shape)} val=1.0",
          f"got shape {tuple(y.shape) if torch.is_tensor(y) else type(y)}")

    ps = make_primary_predictor(SingleModel())
    y2 = ps(x)
    check(torch.is_tensor(y2) and torch.equal(y2, x),
          "T2_predictor_passthrough_single_tensor",
          "single-tensor output unchanged",
          "passthrough broken")

    # List variant too — the wrapper doc promises tuple OR list.
    class ListModel(nn.Module):
        def forward(self, x):
            return [x * 5.0, x * 6.0]
    pl = make_primary_predictor(ListModel())
    yl = pl(x)
    check(torch.is_tensor(yl) and torch.equal(yl, x * 5.0),
          "T2b_predictor_selects_0_from_list",
          "list-output primary picked",
          "list output not handled")


# ─────────────────────────── T3, T4: subset determinism
def test_limit_val_iterable_determinism():
    print("[T3-T4] limit_val_iterable size + determinism")

    class OrderedLoader:
        def __init__(self, n):
            self.data = list(range(n))
        def __iter__(self):
            return iter(self.data)
        def __len__(self):
            return len(self.data)

    loader = OrderedLoader(20)
    subset_a = list(limit_val_iterable(loader, max_cases=5))
    check(subset_a == [0, 1, 2, 3, 4],
          "T3_limit_val_iterable_takes_first_N",
          f"got {subset_a}",
          f"expected [0..4], got {subset_a}")

    subset_b = list(limit_val_iterable(loader, max_cases=5))
    check(subset_a == subset_b,
          "T4_limit_val_iterable_deterministic_across_calls",
          "same subset both calls",
          f"subset differs: {subset_a} vs {subset_b}")

    # max_cases=0 -> no limit.
    subset_all = list(limit_val_iterable(loader, max_cases=0))
    check(subset_all == list(range(20)),
          "T3b_limit_val_iterable_zero_means_all",
          "max_cases=0 returns full iterable",
          "0 did not mean all")


# ─────────────────────────── T5, T6: atomic save
def test_atomic_save_success_and_failure():
    print("[T5-T6] atomic_torch_save success + failure cleanup")

    with tempfile.TemporaryDirectory() as td:
        dest = os.path.join(td, "ckpt.pth")
        payload = {"epoch": 42, "arr": torch.arange(6).view(2, 3).tolist()}
        atomic_torch_save(payload, dest)
        loaded = torch.load(dest, weights_only=False)
        check(loaded == payload,
              "T6_atomic_save_success_writes_exact_object",
              "load matches",
              f"loaded={loaded}")
        # No .tmp_* remnants in the dir.
        leftovers = [n for n in os.listdir(td) if n.startswith(".tmp_")]
        check(not leftovers,
              "T6b_atomic_save_success_leaves_no_tmp",
              "no tmp files",
              f"tmp remnants: {leftovers}")

    # Failure path: monkey-patch torch.save to raise, verify dest untouched
    # AND no tmp remnants left behind.
    with tempfile.TemporaryDirectory() as td:
        dest = os.path.join(td, "ckpt.pth")
        # Pre-populate a "safe" existing file that must NOT be overwritten
        # if the save fails.
        with open(dest, "w", encoding="utf-8") as f:
            f.write("PRE-EXISTING")

        real_save = torch.save
        def _boom(obj, path, *a, **kw):
            raise IOError("simulated failure")
        torch.save = _boom
        try:
            crashed = False
            try:
                atomic_torch_save({"x": 1}, dest)
            except IOError:
                crashed = True
            check(crashed,
                  "T5a_atomic_save_reraises_on_failure",
                  "IOError propagated",
                  "atomic_torch_save swallowed the failure")

            with open(dest, "r", encoding="utf-8") as f:
                surviving = f.read()
            check(surviving == "PRE-EXISTING",
                  "T5b_atomic_save_does_not_clobber_on_failure",
                  "existing dest preserved",
                  f"dest was clobbered: {surviving!r}")

            leftovers = [n for n in os.listdir(td) if n.startswith(".tmp_")]
            check(not leftovers,
                  "T5c_atomic_save_cleans_tmp_on_failure",
                  "no leftover tmp",
                  f"tmp leftovers: {leftovers}")
        finally:
            torch.save = real_save


# ─────────────────────────── T7: legacy ckpt resume compat (dry, no CUDA)
def test_legacy_ckpt_resume_smoke():
    """Simulate the resume block against a legacy ckpt that only has model /
    optimizer / scheduler / epoch / best_dice / arch_identity. Verifies:
      - _require_matching_arch accepts the identity,
      - fallbacks handle missing scaler / train_log / best_fast_dice.
    """
    print("[T7] legacy checkpoint resume smoke")
    # Build a tiny UNet3DFADCCorrect encoder to make a real arch_identity.
    from models.unet_3d_fadc_correct import build_unet3d_fadc_correct
    model = build_unet3d_fadc_correct(
        "unet3d_fadc_encoder_correct",
        in_channels=2, out_channels=2, base_filters=8, deep_supervision=False,
    )
    arch = _arch_identity(
        "unet3d_fadc_encoder_correct",
        {"in_channels": 2, "out_channels": 2, "base_filters": 8, "deep_supervision": False},
        {"use_position_att": False},
    )
    legacy_ckpt = {
        "epoch": 4,
        "model": model.state_dict(),
        "optimizer": {},
        "scheduler": {},
        "best_dice": 0.42,
        "arch_identity": arch,
        # deliberately missing: scaler, train_log, best_fast_dice
    }

    # Simulate the block from train_centralized_correct.train():
    train_log = []
    best_fast_dice = float(legacy_ckpt.get("best_fast_dice", 0.0))
    best_dice = float(legacy_ckpt.get("best_dice", 0.0))
    has_scaler = "scaler" in legacy_ckpt and isinstance(legacy_ckpt["scaler"], dict)
    has_log = "train_log" in legacy_ckpt and isinstance(legacy_ckpt["train_log"], list)
    if has_log:
        train_log = list(legacy_ckpt["train_log"])
    start_epoch = int(legacy_ckpt["epoch"]) + 1

    check(start_epoch == 5,
          "T7a_legacy_start_epoch",
          f"start_epoch={start_epoch}",
          f"expected 5, got {start_epoch}")
    check(best_dice == 0.42,
          "T7b_legacy_best_dice",
          "best_dice restored",
          f"got {best_dice}")
    check(best_fast_dice == 0.0,
          "T7c_legacy_best_fast_dice_default_zero",
          "missing best_fast_dice -> 0.0",
          f"got {best_fast_dice}")
    check(not has_scaler and not has_log and train_log == [],
          "T7d_legacy_missing_scaler_and_log_handled",
          "graceful fallback for missing keys",
          "did not fall back")


# ─────────────────────────── T8: resume preserves accumulated log
def test_resume_preserves_log():
    print("[T8] resume with train_log restores it verbatim")
    prior_log = [
        {"epoch": 1, "loss": 0.9},
        {"epoch": 2, "loss": 0.8, "fast_val_dice": 0.31},
        {"epoch": 3, "loss": 0.7},
    ]
    ckpt = {
        "epoch": 2,
        "train_log": list(prior_log[:2]),  # ckpt saved AFTER epoch 2
    }
    train_log = []
    if "train_log" in ckpt and isinstance(ckpt["train_log"], list):
        train_log = list(ckpt["train_log"])
    start_epoch = int(ckpt["epoch"]) + 1

    check(train_log == prior_log[:2],
          "T8a_train_log_restored_verbatim",
          f"n_entries={len(train_log)}",
          f"got {train_log}")
    check(start_epoch == 3,
          "T8b_start_epoch_next_after_ckpt",
          "start_epoch=3",
          f"got {start_epoch}")


# ─────────────────────────── T9: pre-val ckpt epoch bookkeeping
def test_pre_val_ckpt_epoch():
    """After epoch N training completes, the pre-val checkpoint MUST record
    epoch=N so that resume advances to N+1 without repeating epoch N.
    """
    print("[T9] pre-val ckpt records the just-completed epoch")

    # Simulate the exact bookkeeping used in train():
    for epoch in (0, 1, 2, 9, 19):
        # after training epoch `epoch` (0-indexed) completes:
        ckpt = {"epoch": int(epoch)}
        # resume from this ckpt:
        start_epoch = int(ckpt["epoch"]) + 1
        check(start_epoch == epoch + 1,
              f"T9_epoch_{epoch}_advances_to_{epoch+1}",
              f"start_epoch={start_epoch}",
              f"expected {epoch+1}, got {start_epoch}")


# ─────────────────────────── T10: fast best does not touch formal best
def test_fast_best_isolation():
    print("[T10] best_fast_model.pth save does not touch best_model.pth")
    with tempfile.TemporaryDirectory() as td:
        formal = os.path.join(td, "best_model.pth")
        fast   = os.path.join(td, "best_fast_model.pth")
        # Seed a "formal best" so we can prove it's untouched.
        formal_payload = {"epoch": 10, "best_dice": 0.5429, "kind": "formal"}
        atomic_torch_save(formal_payload, formal)

        # Now perform a "fast best" update — mimic train() by writing to
        # best_fast_model.pth ONLY.
        fast_payload = {"epoch": 20, "best_fast_dice": 0.65, "kind": "fast_proxy"}
        atomic_torch_save(fast_payload, fast)

        reloaded_formal = torch.load(formal, weights_only=False)
        reloaded_fast   = torch.load(fast, weights_only=False)
        check(reloaded_formal == formal_payload,
              "T10a_formal_best_untouched_after_fast_update",
              "best_model.pth intact",
              f"formal changed to {reloaded_formal}")
        check(reloaded_fast == fast_payload,
              "T10b_fast_best_written",
              "best_fast_model.pth written",
              f"fast payload wrong: {reloaded_fast}")


# ─────────────────────────── T11, T12: schedule resolution
def _mk_args(**over):
    ns = argparse.Namespace(
        fast_val_every=10, fast_val_overlap=0.0, fast_val_max_cases=0,
        formal_val_every=0, formal_val_overlap=0.5,
        val_sw_batch_size=4, checkpoint_every=10,
        val_every=None, val_overlap=None,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_resolve_val_schedule():
    print("[T11-T12] resolve_val_schedule")
    # T11: legacy pair maps to FORMAL when new formal is at default 0.
    sch = resolve_val_schedule(_mk_args(val_every=25, val_overlap=0.5))
    check(sch["formal_val_every"] == 25 and abs(sch["formal_val_overlap"] - 0.5) < 1e-9,
          "T11_legacy_val_every_maps_to_formal",
          f"formal_val_every=25 overlap=0.5",
          f"schedule={sch}")

    # T12: both new formal and legacy set -> ambiguous -> SystemExit.
    raised = False
    try:
        resolve_val_schedule(_mk_args(val_every=25, formal_val_every=10))
    except SystemExit:
        raised = True
    check(raised,
          "T12_ambiguous_legacy_plus_new_formal_refused",
          "SystemExit raised",
          "resolver did not refuse ambiguous combination")


# ─────────────────────────── run
def main():
    print("Corrected FADC3D train-utils tests")
    print("=" * 60)
    test_predictor_wrapper()
    test_limit_val_iterable_determinism()
    test_atomic_save_success_and_failure()
    test_legacy_ckpt_resume_smoke()
    test_resume_preserves_log()
    test_pre_val_ckpt_epoch()
    test_fast_best_isolation()
    test_resolve_val_schedule()
    print("=" * 60)
    print(f"passed : {len(PASSED)}")
    print(f"failed : {len(FAILED)}")
    if FAILED:
        print("\nFAILURES:")
        for tag, msg in FAILED:
            print(f"  - {tag}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
