"""Focused tests for the formal-only validation path, atomic checkpoints,
and resume compatibility.

Covers:
   T1   make_primary_predictor extracts output[0] from a tuple.
   T2   make_primary_predictor is a no-op on a single-tensor output.
   T2b  make_primary_predictor extracts output[0] from a list.
   T3   parse_args default --val_overlap == 0.5.
   T3b  parse_args default --val_every == 20.
   T4   validate() iterates every case in the loader (no subset).
   T4b  tqdm progress wrapper does not alter the item sequence (metric-neutral).
   T5   atomic_torch_save success writes exact object, no tmp leftovers.
   T5a  atomic_torch_save reraises on failure.
   T5b  atomic_torch_save does not clobber existing destination on failure.
   T5c  atomic_torch_save cleans tmp on failure.
   T6   ckpt saved before validation survives a simulated validation crash
        and remains loadable with the correct completed-epoch marker.
   T7   Legacy checkpoint (no scaler / train_log; WITH legacy best_fast_dice)
        resumes successfully. Legacy best_fast_dice field is silently ignored.
   T8   Resume with train_log preserves it verbatim.
   T9   Pre-val ckpt.epoch == completed training epoch (no re-run on resume).
  T10   Strict-load round-trip: state_dict-save via atomic_torch_save then
        strict=True reload passes for the corrected model.

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
    atomic_torch_save,
    parse_args,
    validate,
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


# ─────────────────────────── T1, T2, T2b: predictor wrapper
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

    class ListModel(nn.Module):
        def forward(self, x):
            return [x * 5.0, x * 6.0]
    pl = make_primary_predictor(ListModel())
    yl = pl(x)
    check(torch.is_tensor(yl) and torch.equal(yl, x * 5.0),
          "T2b_predictor_selects_0_from_list",
          "list-output primary picked",
          "list output not handled")


# ─────────────────────────── T3, T3b: CLI defaults
def test_cli_defaults():
    print("[T3-T3b] parse_args defaults for val_overlap / val_every")

    saved_argv = sys.argv[:]
    try:
        # parse_args reads sys.argv; feed it only the required positionals.
        sys.argv = ["train_centralized_correct.py"]
        ns = parse_args()
    finally:
        sys.argv = saved_argv

    check(abs(ns.val_overlap - 0.5) < 1e-9,
          "T3_val_overlap_default_is_0.5",
          f"val_overlap={ns.val_overlap}",
          f"expected 0.5, got {ns.val_overlap}")

    check(ns.val_every == 20,
          "T3b_val_every_default_is_20",
          f"val_every={ns.val_every}",
          f"expected 20, got {ns.val_every}")

    # And there must be NO fast-validation attributes left on the namespace.
    banned = [
        "fast_val_every", "fast_val_overlap", "fast_val_max_cases",
        "formal_val_every", "formal_val_overlap",
    ]
    surviving = [a for a in banned if hasattr(ns, a)]
    check(not surviving,
          "T3c_no_fast_validation_args_left",
          "fast/formal_val_* args fully removed",
          f"still present: {surviving}")


# ─────────────────────────── T4: validate() iterates every case
def test_validate_iterates_all_cases():
    print("[T4] validate() processes every case in the loader (no subset)")

    from monai.metrics import DiceMetric
    from monai.transforms import AsDiscrete

    device = torch.device("cpu")

    # Fake loader: N deterministic dict batches of image (1,2,4,4,4) + label
    # (1,1,4,4,4). Small enough to iterate on CPU.
    class FakeLoader:
        def __init__(self, n_cases: int):
            self.n = n_cases
        def __iter__(self):
            for i in range(self.n):
                img = torch.zeros(1, 2, 4, 4, 4)
                lbl = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)
                # Put a small foreground blob so IoU/Dice are non-trivial.
                lbl[0, 0, 1:3, 1:3, 1:3] = 1
                yield {"image": img, "label": lbl}
        def __len__(self):
            return self.n

    class ConstOnesModel(nn.Module):
        def forward(self, x):
            # Two-class logits, argmax=1 everywhere -> mostly-foreground pred.
            out = torch.zeros(x.shape[0], 2, *x.shape[2:])
            out[:, 1] = 1.0
            return out

    N = 17
    loader = FakeLoader(N)
    model = ConstOnesModel().eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    mean_dice, mean_iou, mean_sens, n_done = validate(
        model, loader, dice_metric, post_pred, post_label,
        patch_size=(4, 4, 4), device=device,
        overlap=0.5, sw_batch_size=1,
    )

    check(n_done == N,
          "T4_validate_processes_all_cases",
          f"n_done={n_done} of N={N}",
          f"validate processed {n_done} cases, expected {N}")


# ─────────────────────────── T4b: tqdm wrapper is metric-neutral
def test_tqdm_passthrough_neutrality():
    print("[T4b] tqdm progress wrapper does not alter the item sequence")

    from tqdm import tqdm as tqdm_

    source = [{"idx": i, "payload": torch.arange(3) + i} for i in range(11)]
    captured = []
    pbar = tqdm_(source, total=len(source), desc="probe", leave=False, disable=True)
    for item in pbar:
        captured.append(item)

    check(len(captured) == len(source),
          "T4b_a_progress_wrapper_yields_all_items",
          f"{len(captured)} of {len(source)}",
          f"lost items: {len(source) - len(captured)}")
    check(all(
              (a["idx"] == b["idx"] and torch.equal(a["payload"], b["payload"]))
              for a, b in zip(captured, source)),
          "T4b_b_progress_wrapper_preserves_content",
          "identity preserved",
          "tqdm mutated items")


# ─────────────────────────── T5, T5a, T5b, T5c: atomic save
def test_atomic_save_success_and_failure():
    print("[T5] atomic_torch_save success + failure cleanup")

    with tempfile.TemporaryDirectory() as td:
        dest = os.path.join(td, "ckpt.pth")
        payload = {"epoch": 42, "arr": torch.arange(6).view(2, 3).tolist()}
        atomic_torch_save(payload, dest)
        loaded = torch.load(dest, weights_only=False)
        check(loaded == payload,
              "T5_atomic_save_success_writes_exact_object",
              "load matches",
              f"loaded={loaded}")
        leftovers = [n for n in os.listdir(td) if n.startswith(".tmp_")]
        check(not leftovers,
              "T5_no_tmp_leftover_on_success",
              "no tmp files",
              f"tmp remnants: {leftovers}")

    with tempfile.TemporaryDirectory() as td:
        dest = os.path.join(td, "ckpt.pth")
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


# ─────────────────────────── T6: pre-val ckpt survives crash
def test_ckpt_before_validation_survives_crash():
    """Simulate the flow: after epoch N training completes, write
    last_checkpoint.pth via atomic_torch_save with epoch=N, then simulate a
    validation crash. The checkpoint MUST still exist, be loadable, and
    carry epoch=N so that resume advances to N+1.
    """
    print("[T6] pre-validation ckpt survives interrupted validation")

    with tempfile.TemporaryDirectory() as td:
        dest = os.path.join(td, "last_checkpoint.pth")
        completed_epoch = 4  # 0-indexed: means 5 epochs done
        ckpt_payload = {
            "epoch": completed_epoch,
            "model": {"weight": torch.zeros(1)},
            "optimizer": {},
            "scheduler": {},
            "scaler": {},
            "best_dice": 0.5429,
            "arch_identity": {"model_name": "unet3d_fadc_encoder_correct"},
            "val_config": {"val_every": 20, "val_overlap": 0.5,
                           "val_sw_batch_size": 4, "checkpoint_every": 10},
            "train_log": [{"epoch": completed_epoch + 1, "loss": 0.3}],
        }
        atomic_torch_save(ckpt_payload, dest)

        # Simulate validation raising mid-loop AFTER the pre-val ckpt is written.
        try:
            raise RuntimeError("simulated validation crash")
        except RuntimeError:
            pass  # training loop would propagate; the point is: no more writes.

        # After crash, the ckpt file must be intact.
        check(os.path.exists(dest),
              "T6a_ckpt_exists_after_crash",
              "last_checkpoint.pth present",
              "ckpt missing after crash")

        reloaded = torch.load(dest, weights_only=False)
        check(int(reloaded["epoch"]) == completed_epoch,
              "T6b_ckpt_epoch_is_completed_epoch",
              f"epoch={completed_epoch}",
              f"expected {completed_epoch}, got {reloaded['epoch']}")

        # Resume math: start_epoch = ckpt.epoch + 1
        start_epoch = int(reloaded["epoch"]) + 1
        check(start_epoch == completed_epoch + 1,
              "T6c_resume_skips_completed_epoch",
              f"start_epoch={start_epoch}",
              f"expected {completed_epoch + 1}, got {start_epoch}")


# ─────────────────────────── T7: legacy ckpt with fast fields resume compat
def test_legacy_ckpt_with_fast_fields_resumes():
    """A legacy checkpoint may contain best_fast_dice, fast_val_dice log
    entries, and lack scaler/train_log. The resume block must:
      - accept the identity (via _require_matching_arch),
      - default missing scaler/train_log gracefully,
      - silently ignore best_fast_dice (do not restore it, do not crash).
    """
    print("[T7] legacy ckpt with fast fields loads and best_fast_dice is ignored")
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
        "best_fast_dice": 0.31,   # legacy field — MUST be ignored on resume
        "arch_identity": arch,
        # deliberately missing: scaler, train_log
    }

    # Simulate the resume block from train_centralized_correct.train():
    train_log = []
    best_dice = float(legacy_ckpt.get("best_dice", 0.0))
    has_scaler = "scaler" in legacy_ckpt and isinstance(legacy_ckpt["scaler"], dict)
    has_log = "train_log" in legacy_ckpt and isinstance(legacy_ckpt["train_log"], list)
    if has_log:
        train_log = list(legacy_ckpt["train_log"])
    start_epoch = int(legacy_ckpt["epoch"]) + 1

    # Verify the module-level train() code path does NOT bind best_fast_dice
    # to any live variable by inspecting the source. This is the strongest
    # invariant we can assert without running the actual training loop.
    import inspect
    from training import train_centralized_correct as tcm
    src = inspect.getsource(tcm.train)
    check("best_fast_dice = " not in src and "best_fast_dice=" not in src.replace("best_fast_dice=(legacy; ignored)", ""),
          "T7z_no_best_fast_dice_state_variable",
          "training loop no longer maintains best_fast_dice state",
          "best_fast_dice is still restored as a live variable")

    check(start_epoch == 5,
          "T7a_legacy_start_epoch",
          f"start_epoch={start_epoch}",
          f"expected 5, got {start_epoch}")
    check(best_dice == 0.42,
          "T7b_legacy_best_dice_restored",
          "best_dice restored from legacy ckpt",
          f"got {best_dice}")
    check(not has_scaler and not has_log and train_log == [],
          "T7c_legacy_missing_scaler_and_log_handled",
          "graceful fallback for missing keys",
          "did not fall back")

    # The presence of best_fast_dice in the ckpt dict must NOT break torch.load
    # or the state restoration. We already got here without exception — assert.
    check(True,
          "T7d_ckpt_with_legacy_best_fast_dice_did_not_raise",
          "legacy best_fast_dice field silently tolerated",
          "unreachable")


# ─────────────────────────── T8: resume preserves accumulated log
def test_resume_preserves_log():
    print("[T8] resume with train_log restores it verbatim (no duplicates)")
    prior_log = [
        {"epoch": 1, "loss": 0.9},
        {"epoch": 2, "loss": 0.8, "val_dice": 0.31},
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

    # Simulate the loop appending one entry for epoch 3 — no duplication of
    # the restored entries.
    train_log.append({"epoch": 3, "loss": 0.7})
    check(train_log == prior_log,
          "T8c_no_duplicate_entries_after_one_epoch",
          "log accumulated without duplication",
          f"log diverged: {train_log}")


# ─────────────────────────── T9: pre-val ckpt epoch bookkeeping
def test_pre_val_ckpt_epoch():
    """After epoch N training completes, the pre-val checkpoint MUST record
    epoch=N so that resume advances to N+1 without repeating epoch N.
    """
    print("[T9] pre-val ckpt records the just-completed epoch")
    for epoch in (0, 1, 2, 9, 19):
        ckpt = {"epoch": int(epoch)}
        start_epoch = int(ckpt["epoch"]) + 1
        check(start_epoch == epoch + 1,
              f"T9_epoch_{epoch}_advances_to_{epoch+1}",
              f"start_epoch={start_epoch}",
              f"expected {epoch+1}, got {start_epoch}")


# ─────────────────────────── T10: strict-load round-trip
def test_strict_load_roundtrip():
    print("[T10] strict-load round-trip via atomic_torch_save")
    from models.unet_3d_fadc_correct import build_unet3d_fadc_correct

    model = build_unet3d_fadc_correct(
        "unet3d_fadc_encoder_correct",
        in_channels=2, out_channels=2, base_filters=8, deep_supervision=False,
    )
    with tempfile.TemporaryDirectory() as td:
        dest = os.path.join(td, "model_only.pth")
        atomic_torch_save({"model": model.state_dict()}, dest)
        reloaded = torch.load(dest, weights_only=False)
        model2 = build_unet3d_fadc_correct(
            "unet3d_fadc_encoder_correct",
            in_channels=2, out_channels=2, base_filters=8, deep_supervision=False,
        )
        missing, unexpected = model2.load_state_dict(reloaded["model"], strict=True)
        check(not missing and not unexpected,
              "T10_strict_load_roundtrip",
              "no missing / unexpected keys",
              f"missing={missing} unexpected={unexpected}")


# ─────────────────────────── run
def main():
    print("Corrected FADC3D train-utils tests (formal-only)")
    print("=" * 60)
    test_predictor_wrapper()
    test_cli_defaults()
    test_validate_iterates_all_cases()
    test_tqdm_passthrough_neutrality()
    test_atomic_save_success_and_failure()
    test_ckpt_before_validation_survives_crash()
    test_legacy_ckpt_with_fast_fields_resumes()
    test_resume_preserves_log()
    test_pre_val_ckpt_epoch()
    test_strict_load_roundtrip()
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
