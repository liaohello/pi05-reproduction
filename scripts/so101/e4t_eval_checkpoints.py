#!/usr/bin/env python3
"""E4-T / T2 driver: evaluate every completed checkpoint of a running E4-T training run.

Completion is decided from the checkpoint directory itself (Orbax writes ``params/_METADATA`` and
``train_state/_METADATA`` only when a save commits), never from a directory name alone.  The
authoritative optimizer step is the Orbax directory index plus one, with the step-0 reference
saved as index 0; this is the same relation scripts/so101/read_train_state.py asserts, and it is
re-checked here against the training log so an evaluation can never be attributed to the wrong step.

Nothing here touches MuJoCo: each evaluation is the open-loop prediction job in
scripts/so101/evaluate_e4t_openloop.py.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "deploy"))
sys.path.insert(0, str(HERE))
import common as deploy  # noqa: E402
import so101_common as common  # noqa: E402

REPO = common.REPO_ROOT
COMPLETE_MARKERS = ("params/_METADATA", "train_state/_METADATA", "_CHECKPOINT_METADATA",
                    "assets/so101/norm_stats.json")


def completed_checkpoints(root: Path) -> list[tuple[int, Path]]:
    """Every numeric Orbax directory under root whose committed save is complete."""
    found = []
    for candidate in root.rglob("*"):
        if not candidate.is_dir() or not candidate.name.isdigit():
            continue
        if all((candidate / marker).is_file() for marker in COMPLETE_MARKERS):
            found.append((int(candidate.name), candidate))
    # Directory-name sorting is lexical (e.g. "5000" comes after "25000").
    # Keep checkpoint iteration and any caller selecting the final save in numeric order.
    return sorted(found, key=lambda item: (item[0], str(item[1])))


def logged_steps(run_dir: Path) -> set[int]:
    """Optimizer steps the training process itself has reported, used to reject a premature save."""
    steps: set[int] = set()
    for log in run_dir.glob("train_*.log"):
        text = log.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"(?:step|global_step)\D{0,3}(\d{1,7})", text, re.IGNORECASE):
            steps.add(int(match.group(1)))
        # the progress bar prints "441it/30.0kit", which names the same counter differently
        for match in re.finditer(r"(\d{1,7})it/", text):
            steps.add(int(match.group(1)))
    return steps


def expected_step(directory_index: int) -> int:
    """Orbax directory index to authoritative TrainState.step.

    openpi saves after applying the update, so directory d holds step d+1. Measured three times: the
    batch-8 probe saved directory 0 with TrainState.step 1, the E2 run saved directory 99 with
    TrainState.step 100, and this round's 30k run saved directory 5000 with TrainState.step 5001
    (runs/so101/e4t/train_state_dir5000.json, read back by scripts/so101/read_train_state.py).  The
    loop index is zero-based and skips step 0 when applying save_interval, so an intermediate
    checkpoint lands one update after each multiple of the interval; the final save still carries the
    targeted update count.  ``scripts/so101/read_train_state.py`` re-checks this on the real store.
    """
    return directory_index + 1


def evaluation_label(step: int, prefix: str = "") -> str:
    """Give each split a distinct output label while preserving validation's old names."""
    return f"{prefix}step{int(step):06d}"


def already_evaluated(output: Path, label: str) -> bool:
    report = output / label / "report.json"
    if not report.is_file():
        return False
    document = common.load_json(report)
    return bool(document.get("success", int(document.get("windows", 0) or 0) > 0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="training run directory that holds checkpoints/ and config_initial.json")
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True, help="external python for the launcher")
    parser.add_argument("--gpu-index", type=int, required=True,
                        help="card the evaluation containers are admitted on (may differ from training)")
    parser.add_argument("--eval-split", default="validation", choices=("train", "validation"))
    parser.add_argument("--label-prefix", default=None,
                        help="prefix output labels to keep train and validation reports separate; "
                             "train defaults to 'train_'")
    parser.add_argument("--windows", type=Path, default=None,
                        help="override the frozen window file (default: <view>/e4t_windows.json)")
    parser.add_argument("--run-config", type=Path, default=None,
                        help="override the run config the checkpoints were trained with "
                             "(default: <run-dir>/config_initial.json)")
    parser.add_argument("--noise-copies", type=int, default=3)
    parser.add_argument("--required-memory-mib", type=int, default=12000)
    parser.add_argument("--memory-margin-mib", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="only evaluate checkpoints whose authoritative step is at most this")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many evaluations")
    parser.add_argument("--once", action="store_true", help="one scan, no polling")
    parser.add_argument("--scan-interval", type=int, default=300)
    parser.add_argument("--stop-when-run-finishes", action="store_true",
                        help="stop scanning once run_initial.json exists (the training job reported)")
    parser.add_argument("--reference-checkpoint", type=Path, default=None,
                        help="optional initialisation checkpoint to score before the saved run checkpoints")
    parser.add_argument("--reference-run-config", type=Path, default=None,
                        help="saved config bound to --reference-checkpoint")
    parser.add_argument("--reference-label", default=None,
                        help="output label for --reference-checkpoint (defaults to <prefix>init_stepNNNNNN)")
    parser.add_argument("--reference-step", type=int, default=1,
                        help="authoritative TrainState.step of --reference-checkpoint")
    parser.add_argument("--reference-exp-name", default="e4t_30k_state_on_b8")
    parser.add_argument("--dry-run", action="store_true", help="print the planned jobs and launch nothing")
    args = parser.parse_args()

    if (args.reference_checkpoint is None) != (args.reference_run_config is None):
        raise SystemExit("--reference-checkpoint and --reference-run-config must be supplied together")
    label_prefix = args.label_prefix
    if label_prefix is None:
        label_prefix = "train_" if args.eval_split == "train" else ""

    run_dir = args.run_dir.expanduser().resolve()
    view = args.view.expanduser().resolve()
    assets = deploy.load_assets()
    optimisation = common.load_json(REPO / "configs/so101/e4t_training.json")["optimisation"]
    run_config = (args.run_config or run_dir / "config_initial.json").expanduser().resolve()
    windows = (args.windows or view / "e4t_windows.json").expanduser().resolve()
    for required in (run_config, windows, view / "meta.json"):
        if not required.is_file():
            raise SystemExit(f"missing input: {required}")
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        raise SystemExit(f"no checkpoint tree under {checkpoint_root}")

    manifest_path = run_dir / f"evaluation_{args.eval_split}_driver.json"
    results = []
    if manifest_path.is_file():
        results = common.load_json(manifest_path).get("evaluations", [])
    # a failed evaluation is retried on the next scan; only a completed one closes a label, so an
    # admission failure while another job holds the card cannot silently drop a checkpoint
    done_labels = {row["label"] for row in results if row.get("success")}
    print(json.dumps({"checkpoint_root": str(checkpoint_root), "run_config": str(run_config),
                      "windows": str(windows), "gpu_index": args.gpu_index,
                      "eval_split": args.eval_split, "label_prefix": label_prefix,
                      "resume_labels": sorted(done_labels)}, indent=2), flush=True)

    if args.reference_checkpoint is not None:
        reference_checkpoint = args.reference_checkpoint.expanduser().resolve()
        reference_config = args.reference_run_config.expanduser().resolve()
        if not reference_checkpoint.is_dir() or not reference_config.is_file():
            raise SystemExit("reference checkpoint/configuration is missing")
        reference_label = args.reference_label or (
            f"{label_prefix}init_step{int(args.reference_step):06d}")
        if not already_evaluated(run_dir / "evaluation", reference_label):
            command = [str(args.python), "scripts/so101/e4t_run_training.py", "--mode", "eval",
                       "--run-dir", str(run_dir), "--train-view", str(view),
                       "--base-checkpoint", assets["checkpoint"], "--tokenizer", assets["tokenizer"],
                       "--exp-name", args.reference_exp_name,
                       "--batch-size", str(optimisation["primary_batch_size"]),
                       "--steps", str(optimisation["planned_updates"]),
                       "--decay-steps", str(optimisation["decay_steps"]),
                       "--gpu-index", str(args.gpu_index),
                       "--required-memory-mib", str(args.required_memory_mib),
                       "--memory-margin-mib", str(args.memory_margin_mib),
                       "--eval-checkpoint", str(reference_checkpoint),
                       "--eval-run-config", str(reference_config),
                       "--eval-windows", str(windows), "--eval-split", args.eval_split,
                       "--eval-label", reference_label,
                       "--eval-train-state-step", str(args.reference_step),
                       "--noise-copies", str(args.noise_copies),
                       "--eval-assets-root", str(view / "assets")]
            process = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
            report = run_dir / "evaluation" / reference_label / "report.json"
            row = {"label": reference_label, "authoritative_step": int(args.reference_step),
                   "reference": True, "returncode": process.returncode,
                   "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "success": False}
            if report.is_file():
                document = common.load_json(report)
                row["success"] = bool(document.get("success", int(document.get("windows", 0) or 0) > 0))
                row["report_sha256"] = common.sha256_file(report)
                row["recorded_step"] = document.get("checkpoint_train_state_step")
                if row["recorded_step"] != int(args.reference_step):
                    row["success"] = False
                    row["problem"] = "reference report has a different TrainState.step"
            else:
                row["problem"] = "reference evaluation did not produce report.json"
            results.append(row)
            common.write_json(manifest_path, {
                "schema_version": 1, "branch": "so101", "phase": "E3",
                "substage": "E4-T-T2-openloop", "split": args.eval_split,
                "checkpoint_root": str(checkpoint_root),
                "run_config_sha256": common.sha256_file(run_config),
                "windows_sha256": common.sha256_file(windows),
                "view_sha256": common.load_json(view / "meta.json")["view_sha256"],
                "evaluations": results, "no_learned_action_executed": True})
            print(json.dumps({"reference": reference_label, "step": args.reference_step,
                              "success": row["success"]}, indent=2), flush=True)
            if not row["success"]:
                return 1

    launched = 0
    while True:
        reported = logged_steps(run_dir)
        batch = completed_checkpoints(checkpoint_root)
        parents = sorted({checkpoint.parent for _, checkpoint in batch})
        if len(parents) > 1:
            raise SystemExit(f"completed checkpoints belong to several experiments: {parents}")
        exp_name = parents[0].name if parents else ""
        for directory_index, checkpoint in batch:
            step = expected_step(directory_index)
            if args.max_steps and step > args.max_steps:
                continue
            # A save at loop step d is logged as step d, while the committed TrainState holds d+1.
            # The final save is therefore the one checkpoint whose step exceeds every logged step,
            # and it must not be dropped by the premature-save guard.
            if reported and step > max(reported) + 1:
                print(f"skip {checkpoint.name}: no training log line past step {step} yet", flush=True)
                continue
            label = evaluation_label(step, label_prefix)
            if label in done_labels or already_evaluated(run_dir / "evaluation", label):
                continue
            command = [str(args.python), "scripts/so101/e4t_run_training.py", "--mode", "eval",
                       "--run-dir", str(run_dir), "--train-view", str(view),
                       "--base-checkpoint", assets["checkpoint"], "--tokenizer", assets["tokenizer"],
                       "--exp-name", exp_name,
                       "--batch-size", str(optimisation["primary_batch_size"]),
                       "--steps", str(optimisation["planned_updates"]),
                       "--decay-steps", str(optimisation["decay_steps"]),
                       "--gpu-index", str(args.gpu_index),
                       "--required-memory-mib", str(args.required_memory_mib),
                       "--memory-margin-mib", str(args.memory_margin_mib),
                       "--eval-checkpoint", str(checkpoint),
                       "--eval-run-config", str(run_config),
                       "--eval-windows", str(windows),
                       "--eval-split", args.eval_split, "--eval-label", label,
                       "--eval-train-state-step", str(step),
                       "--noise-copies", str(args.noise_copies)]
            print(json.dumps({"launch": label, "checkpoint": str(checkpoint),
                              "directory_index": directory_index, "authoritative_step": step,
                              "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}),
                  flush=True)
            if args.dry_run:
                print(json.dumps({"dry_run": True, "label": label,
                                  "argv": [value if value not in (assets["checkpoint"],
                                                                  assets["tokenizer"]) else "<asset>"
                                           for value in command]}, indent=2), flush=True)
                launched += 1
                continue
            process = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
            row = {"label": label, "directory_index": directory_index,
                   "authoritative_step": step, "checkpoint": str(checkpoint),
                   "returncode": process.returncode,
                   "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            report = run_dir / "evaluation" / label / "report.json"
            if report.is_file():
                document = common.load_json(report)
                row["success"] = bool(document.get("success",
                                                 int(document.get("windows", 0) or 0) > 0))
                row["report_sha256"] = common.sha256_file(report)
                row["checkpoint_name"] = document.get("checkpoint_name")
                row["recorded_step"] = document.get("checkpoint_train_state_step")
                if row.get("recorded_step") != step:
                    row["success"] = False
                    row["problem"] = f"report attributes step {row.get('recorded_step')}, expected {step}"
            else:
                row["success"] = False
                row["problem"] = "no report.json produced"
                row["stderr_tail"] = (process.stderr or "")[-2000:]
            results.append(row)
            common.write_json(manifest_path, {
                "schema_version": 1, "branch": "so101", "phase": "E3",
                "substage": "E4-T-T2-openloop", "split": args.eval_split,
                "checkpoint_root": str(checkpoint_root), "run_config_sha256": common.sha256_file(run_config),
                "windows_sha256": common.sha256_file(windows),
                "view_sha256": common.load_json(view / "meta.json")["view_sha256"],
                "evaluations": results,
                "no_learned_action_executed": True})
            print(json.dumps(row, indent=2), flush=True)
            launched += 1
            if args.limit and launched >= args.limit:
                return 0
        if args.once:
            return 0
        if args.stop_when_run_finishes and (run_dir / "run_initial.json").is_file():
            return 0
        time.sleep(max(30, args.scan_interval))


if __name__ == "__main__":
    raise SystemExit(main())
