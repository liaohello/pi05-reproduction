#!/usr/bin/env python3
"""E4-T T1 evidence: rebuild the training record from the artifacts the run itself produced.

Every number in the output is read from a file under the run directory, so the record can be
regenerated after the fact instead of transcribed: the run-config snapshot the containers were
launched from, the exact-owner registry, the launcher JSON blocks, the per-checkpoint open-loop
reports and the learning-rate trace.  Narrative fields that describe a decision rather than a
measurement (why a segment was replaced, what the wall-limit amendment says) are carried over from
the previous version of the same file, passed with --narrative.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e4t_eval_checkpoints as driver  # noqa: E402
import so101_common as common  # noqa: E402

NARRATIVE_KEYS = ("throughput_samples", "wall_limit", "sampling", "learning_rate_trace_note",
                  "claim")


def launcher_blocks(path: Path) -> list[dict]:
    """The container records the launcher prints as one JSON object per run."""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    out, depth, start = [], 0, None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunk = text[start:index + 1]
                try:
                    out.append(json.loads(chunk))
                except json.JSONDecodeError:
                    pass
                start = None
    return [row for row in out if row.get("exact_name")]


def strip_mount(value):
    """Publish repo-relative paths: the containers see the same tree mounted at /work."""
    if isinstance(value, str):
        return value[len("/work/"):] if value.startswith("/work/") else value
    if isinstance(value, list):
        return [strip_mount(item) for item in value]
    if isinstance(value, dict):
        return {key: strip_mount(item) for key, item in value.items()}
    return value


PUBLIC_LAUNCH_FLAGS = {
    "state-mode", "batch-size", "steps", "save-interval", "data-loader-workers",
    "lr-warmup-steps", "peak-lr", "decay-steps", "decay-lr", "dataset-label",
}


def public_gpu_metrics(gpu: dict | None) -> dict | None:
    """Keep run-level memory evidence without publishing device identity or capacity."""
    if not isinstance(gpu, dict):
        return None
    return {key: gpu[key] for key in (
        "peak_mib_sampled", "jax_memory_fraction", "utilisation_is_not_an_admission_gate",
    ) if key in gpu}


def public_container_record(row: dict) -> dict:
    """Summarise a launch; exact identity remains in the ignored registry."""
    result = {key: row[key] for key in (
        "mode", "exit_code", "wall_time_s", "steps", "batch_size", "stop_method",
    ) if key in row}
    for key in ("role", "evaluation_label", "owner_verified_at_registration"):
        if key in row:
            result[key] = row[key]
    result["gpu"] = public_gpu_metrics(row.get("gpu"))
    return result


def resource_role(command: list[str]) -> str:
    if "--mode" in command:
        index = command.index("--mode")
        if index + 1 < len(command) and command[index + 1] == "probe":
            return "memory_probe"
    joined = " ".join(map(str, command))
    if "train_so101.py" in joined:
        return "training"
    if "evaluate_e4t_openloop.py" in joined:
        return "openloop_evaluation"
    return "other"


def command_value(command: list[str], flag: str) -> str | None:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError):
        return None


def registered_container_records(run: Path, registry: list[dict]) -> list[tuple[str, dict]]:
    """Read each launch's own payload, falling back to the legacy latest-run file by run id."""
    records = []
    for entry in registry:
        run_id = entry.get("run_id")
        exact_name = str(entry.get("exact_name", ""))
        mode = exact_name.removeprefix("so101-e4t-").split("-", 1)[0]
        if not run_id or mode not in {"smoke", "probe", "initial", "eval"}:
            continue
        candidates = (run / f"run_{mode}_{run_id}.json", run / f"run_{mode}.json")
        payload = None
        for path in candidates:
            if not path.is_file():
                continue
            try:
                document = common.load_json(path)
                if (document.get("run_id") == run_id
                        or document.get("exact_name") == exact_name):
                    payload = document
                    break
            except (OSError, ValueError):
                continue
        if payload is None:
            continue
        command = entry.get("command") or []
        label = command_value(command, "--eval-label") or command_value(command, "--label")
        payload["role"] = resource_role(command)
        payload["owner_verified_at_registration"] = bool(entry.get("matched_all_registration_fields"))
        if label:
            payload["evaluation_label"] = label
        records.append((str(run_id), payload))
    return records


def split_curve(reports: dict[str, dict], split: str) -> list[dict]:
    """Build an action-error curve from one split's frozen-plan reports only."""
    selected = [(key, value) for key, value in reports.items()
                if value.get("split") == split and value.get("window_set") == "frozen_plan"]
    selected.sort(key=lambda item: (item[1].get("checkpoint_train_state_step", -1), item[0]))
    points = []
    for key, report in selected:
        aggregates = report.get("aggregates") or {}
        clean = aggregates.get("clean_active") or {}
        recovery = aggregates.get("recovery_post_active") or {}
        points.append({
            "label": key,
            "train_state_step": report.get("checkpoint_train_state_step"),
            "run_config_sha256": report.get("run_config_sha256"),
            "windows": report.get("windows"),
            "windows_expected": report.get("windows_expected"),
            "noise_set_sha256": report.get("noise_set_sha256"),
            "clean_active": {metric: clean.get(metric) for metric in (
                "joint_macro_mae_deg", "joint_p95_deg", "gripper_macro_mae_pct",
                "true12_mse", "padding20_mse")},
            "recovery_post_active": {metric: recovery.get(metric) for metric in (
                "joint_macro_mae_deg", "joint_p95_deg", "gripper_macro_mae_pct",
                "true12_mse", "padding20_mse")},
            "per_category_post_active": {
                category: (aggregates.get(f"recovery_{category}_post_active") or {}).get(
                    "joint_macro_mae_deg") for category in ("A", "B", "C")},
            "horizon_mae_joint_deg": report.get("horizon_mae_joint_deg"),
        })
    return points


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/e4t_protocol.json")
    parser.add_argument("--training-config", type=Path,
                        default=common.REPO_ROOT / "configs/so101/e4t_training.json")
    parser.add_argument("--narrative", type=Path, default=None,
                        help="previous version of this file; its decision records are carried over")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = args.run_dir.expanduser().resolve()
    view = args.view.expanduser().resolve()
    config = common.load_json(run / "config_initial.json")
    meta = common.load_json(view / "meta.json")
    registry = common.load_json(run / "resource_registry.json").get("entries", [])
    narrative = common.load_json(args.narrative) if args.narrative and args.narrative.is_file() else {}

    # Keep the launcher blocks for older attempts, and recover new per-launch resource samples from
    # the exact-owner registry's matching run-id payload. Latest-run aliases are accepted only when
    # their run_id matches, so parallel evals cannot lend each other a memory peak.
    launch_records: dict[str, dict] = {}
    for log in sorted(run.glob("t1_launch*.log")) + sorted(run.glob("init_ref*.log")) \
            + sorted(run.glob("init_anchored*.log")):
        for row in launcher_blocks(log):
            identity = str(row.get("run_id") or row.get("exact_name") or f"legacy-{len(launch_records)}")
            launch_records[identity] = row
    for identity, row in registered_container_records(run, registry):
        launch_records[identity] = row
    containers = list(launch_records.values())

    def container_label(row: dict):
        command = row.get("command") or []
        return command[command.index("--label") + 1] if "--label" in command else None

    by_label = {container_label(row): row for row in containers if container_label(row)}
    by_label.update({row["evaluation_label"]: row for row in containers
                     if row.get("evaluation_label")})
    # the registry keeps the exact argv and env of every container this round started, so the
    # effective knobs are read from there rather than from a log the launcher only writes on exit
    initial_entries = sorted((row for row in registry
                              if str(row.get("exact_name", "")).startswith("so101-e4t-initial-")),
                             key=lambda row: row["exact_name"])
    launch_flags: dict[str, str] = {}
    xla_fraction = None
    if initial_entries:
        command = initial_entries[-1].get("command") or []
        for index, token in enumerate(command):
            if token.startswith("--") and index + 1 < len(command) \
                    and not command[index + 1].startswith("--"):
                value = command[index + 1]
                # the containers see the repo at /work; publish the repo-relative form, not a mount path
                launch_flags[token.lstrip("-")] = value[len("/work/"):] if value.startswith("/work/") else value
        for token in (initial_entries[-1].get("launch_args") or []):
            if str(token).startswith("XLA_PYTHON_CLIENT_MEM_FRACTION="):
                xla_fraction = float(str(token).split("=", 1)[1])
    public_launch_flags = {key: value for key, value in launch_flags.items()
                           if key in PUBLIC_LAUNCH_FLAGS}
    reports = {}
    keep = ("windows", "episodes", "joint_macro_mae_deg", "joint_p95_deg",
            "joint_p95_episode_macro_deg", "gripper_macro_mae_pct",
            "hold_joint_mae_deg", "train_mean_joint_mae_deg", "pred_delta_rms_deg_per_frame",
            "gt_delta_rms_deg_per_frame", "pred_reversals_per_window",
            "left_joint_mae_deg", "right_joint_mae_deg", "left_gripper_mae_pct",
            "right_gripper_mae_pct", "action12_mse", "true12_mse", "padding20_mse",
            "per_copy_joint_macro_mae_deg", "per_copy_gripper_macro_mae_pct")
    for path in sorted((run / "evaluation").glob("*/report.json")):
        if "superseded" in path.parent.name:
            continue  # re-scored under a later methodology; never part of the published record
        report = common.load_json(path)
        label = report.get("label")
        window_set = report.get("window_set") or "frozen_plan"
        per_copy = {}
        for candidate in (run / "per_copy" / f"{path.parent.name}.json",
                          path.parent / "per_copy.json"):
            if candidate.is_file():
                per_copy = common.load_json(candidate)
                break
        anchored = {}
        for candidate in (run / "per_copy" / f"{path.parent.name}_anchored.json",
                          path.parent.parent / f"{path.parent.name}_anchored" / "per_copy.json"):
            if candidate.is_file():
                anchored = common.load_json(candidate)
                break
        timing = ((anchored.get("event_timing") or per_copy.get("event_timing")
                   or report.get("event_timing") or {}))
        timing_instrument = ("event_anchored_per_copy" if anchored.get("event_timing")
                             else "frozen_plan_per_copy" if per_copy.get("event_timing")
                             else "frozen_plan_mean_of_copies")
        aggregates = (per_copy.get("aggregates") or report.get("aggregates") or {})
        reports[f"{label}@{window_set}"] = {
            "report": str(path.relative_to(common.REPO_ROOT)),
            "report_sha256": common.sha256_file(path),
            "split": report.get("split"), "window_set": window_set,
            "checkpoint_train_state_step": report.get("checkpoint_train_state_step"),
            "run_config_sha256": report.get("run_config_sha256"),
            "view_sha256": report.get("view_sha256"),
            "windows": report.get("windows"), "windows_expected": report.get("windows_expected"),
            "success": report.get("success"), "noise_copies": report.get("noise_copies"),
            "noise_set_sha256": report.get("noise_set_sha256"),
            "admission": public_gpu_metrics(by_label.get(label, {}).get("gpu")),
            "metric_source": ("per_noise_copy_episode_equal" if per_copy
                              else "mean_of_copies_legacy"),
            "event_timing_instrument": timing_instrument,
            "event_timing": {key: value for key, value in timing.items()
                             if key not in ("unpaired_examples", "per_copy")},
            "aggregates": {group: {key: value for key, value in row.items() if key in keep}
                           for group, row in aggregates.items()
                           if group in ("all", "clean", "clean_active", "clean_static",
                                        "recovery", "recovery_pre_active",
                                        "recovery_post_active", "recovery_post_static",
                                        "recovery_A", "recovery_A_post_active",
                                        "recovery_B", "recovery_B_post_active", "recovery_C",
                                        "recovery_C_post_active")},
            "legacy_mean_of_copies": {
                "aggregates": {group: {key: value for key, value in row.items() if key in keep}
                               for group, row in (report.get("aggregates") or {}).items()
                               if group in ("clean_active", "recovery_post_active")},
                "event_timing": {key: value for key, value in
                                 (report.get("event_timing") or {}).items()
                                 if key not in ("unpaired_examples", "per_copy")},
            },
        }
    driver_manifests = {}
    for split in ("validation", "train"):
        driver_manifest = run / f"evaluation_{split}_driver.json"
        if driver_manifest.is_file():
            driver_data = common.load_json(driver_manifest)
            driver_manifests[split] = {
                "artifact": str(driver_manifest.relative_to(common.REPO_ROOT)),
                "artifact_sha256": common.sha256_file(driver_manifest),
                "split": driver_data.get("split"), "view_sha256": driver_data.get("view_sha256"),
                "run_config_sha256": driver_data.get("run_config_sha256"),
                "windows_sha256": driver_data.get("windows_sha256"),
                "evaluations": [{key: row.get(key) for key in
                                 ("label", "authoritative_step", "success", "report_sha256",
                                  "recorded_step", "reference") if key in row}
                                for row in driver_data.get("evaluations", [])],
            }
    raw_evaluations = (common.load_json(run / "evaluation_validation_driver.json").get("evaluations", [])
                       if (run / "evaluation_validation_driver.json").is_file() else [])
    # the driver manifest carries the checkpoint's absolute path and, for a failed attempt, a
    # traceback with the same mount path; neither is public evidence
    evaluations = [{key: row.get(key) for key in
                    ("label", "authoritative_step", "success", "report_sha256", "recorded_step",
                     "reference", "returncode", "problem") if key in row}
                   for row in raw_evaluations]

    checkpoints = driver.completed_checkpoints(run / "checkpoints")
    log = run / "train_initial.log"
    steps: list[int] = []
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
        steps = [int(m) for m in re.findall(r"(?:^|\n)Step (\d+):", text)]
    gates = common.load_json(args.training_config)["validation_gate"]

    run_record_path = run / "run_initial.json"
    run_record = common.load_json(run_record_path) if run_record_path.is_file() else {}
    document = {
        "schema_version": 2, "branch": "so101", "phase": "E3", "substage": "E4-T-T1-training",
        "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "rebuilt_from": "config_initial.json, resource_registry.json, launcher logs, the evaluation "
                        "driver manifest and every evaluation report.json",
        "run": {
            "exp_name": config["exp_name"], "mode": config["mode"], "state_mode": config["state_mode"],
            "batch_size": config["batch_size"], "steps": config["steps"],
            "save_interval": config["save_interval"], "lr_schedule": config["lr_schedule"],
            "data_loader_workers": launch_flags.get("data-loader-workers"),
            "seed": {"value": common.load_json(args.training_config)["optimisation"]["seed"],
                     "configured_in": "configs/so101/e4t_training.json optimisation.seed",
                     "applied_by": ("the sampler call in scripts/so101/train_so101.py passes this "
                                    "value as a literal, which is how the two stay equal; the run "
                                    "config snapshot does not carry it")},
            "launch_flags": public_launch_flags,
            "jax_memory_fraction_at_launch": xla_fraction,
            "config_snapshot_gap": ("config_initial.json does not carry data_loader_workers or seed, "
                                    "so both are read from the registered container argv; the "
                                    "snapshot writer should record them for later rounds"),
            "fsdp_devices": config.get("fsdp_devices"),
            "config_initial_sha256": common.sha256_file(run / "config_initial.json"),
            "train_view_meta_sha256": config["train_view_meta_sha256"],
            "norm_stats_file_sha256": config["norm_stats_sha256"],
            "norm_stats_canonical_sha256": meta.get("norm_stats_canonical_sha256"),
            "record": {
                "exit_code": run_record.get("exit_code"),
                "wall_time_s": run_record.get("wall_time_s"),
                "steps": run_record.get("steps"),
                "started_at": None,
                "finished_at": run_record.get("finished_at"),
                "timestamp_note": ("run_initial.json writes the same timestamp into both fields; "
                                   "wall_time_s is the measured duration and is authoritative"),
                "seconds_per_update_including_init_and_saves": (
                    round(float(run_record["wall_time_s"]) / float(run_record["steps"]), 4)
                    if run_record.get("wall_time_s") and run_record.get("steps") else None),
                "gpu": public_gpu_metrics(run_record.get("gpu")),
                "source": "run_initial.json written by the registered launcher on exit",
            },
            "norm_stats_binding": ("the run config names the train-only statistics by file hash and "
                                   "the frozen view binds them canonically; both are recorded so the "
                                   "same document is identified either way"),
            "view_sha256": meta["view_sha256"],
            "data_admission": config.get("data_admission"), "sampling_plan": config.get("sampling_plan"),
            "sampling_summary": config.get("sampling_summary"),
            "protocol_sha256_frozen_in_view": meta["protocol_sha256"],
            "protocol_template_sha256": common.sha256_file(args.protocol),
            "training_template_sha256": common.sha256_file(args.training_config),
            "validation_gate_sha256": common.sha256_json(gates),
        },
        "resource_ownership": {
            "registry": str((run / "resource_registry.json").relative_to(common.REPO_ROOT)),
            "entry_count": len(registry),
            "roles": dict(Counter(resource_role(row.get("command") or []) for row in registry)),
            "status_counts": dict(Counter(row.get("status", "unknown") for row in registry)),
            "owner_verified_at_registration_count": sum(
                bool(row.get("matched_all_registration_fields")) for row in registry),
            "identity_note": "exact container id, name, owner and GPU UUID remain only in the ignored private registry",
            "cleanup_policy": "exact registered entry plus confirm token; no broad filter, no purge, "
                              "no daemon restart, no container that was not registered here",
        },
        "containers": [public_container_record(row) for row in containers],
        "evaluations_by_report": reports,
        "initialisation_reference": {
            "label": "init_step1", "train_state_step": 1,
            "why_not_the_published_weights": (narrative.get("initialisation_reference") or {}).get(
                "why_not_the_published_weights", "not carried over"),
            "direction_of_the_error": (narrative.get("initialisation_reference") or {}).get(
                "direction_of_the_error"),
            "parameters_sha256": (narrative.get("initialisation_reference") or {}).get(
                "parameters_sha256"),
            "source_run": "batch-8 memory-probe checkpoint, Orbax directory 0",
            "reports": {key: value for key, value in reports.items() if key.startswith("init_")},
            "superseded_first_scoring": ("640 windows under the pre-fix evaluator, kept as "
                                         "evaluation/init_step1_superseded_old_eval and ignored by "
                                         "the summariser"),
        },
        "checkpoints": [{"orbax_directory": index, "authoritative_step": driver.expected_step(index),
                         "path": str(path.relative_to(common.REPO_ROOT))} for index, path in checkpoints],
        "checkpoint_step_readback": [
            {"artifact": str(path.relative_to(common.REPO_ROOT)),
             "orbax_step_directory": row["orbax_step_directory"],
             "train_state_step": row["train_state_step"],
             "directory_to_train_state_offset": row["directory_to_train_state_offset"],
             "param_tensors": row["param_tensors"],
             "trainable_parameter_tensors": row.get("trainable_parameter_tensors"),
             "trainable_parameter_count": row.get("trainable_parameter_count"),
             "trainable_parameter_names": row.get("trainable_parameter_names"),
             "parameters_sha256": row["parameters_sha256"]}
            for path in sorted(set(run.glob("train_state_dir*.json"))
                               | set(run.glob("train_state_final.json")))
            for row in [common.load_json(path)]],
        "step_mapping_note": ("the Orbax directory index is a zero-based loop index and TrainState.step "
                              "is the applied-update counter, so directory d holds step d+1; this run "
                              "therefore saves its intermediate checkpoints one update after each "
                              "multiple of 5,000 while the final save still carries the targeted "
                              "update count. Measured on this store, not assumed."),
        "steps_logged": len(steps), "highest_step_logged": max(steps) if steps else None,
        "evaluations": evaluations,
        "evaluations_by_split": driver_manifests,
        "learning_rate_trace_artifact": next((str(cand.relative_to(common.REPO_ROOT)) for cand in
                                              (run / "lr_trace_authoritative.json", run / "lr_trace.json")
                                              if cand.is_file()), "missing"),
        "openloop_reports": sorted(str(path.relative_to(common.REPO_ROOT)) for path in
                                   (run / "evaluation").glob("*/report.json")),
    }
    train_curve = split_curve(reports, "train")
    train_noise_hashes = {point.get("noise_set_sha256") for point in train_curve
                          if point.get("noise_set_sha256")}
    document["training_split_openloop_curve"] = {
        "split": "train", "window_set": "frozen_plan", "points": train_curve,
        "noise_set_identical_across_points": bool(train_noise_hashes)
        and len(train_noise_hashes) == 1,
        "used_for_model_selection": False,
        "claim": "train-split open-loop action prediction on recorded expert observations; "
                 "not a closed-loop result",
    }
    train_curve_artifacts = {}
    for name in ("train_curve_summary.json", "openloop_figures_train/figure_manifest.json"):
        candidate = run / name
        if candidate.is_file():
            train_curve_artifacts[name] = {
                "path": str(candidate.relative_to(common.REPO_ROOT)),
                "sha256": common.sha256_file(candidate),
            }
    if train_curve_artifacts:
        document["training_split_curve_artifacts"] = train_curve_artifacts
    # the trace is taken from the run's own artifact, never carried over: the checkpoint steps are
    # authoritative (`lr_trace_authoritative.json`) and a carried-over trace would name the
    # directory multiples instead
    for candidate in (run / "lr_trace_authoritative.json", run / "lr_trace.json"):
        if candidate.is_file():
            document["learning_rate_trace"] = common.load_json(candidate)
            document["learning_rate_trace"]["artifact"] = str(candidate.relative_to(common.REPO_ROOT))
            break
    for key in NARRATIVE_KEYS:
        if key in narrative:
            document[key] = narrative[key]
        else:
            document[key] = "not carried over: the previous record had no such field"
    document["narrative_source"] = str(args.narrative) if args.narrative else None
    document["boundaries"] = {"policy_actions_executed_in_simulation": False,
                              "real_robot_touched": False, "public_push": False,
                              "validation_used_for_training": False,
                              "gate_thresholds_changed_after_training": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(args.output, strip_mount(document))
    print(json.dumps({"steps_logged": document["steps_logged"],
                      "highest_step_logged": document["highest_step_logged"],
                      "checkpoints": [row["authoritative_step"] for row in document["checkpoints"]],
                      "containers": len(containers),
                      "reports": len(document["openloop_reports"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
