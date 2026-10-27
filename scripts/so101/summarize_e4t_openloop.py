#!/usr/bin/env python3
"""E4-T T3: fold per-checkpoint open-loop reports into curves and the gate verdict.

Reads the report.json produced by evaluate_e4t_openloop.py for every saved
checkpoint, plots the validation error against TrainState.step and applies the
gates that configs/so101/e4t_training.json pre-registers before training.  It
never reruns a threshold: a missing group is reported as unmeasured, which is a
gate failure, not a pass.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402


def _group(report: dict, name: str) -> dict:
    """Prefer the per-sampled-copy, episode-equal aggregation when the re-derivation exists."""
    per_copy = (report.get("per_copy") or {}).get("aggregates") or {}
    if name in per_copy:
        return per_copy[name]
    return (report.get("aggregates") or {}).get(name) or {}


def _value(report: dict, group: str, key: str):
    return _group(report, group).get(key)


def _legacy(report: dict, group: str, key: str = "joint_macro_mae_deg"):
    """The first-pass mean-of-copies number, kept beside the corrected metric for audit."""
    return ((report.get("aggregates") or {}).get(group) or {}).get(key)


def _step(report: dict) -> int:
    return int(report.get("checkpoint_train_state_step", -1))


def _check(value, limit: float) -> str:
    if value is None:
        return "unmeasured"
    return "pass" if float(value) <= float(limit) else "fail"


def _beats(candidate, reference, key: str) -> str:
    if candidate is None or reference is None:
        return "unmeasured"
    return "pass" if float(candidate) < float(reference) else "fail"


def event_timing_check(timing: dict, limit: float) -> tuple[str, int | None, dict]:
    """Compare every paired gripper event against the pre-registered frame tolerance.

    The evaluator pairs each ground-truth event with the nearest predicted event of the same arm
    inside a fixed window.  A ground-truth event that is unpaired, or an event kind with ground
    truth but no accepted pairing, is a timing failure: it is never dropped from the verdict,
    because silently ignoring the missed events would report the timing of the events that happen
    to match as if it described all of them.
    """
    per_kind, worst = {}, None
    for kind in ("close", "release"):
        block = timing.get(kind) or {}
        ground = int(block.get("ground_truth_events") or 0)
        unpaired = int(block.get("unpaired") or 0)
        value = block.get("max_abs_frames")
        if ground == 0:
            per_kind[kind] = "unmeasured"
            continue
        if value is not None:
            # report the measured worst offset even when missing events already fail the check
            worst = int(value) if worst is None else max(worst, int(value))
        if unpaired > 0 or not block.get("measurable") or value is None:
            per_kind[kind] = "fail"
            continue
        per_kind[kind] = "pass" if int(value) <= float(limit) else "fail"
    if any(value == "fail" for value in per_kind.values()):
        return "fail", worst, per_kind
    if any(value == "unmeasured" for value in per_kind.values()):
        return "unmeasured", worst, per_kind
    return "pass", worst, per_kind


def diagnostic_checkpoint(rows: list[dict]) -> dict | None:
    """Choose the lowest-error trained checkpoint when no candidate clears first filters.

    This is a validation-set diagnostic selection only. It must not turn a failed timing or
    gripper gate into a closed-loop-eligible checkpoint.
    """
    candidates = [row for row in rows
                  if int(row.get("step", -1)) > 0 and row.get("primary_objective_deg") is not None]
    return min(candidates, key=lambda row: (float(row["primary_objective_deg"]),
                                            int(row["step"]))) if candidates else None


def load_noise_map(report_path: Path) -> dict:
    """Map every evaluated window to the noise draws used for it, so the curve is one noise set."""
    archive = report_path.parent / "records.jsonl.gz"
    if not archive.exists():
        return {}
    window_noise = {}
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            window_noise[f"{row['episode']}:{row['start']}"] = row["noise_hashes"]
    return window_noise


def _noise_digest(window_noise: dict) -> str | None:
    """Hash the canonical record map with the same serializer used by the evaluator."""
    if not window_noise:
        return None
    canonical = {key: sorted(value) for key, value in sorted(window_noise.items())}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def noise_set_digest(report: dict) -> tuple[str | None, str]:
    """The noise set this checkpoint was scored on.

    The per-window records are authoritative because they are written independently of the summary
    claim; a report without them falls back to the digest the evaluator hashed itself.
    """
    archive = load_noise_map(Path(report["_path"])) if report.get("_path") else {}
    if archive:
        return _noise_digest(archive), "records.jsonl.gz"
    return report.get("noise_set_sha256"), "report-field"


def evaluate(report: dict, gates: dict, initial: dict | None,
             reference_noise_digest: str | None = None) -> dict:
    noise_digest, noise_source = noise_set_digest(report)
    clean_mae = _value(report, "clean_active", "joint_macro_mae_deg")
    rec_mae = _value(report, "recovery_post_active", "joint_macro_mae_deg")
    clean_p95 = _value(report, "clean_active", "joint_p95_deg")
    rec_p95 = _value(report, "recovery_post_active", "joint_p95_deg")
    clean_grip = _value(report, "clean_active", "gripper_macro_mae_pct")
    rec_grip = _value(report, "recovery_post_active", "gripper_macro_mae_pct")
    timing = ((report.get("per_copy") or {}).get("event_timing")
              or report.get("event_timing") or {})
    timing_instrument = "frozen_plan_mean_of_copies"
    anchored = report.get("anchored") or {}
    if anchored.get("per_copy"):
        # the frozen plan contains only a fraction of the recorded transitions, and an event that
        # falls outside a 10-step window cannot be predicted inside it.  The pre-registered
        # event-anchored pass places one window on every transition; timing is read from it.
        timing = anchored["per_copy"].get("event_timing") or {}
        timing_instrument = "event_anchored_per_copy"
    elif anchored.get("report"):
        timing = anchored["report"].get("event_timing") or timing
        timing_instrument = "event_anchored_mean_of_copies"
    timing_verdict, worst_timing, timing_per_kind = event_timing_check(
        timing, gates["key_event_timing_tolerance_control_frames"])
    expected_windows = report.get("windows_expected")
    if expected_windows is None:
        completeness = "unmeasured"
    elif report.get("windows") == expected_windows and report.get("success") is True:
        completeness = "pass"
    else:
        completeness = "fail"
    checks = {
        "evaluation_uses_frozen_window_plan": completeness,
        # The registered rule is an episode-equal macro inside each group. A report that only has
        # the legacy window-macro mean-of-copies numbers cannot satisfy it.
        "episode_equal_aggregation": "pass" if report.get("per_copy") else "fail",
        "clean_active_joint_macro_mae": _check(clean_mae, gates["clean_active_joint_macro_mae_deg"]),
        "recovery_post_active_joint_macro_mae": _check(rec_mae,
                                                       gates["recovery_post_event_active_joint_macro_mae_deg"]),
        "clean_active_joint_p95": _check(clean_p95, gates["clean_active_joint_p95_deg"]),
        "recovery_post_active_joint_p95": _check(rec_p95,
                                                 gates["recovery_post_event_active_joint_p95_deg"]),
        "gripper_macro_mae": ("pass" if clean_grip is not None and rec_grip is not None
                              and max(float(clean_grip), float(rec_grip)) <= gates["gripper_mae_pct"]
                              else "fail" if None not in (clean_grip, rec_grip) else "unmeasured"),
        "key_event_timing_frames": timing_verdict,
        "timing_on_event_anchored_pass": ("pass" if timing_instrument == "event_anchored_per_copy"
                                          else "fail"),
        # Every checkpoint must see the same noise draws, or a curve movement could be the draws.
        "shared_noise_set": ("unmeasured" if not noise_digest or not reference_noise_digest
                             else "pass" if noise_digest == reference_noise_digest else "fail"),
    }
    if gates.get("must_beat_hold_baseline"):
        for group in ("clean_active", "recovery_post_active"):
            checks[f"beat_hold_{group}"] = _beats(
                _value(report, group, "joint_macro_mae_deg"), _value(report, group, "hold_joint_mae_deg"),
                "joint_macro_mae_deg")
    if gates.get("must_beat_train_mean_baseline"):
        for group in ("clean_active", "recovery_post_active"):
            checks[f"beat_train_mean_{group}"] = _beats(
                _value(report, group, "joint_macro_mae_deg"),
                _value(report, group, "train_mean_joint_mae_deg"), "joint_macro_mae_deg")
    if gates.get("must_beat_initialization"):
        if initial is None:
            # without a reference the whole family would silently disappear from the checks and the
            # gate could pass without ever testing initialisation
            checks["initialization_reference_available"] = "fail"
        else:
            for group in ("clean_active", "recovery_post_active"):
                if _step(report) > _step(initial):
                    checks[f"beat_initialization_{group}"] = _beats(
                        _value(report, group, "joint_macro_mae_deg"),
                        _value(initial, group, "joint_macro_mae_deg"), "joint_macro_mae_deg")
    if gates.get("static_only_success_is_insufficient"):
        # a checkpoint that is accurate only because every scored window is static must not pass
        checks["active_groups_measured"] = (
            "pass" if int(_group(report, "clean_active").get("windows", 0) or 0) > 0
            and int(_group(report, "recovery_post_active").get("windows", 0) or 0) > 0 else "fail")
    if timing_verdict != "unmeasured" and gates.get("key_event_timing_tolerance_ms"):
        # the frame and millisecond tolerances describe the same window at this round's 50 Hz
        # control rate; a disagreement between them is a configuration error, not a measurement
        frames = gates["key_event_timing_tolerance_control_frames"]
        checks["timing_tolerance_units_consistent"] = (
            "pass" if abs(frames * (1000.0 / float(report.get("control_fps", 50)))
                         - float(gates["key_event_timing_tolerance_ms"])) < 1e-6 else "fail")
    results = set(checks.values())
    primary = None if clean_mae is None or rec_mae is None else 0.5 * (float(clean_mae) + float(rec_mae))
    return {
        "step": _step(report), "label": report.get("label"), "split": report.get("split"),
        "windows": report.get("windows"), "windows_expected": expected_windows,
        "primary_objective_deg": primary,
        "clean_active_joint_macro_mae_deg": clean_mae,
        "recovery_post_active_joint_macro_mae_deg": rec_mae,
        "clean_active_joint_p95_deg": clean_p95, "recovery_post_active_joint_p95_deg": rec_p95,
        "gripper_macro_mae_pct_clean": clean_grip, "gripper_macro_mae_pct_recovery": rec_grip,
        "metric_source": ("per_noise_copy_episode_equal" if report.get("per_copy")
                          else "mean_of_copies_legacy"),
        "legacy_mean_of_copies": {
            "clean_active_joint_macro_mae_deg": _legacy(report, "clean_active"),
            "recovery_post_active_joint_macro_mae_deg": _legacy(report, "recovery_post_active"),
            "clean_active_joint_p95_deg": _legacy(report, "clean_active", "joint_p95_deg"),
            "recovery_post_active_joint_p95_deg": _legacy(report, "recovery_post_active",
                                                          "joint_p95_deg"),
            "clean_active_gripper_macro_mae_pct": _legacy(report, "clean_active",
                                                          "gripper_macro_mae_pct"),
            "recovery_post_active_gripper_macro_mae_pct": _legacy(
                report, "recovery_post_active", "gripper_macro_mae_pct"),
        },
        "worst_event_timing_frames": worst_timing,
        "horizon_mae_joint_deg": ((report.get("per_copy") or {}).get("horizon_mae_joint_deg")
                                  or report.get("horizon_mae_joint_deg")),
        "velocity_shape": {
            key: {"predicted": _value(report, group, "pred_delta_rms_deg_per_frame"),
                  "ground_truth": _value(report, group, "gt_delta_rms_deg_per_frame"),
                  "predicted_reversals_per_window": _value(report, group, "pred_reversals_per_window")}
            for group, key in (("clean_active", "clean_active"),
                               ("recovery_post_active", "recovery_post_active"))},
        "event_timing": {key: value for key, value in timing.items() if key != "unpaired_examples"},
        "event_timing_verdict": timing_per_kind,
        "event_timing_instrument": timing_instrument,
        "noise_set_sha256": noise_digest,
        "noise_set_source": noise_source,
        "per_category_post_active": {
            f"recovery_{key}_post_active": _value(report, f"recovery_{key}_post_active",
                                                  "joint_macro_mae_deg") for key in ("A", "B", "C")},
        # A category with zero windows is not a measured failure; it means this round collected none.
        "categories_without_windows": sorted(
            f"recovery_{key}" for key in ("A", "B", "C")
            if int(_group(report, f"recovery_{key}").get("windows", 0) or 0) == 0),
        "checks": checks,
        "first_filter_passed": all(checks[key] == "pass" for key in
                                   ("gripper_macro_mae", "key_event_timing_frames")),
        "gate_passed": results == {"pass"},
        "gate_failed_checks": sorted(key for key, value in checks.items() if value != "pass"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, required=True,
                        help="directory containing one subdirectory per evaluated checkpoint")
    parser.add_argument("--gates", type=Path,
                        default=common.REPO_ROOT / "configs/so101/e4t_training.json")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--window-set", default="frozen_plan",
                        choices=("frozen_plan", "event_anchored_supplementary",
                                 "frozen_plan_plus_dense_sweep"),
                        help="a curve may only join rows scored on the same window set")
    parser.add_argument("--expected-config-sha256", default=None,
                        help="run config the checkpoints must have been trained with")
    parser.add_argument("--run-config", type=Path, default=None,
                        help="the run config snapshot the training reported, so the statistics "
                             "binding can be checked against the frozen view")
    parser.add_argument("--view", type=Path, default=None,
                        help="the frozen training view; its meta.json binds the canonical statistics")
    parser.add_argument("--initial-label", default=None,
                        help="label of the report that represents the initialisation reference; "
                             "without it the step-0 report is used, and without a step-0 report the "
                             "beat-initialisation checks are omitted rather than assumed")
    parser.add_argument("--per-copy-dir", type=Path, default=None,
                        help="directory holding <label>.json per-copy re-derivations; default "
                             "<reports>/per_copy, with a fallback beside report.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gates = common.load_json(args.gates)["validation_gate"]
    per_copy_dir = (args.per_copy_dir.expanduser().resolve()
                    if args.per_copy_dir else args.reports / "per_copy")
    reports, superseded = [], []
    for path in sorted(args.reports.glob("*/report.json")):
        report = common.load_json(path)
        per_copy_path = per_copy_dir / f"{path.parent.name}.json"
        if not per_copy_path.is_file():
            per_copy_path = path.parent / "per_copy.json"
        if per_copy_path.is_file():
            report["per_copy"] = common.load_json(per_copy_path)
        anchored_dir = path.parent.parent / f"{path.parent.name}_anchored"
        if (anchored_dir / "report.json").is_file():
            anchored = {"report": common.load_json(anchored_dir / "report.json")}
            anchored_pc = per_copy_dir / f"{anchored_dir.name}.json"
            if not anchored_pc.is_file():
                anchored_pc = anchored_dir / "per_copy.json"
            if anchored_pc.is_file():
                anchored["per_copy"] = common.load_json(anchored_pc)
            report["anchored"] = anchored
        if "superseded" in path.parent.name:
            # kept beside the live tree for provenance; scored under an older methodology
            superseded.append(path.parent.name)
            continue
        if (report.get("window_set") or "frozen_plan") != args.window_set:
            continue
        if report.get("split") == args.split:
            report["_path"] = str(path)
            reports.append(report)
    if not reports:
        raise SystemExit(f"no {args.split} reports found under {args.reports}")
    labels = [report.get("label") for report in reports]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise SystemExit(f"two reports share the label {duplicates}; a curve row must be unique")
    if args.initial_label:
        initial = next((row for row in reports if row.get("label") == args.initial_label), None)
        if initial is None:
            raise SystemExit(f"no report labelled {args.initial_label}")
    else:
        initial = min((row for row in reports if _step(row) == 0), key=_step, default=None)
    # The initialisation reference is not a checkpoint of this run: it is a state saved by the
    # batch-8 probe under that run's own config, so the one-config rule is applied to the
    # checkpoints and the reference's config is published separately instead of being forced to fit.
    checkpoints = [report for report in reports if report is not initial]
    config_shas = sorted({report.get("run_config_sha256") for report in checkpoints})
    if len(config_shas) != 1 or not config_shas[0]:
        raise SystemExit(f"the evaluated checkpoints do not share one run config: {config_shas}")
    if args.expected_config_sha256 and args.expected_config_sha256 != config_shas[0]:
        raise SystemExit(f"run config mismatch: reports say {config_shas[0]}, "
                         f"expected {args.expected_config_sha256}")
    initial_config_sha = None if initial is None else initial.get("run_config_sha256")
    # The reference is the initialisation report when present, else the lowest step; every other
    # checkpoint's noise set is compared against it.
    reference_report = min(reports, key=lambda row: (_step(row) != 0, _step(row)))
    reference_noise = noise_set_digest(reference_report)[0]
    bindings = {}
    if args.view or args.run_config:
        if not (args.view and args.run_config):
            raise SystemExit("--view and --run-config must be given together")
        meta = common.load_json(args.view / "meta.json")
        run_config = common.load_json(args.run_config)
        view_shas = sorted({report.get("view_sha256") for report in reports})
        if view_shas != [meta.get("view_sha256")]:
            raise SystemExit(f"reports were not all scored on the frozen view: {view_shas} "
                             f"vs {meta.get('view_sha256')}")
        if common.sha256_file(args.run_config) != config_shas[0]:
            raise SystemExit(f"--run-config {common.sha256_file(args.run_config)} is not the config "
                             f"the reports carry: {config_shas[0]}")
        # The run config names the statistics by file hash while the view binds them canonically;
        # both must point at the same train-only statistics document.
        if not run_config.get("norm_stats_sha256") or (
                run_config.get("norm_stats_sha256") != meta.get("norm_stats_sha256")):
            raise SystemExit(f"the run config statistics {run_config.get('norm_stats_sha256')} do not "
                             f"agree with the frozen view {meta.get('norm_stats_sha256')}")
        if run_config.get("train_view_meta_sha256") != common.sha256_file(args.view / "meta.json"):
            raise SystemExit("the run config does not name this view's meta.json")
        bindings = {
            "view_sha256": meta["view_sha256"],
            "view_meta_sha256": run_config["train_view_meta_sha256"],
            "norm_stats_file_sha256": run_config["norm_stats_sha256"],
            "norm_stats_canonical_sha256": meta.get("norm_stats_canonical_sha256"),
            "norm_stats_note": ("the reports run config binds the train-only statistics by file hash "
                               "and the frozen view binds them canonically; both were checked to "
                               "agree, and the per-checkpoint reports inherit that binding because "
                               "each one names this run config and this view"),
        }
    rows = sorted((evaluate(report, gates, initial, reference_noise) for report in reports),
                  key=lambda row: row["step"])
    candidates = [row for row in rows if row["first_filter_passed"]
                  and row["primary_objective_deg"] is not None and row["step"] > 0]
    best = None
    if candidates:
        best = min(candidates, key=lambda row: (row["primary_objective_deg"], row["step"]))
    diagnostic = diagnostic_checkpoint(rows)
    passing = [row for row in rows if row["gate_passed"]]
    summary = {
        "schema_version": 1, "branch": "so101", "phase": "E3", "substage": "E4-T-T3-open-loop-gate",
        "split": args.split, "gates_sha256": common.sha256_json(gates),
        "window_set": args.window_set,
        "pre_registered_before_training": bool(gates.get("pre_registered_before_training")),
        "run_config_sha256": config_shas[0],
        "checkpoint_config_binding": "every evaluated report carries this run config hash and only this one",
        "initial_reference_run_config_sha256": initial_config_sha,
        "initial_reference_config_note": ("the initialisation reference is a state saved by the batch-8 "
                                          "probe under that run's own config; it shares the model, LoRA "
                                          "structure, frozen view and train-only statistics with this "
                                          "run, and its own config hash is published separately rather "
                                          "than forced to equal the checkpoints'"),
        "statistics_and_view_bindings": bindings or "not checked: --view/--run-config not given",
        "noise_set_sha256": reference_noise,
        "noise_set_reference_label": reference_report.get("label"),
        "noise_set_sources": sorted({row["noise_set_source"] for row in rows}),
        "noise_set_identical_across_checkpoints": bool(reference_noise) and all(
            row["noise_set_sha256"] == reference_noise for row in rows),
        "noise_set_note": ("every checkpoint was scored on identical per-window noise draws, so a curve "
                           "movement comes from the weights and not from the sampling"
                           if reference_noise else "no per-window noise records were found"),
        "initial_reference_step": None if initial is None else _step(initial),
        "initial_reference_label": None if initial is None else initial.get("label"),
        "initial_reference_is_zero_updates": bool(initial is not None and _step(initial) == 0),
        "curve": rows,
        "superseded_report_directories_ignored": sorted(superseded),
        "gate_passed_checkpoints": [row["step"] for row in passing],
        "best_checkpoint": best,
        "diagnostic_checkpoint": diagnostic,
        "best_is_gate_passing": bool(best is not None and best["gate_passed"]),
        "decision": ("eligible_for_closed_loop" if best is not None and best["gate_passed"]
                     else "not_eligible_for_closed_loop"),
        "best_checkpoint_claim": ("diagnostic validation selection; the validation set informed the "
                                  "protocol and is not an independent final test"),
        "closed_loop_started": False,
        "action_executed_in_simulation": False,
        "claim": "open-loop prediction error on recorded expert observations; not a success rate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(args.output, summary)
    print(json.dumps({"checkpoints": [row["step"] for row in rows],
                      "gate_passed": summary["gate_passed_checkpoints"],
                      "best_step": None if best is None else best["step"],
                      "decision": summary["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
