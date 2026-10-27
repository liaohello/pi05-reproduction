#!/usr/bin/env python3
"""Write the E4-T stage budget as measured from this round's own records.

The round reserves 8 h for collection/adaptation, 8 h for training and 8 h for evaluation and
plotting inside a 24 h total.  Every number here is read back from artifacts (ledger timestamps,
the training run report, the evaluation driver manifest); nothing is estimated by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402

REPO = common.REPO_ROOT
RESERVATION_S = {"collection_adaptation": 8 * 3600, "training": 8 * 3600,
                 "evaluation_plotting": 8 * 3600, "round_total": 24 * 3600}
FAMILIES = {
    "collection_adaptation": ("E4-T-T-DATA-pilot", "E4-T-T-DATA-collection", "E4-T-T-DATA-gate",
                             "E4-T-T1-batch-probe", "E4-T-T0-selfcheck"),
    "training": ("E4-T-T1-training",),
    "evaluation_plotting": ("E4-T-T2-openloop", "E4-T-T3-openloop-gate"),
}


def parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


def evaluation_evidence(run_dir: Path) -> dict:
    """Count both split drivers and supplementary reports without counting retries twice."""
    driver_attempts, completed_by_split, starts = {}, {}, []
    for split in ("train", "validation"):
        path = run_dir / f"evaluation_{split}_driver.json"
        if not path.is_file():
            continue
        rows = common.load_json(path).get("evaluations", [])
        driver_attempts[split] = len(rows)
        completed_by_split[split] = len({row.get("label") for row in rows
                                         if row.get("success") and row.get("label")})
        for row in rows:
            if row.get("started"):
                try:
                    starts.append(parse(row["started"]))
                except (TypeError, ValueError):
                    pass

    report_rows = []
    for path in (run_dir / "evaluation").glob("*/report.json"):
        try:
            row = common.load_json(path)
            if row.get("split") not in {"train", "validation"}:
                continue
            report_rows.append(row)
        except (OSError, ValueError):
            continue
    frozen_by_split = {}
    supplementary_by_window_set = {}
    report_times = []
    for row in report_rows:
        window_set = row.get("window_set") or "frozen_plan"
        if window_set == "frozen_plan":
            frozen_by_split.setdefault(row["split"], set()).add(row.get("label"))
        else:
            supplementary_by_window_set[window_set] = supplementary_by_window_set.get(window_set, 0) + 1
        if row.get("created"):
            try:
                report_times.append(parse(row["created"]))
            except (TypeError, ValueError):
                pass
    report_mtime_times = []
    for path in (run_dir / "evaluation").glob("*/report.json"):
        try:
            row = common.load_json(path)
            if row.get("created"):
                parse(row["created"])
                continue
            report_mtime_times.append(datetime.fromtimestamp(path.stat().st_mtime, timezone.utc))
        except (OSError, ValueError, TypeError):
            continue

    # Supplementary evaluations do not use the train/validation driver manifests. Their
    # exact-owner resource registrations provide launch timestamps; report file mtimes
    # provide completion evidence when the report itself has no `created` field.
    resource_launches, resource_finishes = [], []
    registry_path = run_dir / "resources.json"
    if not registry_path.is_file():
        registry_path = run_dir / "resource_registry.json"
    try:
        registry = common.load_json(registry_path) if registry_path.is_file() else {}
        for entry in registry.get("entries", []):
            exact_name = str(entry.get("exact_name", ""))
            if not exact_name.startswith("so101-e4t-eval-") or not entry.get("created"):
                continue
            try:
                resource_launches.append(parse(entry["created"]))
            except (TypeError, ValueError):
                continue
            if entry.get("finished_at"):
                try:
                    resource_finishes.append(parse(entry["finished_at"]))
                except (TypeError, ValueError):
                    pass
    except (OSError, ValueError):
        pass

    all_launches = starts + resource_launches
    all_end_times = report_times + report_mtime_times + resource_finishes
    wall_span = (int((max(all_end_times) - min(all_launches)).total_seconds())
                 if all_launches and all_end_times else None)
    summary = {
        "driver_attempts_by_split": driver_attempts,
        "successful_driver_checkpoints_by_split": completed_by_split,
        "frozen_plan_reports_by_split": {
            split: len(labels - {None}) for split, labels in frozen_by_split.items()},
        "supplementary_reports_by_window_set": supplementary_by_window_set,
        "first_launch": min(starts).isoformat() if starts else None,
        "last_launch": max(starts).isoformat() if starts else None,
        "report_creation_span_s": (int((max(report_times) - min(report_times)).total_seconds())
                                   if len(report_times) > 1 else None),
        "launch_span_s": (int((max(starts) - min(starts)).total_seconds())
                          if len(starts) > 1 else None),
        "registered_evaluation_launches": len(resource_launches),
        "first_registered_evaluation_launch": (min(resource_launches).isoformat()
                                                if resource_launches else None),
        "last_registered_evaluation_finish": (max(resource_finishes).isoformat()
                                              if resource_finishes else None),
        "last_report_mtime_utc": (max(report_mtime_times).isoformat()
                                  if report_mtime_times else None),
        "evaluation_wall_upper_bound_s": wall_span,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=REPO / "experiments/ledger.jsonl")
    parser.add_argument("--run-dir", type=Path, default=REPO / "runs/so101/e4t")
    parser.add_argument("--training-config", type=Path,
                        default=REPO / "configs/so101/e4t_training.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in
               args.ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    e4t = [row for row in records if str(row.get("substage", "")).startswith("E4-T")]
    if not e4t:
        raise SystemExit("no E4-T records in the ledger")
    stamps = sorted(parse(row["ts"]) for row in e4t)
    round_start, last_record = stamps[0], stamps[-1]
    optimisation = common.load_json(args.training_config)["optimisation"]
    RESERVATION_S["training"] = int(optimisation["training_wall_limit_s"])

    stages = {}
    for stage, prefixes in FAMILIES.items():
        rows = [row for row in e4t if any(str(row["substage"]).startswith(p) for p in prefixes)]
        times = sorted(parse(row["ts"]) for row in rows)
        stages[stage] = {
            "reserved_s": RESERVATION_S[stage],
            "ledger_records": [row["substage"] for row in rows],
            "first_record": None if not times else times[0].isoformat(),
            "last_record": None if not times else times[-1].isoformat(),
            "ledger_span_s": None if len(times) < 2 else int((times[-1] - times[0]).total_seconds()),
            "status": "no record yet" if not rows else "recorded",
        }

    run_report = args.run_dir / "run_initial.json"
    if run_report.is_file():
        report = common.load_json(run_report)
        stages["training"]["measured_wall_s"] = round(float(report["wall_time_s"]), 1)
        stages["training"]["exit_code"] = report["exit_code"]
        # the public budget keeps the memory/utilisation observations but not the machine's GPU UUID
        stages["training"]["gpu_admission"] = {
            key: value for key, value in report["gpu"].items()
            if key not in ("selected_uuid", "requested_index")}
        stages["training"]["status"] = "measured from run_initial.json"

    evaluation = evaluation_evidence(args.run_dir)
    if (evaluation["driver_attempts_by_split"] or evaluation["frozen_plan_reports_by_split"]
            or evaluation["supplementary_reports_by_window_set"]):
        stages["evaluation_plotting"].update(evaluation)
        stages["evaluation_plotting"]["checkpoints_evaluated"] = sum(
            evaluation["frozen_plan_reports_by_split"].values())
        stages["evaluation_plotting"]["status"] = (
            "measured from train/validation driver manifests and per-report split/window-set metadata")
        if evaluation.get("evaluation_wall_upper_bound_s") is not None:
            # Span from the first registered/driver launch to the latest report write. It
            # includes gaps and overlaps training, so it is an upper bound, not GPU time.
            stages["evaluation_plotting"]["measured_wall_s"] = evaluation[
                "evaluation_wall_upper_bound_s"]
            stages["evaluation_plotting"]["measured_wall_note"] = (
                "span from the first driver/resource launch to the latest report timestamp or "
                "registered job finish; an upper bound because jobs overlap training and one another")

    optimisation = common.load_json(args.training_config)["optimisation"]
    RESERVATION_S["training"] = int(optimisation["training_wall_limit_s"])

    for stage in stages:
        measured = stages[stage].get("measured_wall_s")
        reference = measured if measured is not None else stages[stage]["ledger_span_s"]
        stages[stage]["used_s"] = reference
        stages[stage]["verdict"] = (
            "no measurement yet" if reference is None else
            "within reservation" if reference <= RESERVATION_S[stage] else
            f"over reservation by {int(reference - RESERVATION_S[stage])} s")
    elapsed = int((datetime.now(timezone.utc) - round_start.astimezone(timezone.utc)).total_seconds())
    report = {
        "schema_version": 1, "branch": "so101", "phase": "E3", "substage": "E4-T-budget",
        "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "round_start": round_start.isoformat(), "elapsed_s": elapsed,
        "round_total_reserved_s": RESERVATION_S["round_total"],
        "round_total_remaining_s": RESERVATION_S["round_total"] - elapsed,
        "overdue": elapsed > RESERVATION_S["round_total"],
        "stages": stages,
        "training_budget_from_config": {
            "planned_updates": optimisation["planned_updates"],
            "training_wall_limit_s": optimisation["training_wall_limit_s"],
            "primary_batch_size": optimisation["primary_batch_size"]},
        "attempt_budgets": {
            "pilot_maximum_attempts": common.load_json(
                REPO / "configs/so101/e4t_protocol.json")["collection"]["pilot_maximum_attempts"],
            "main_collection_maximum_attempts": common.load_json(
                REPO / "configs/so101/e4t_protocol.json")["collection"]["main_maximum_attempts"]},
        "note": "stage spans are taken from ledger timestamps; the training and evaluation entries "
                "add the measured wall clock once those artifacts exist",
        "claim": "budget accounting only; not a performance result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({stage: {key: value for key, value in row.items()
                              if key in ("status", "used_s", "measured_wall_s", "reserved_s",
                                         "verdict", "checkpoints_evaluated")}
                      for stage, row in report["stages"].items()}, indent=2))
    print(json.dumps({"elapsed_s": report["elapsed_s"],
                      "round_total_remaining_s": report["round_total_remaining_s"],
                      "overdue": report["overdue"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
