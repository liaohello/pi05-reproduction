#!/usr/bin/env python3
"""E4-T T2 diagnosis: why the key-event timing gate can fail while the gripper error is tiny.

The gate compares the expert's close/release transition with the predicted one, and the prediction
the gate sees is the mean over the paired noise copies.  A discrete close that the copies place on
different frames averages into a ramp, and a ramp can pass through the open and closed thresholds
without ever producing a frame pair that satisfies them.  This script reads the per-copy predictions
that the evaluator already stores in ``arrays.npz`` and counts, for each event window, how many
copies do produce a crossing and how far that crossing sits from the expert's frame.

It changes no gate: the pre-registered determination stays on the mean prediction over the frozen
window plan.  This is the explanation of that result.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402

JOINT, GRIPPER = slice(0, 10), slice(10, 12)
ARMS = {"left": 5, "right": 11}


def crossings(series, open_percent: float, closed_percent: float) -> list[int]:
    """Frame indices where the series goes from open to closed or back."""
    closing = [i for i in range(1, len(series))
               if series[i - 1] >= open_percent and series[i] <= closed_percent]
    opening = [i for i in range(1, len(series))
               if series[i - 1] <= closed_percent and series[i] >= open_percent]
    return closing + opening


def nearest_offset(frame: int, events) -> int:
    """Signed distance from a predicted crossing to the nearest expert crossing."""
    return min((int(frame) - int(event) for event in events), key=abs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True,
                        help="an evaluation output directory holding report.json, records.jsonl.gz "
                             "and arrays.npz")
    parser.add_argument("--training-config", type=Path,
                        default=common.REPO_ROOT / "configs/so101/e4t_training.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    directory = args.report_dir.expanduser().resolve()
    report = common.load_json(directory / "report.json")
    gates = common.load_json(args.training_config)["validation_gate"]
    open_percent = float(gates["gripper_open_threshold_percent"])
    closed_percent = float(gates["gripper_closed_threshold_percent"])
    with gzip.open(directory / "records.jsonl.gz", "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    arrays = np.load(directory / "arrays.npz", allow_pickle=False)
    truth, predictions = arrays["ground_truth"], arrays["predictions"]
    if len(records) != len(truth):
        raise SystemExit(f"{len(records)} records but {len(truth)} stored windows")

    rows = []
    for index, record in enumerate(records):
        for arm, column in ARMS.items():
            expert = [event["frame"] for event in record["gt_close_events"] + record["gt_release_events"]
                      if event["arm"] == column]
            if not expert:
                continue
            mean = predictions[index].mean(axis=0)[:, column]
            mean_events = crossings(mean, open_percent, closed_percent)
            per_copy = []
            for copy in range(predictions.shape[1]):
                series = predictions[index][copy][:, column]
                for frame in crossings(series, open_percent, closed_percent):
                    per_copy.append({"copy": copy, "frame": frame,
                                     "offset_frames": nearest_offset(frame, expert)})
            rows.append({"episode": record["episode"], "start": record["start"],
                         "window_set": report.get("window_set"), "arm": arm,
                         "expert_frames": expert,
                         "mean_crossings": mean_events,
                         "mean_offset_frames": (min((nearest_offset(frame, expert) for frame in mean_events),
                                                    key=abs) if mean_events else None),
                         "copies_with_a_crossing": len({item["copy"] for item in per_copy}),
                         "noise_copies": int(predictions.shape[1]),
                         "copy_offsets": per_copy,
                         "gripper_mae_pct": record["gripper_mae_pct"]})
    with_crossing = [row for row in rows if row["mean_crossings"]]
    copy_hits = [row["copies_with_a_crossing"] for row in rows]
    offsets = [item["offset_frames"] for row in rows for item in row["copy_offsets"]]
    summary = {
        "schema_version": 1, "branch": "so101", "phase": "E3",
        "substage": "E4-T-T2-event-smearing",
        "report_dir": str(directory.name), "checkpoint_train_state_step": report.get("checkpoint_train_state_step"),
        "window_set": report.get("window_set"), "windows": len(records),
        "thresholds": {"open_percent": open_percent, "closed_percent": closed_percent},
        "expert_event_windows": len(rows),
        "windows_whose_mean_prediction_crosses": len(with_crossing),
        "mean_crossing_share": (len(with_crossing) / len(rows) if rows else None),
        "mean_offsets_frames": sorted(row["mean_offset_frames"] for row in with_crossing),
        "copies_with_a_crossing_histogram": {str(count): copy_hits.count(count)
                                             for count in sorted(set(copy_hits))},
        "per_copy_offsets_frames": {"n": len(offsets),
                                    "min": min(offsets) if offsets else None,
                                    "median": float(np.median(np.abs(offsets))) if offsets else None,
                                    "max": max(offsets) if offsets else None,
                                    "within_3": sum(1 for value in offsets if abs(value) <= 3)},
        "reading": ("the gate is evaluated on the mean over noise copies; if the copies place a "
                    "discrete close on different frames the mean becomes a ramp that need not cross "
                    "both thresholds, so the gate can report no paired event while the per-copy "
                    "trajectories do close, and within a few frames of the expert"),
        "gate_is_not_changed_by_this_analysis": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(args.output, {**summary, "per_window": rows})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
