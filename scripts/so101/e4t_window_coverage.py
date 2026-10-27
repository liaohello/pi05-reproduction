#!/usr/bin/env python3
"""E4-T T2 audit: how much of the frozen window plan can see a gripper event at all.

The key-event-timing gate compares a predicted close/release with the expert's own close/release
inside the same window, so it can only be measured on windows that contain such a transition.  This
script counts, per split, how many physical transitions exist in the recorded expert actions and how
many of them fall inside at least one planned 10-step window.  It reads only the frozen view and the
frozen window plan; it never changes them, because the view hash covers both.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402

GRIPPER = (5, 11)


def transitions(actions, open_percent: float, closed_percent: float) -> list[dict]:
    found = []
    for arm in GRIPPER:
        states = np.asarray(actions, dtype=float)[:, arm]
        for frame in range(1, len(states)):
            if states[frame - 1] >= open_percent and states[frame] <= closed_percent:
                found.append({"arm": arm, "kind": "close", "frame": frame})
            if states[frame - 1] <= closed_percent and states[frame] >= open_percent:
                found.append({"arm": arm, "kind": "release", "frame": frame})
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--training-config", type=Path,
                        default=common.REPO_ROOT / "configs/so101/e4t_training.json")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "both"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    view = args.view.expanduser().resolve()
    gates = common.load_json(args.training_config)["validation_gate"]
    plan = common.load_json(view / "e4t_windows.json")
    meta = common.load_json(view / "meta.json")
    rows = []
    for episode in meta["episodes"]:
        index = int(episode["episode_index"])
        if args.split != "both" and episode["split"] != args.split:
            continue
        with np.load(view / "episodes" / f"episode_{index:03d}" / "episode.npz",
                     allow_pickle=False) as data:
            actions = np.asarray(data["action"], dtype=float)
        starts = [int(value) for value in plan.get(str(index), [])]
        events = transitions(actions, gates["gripper_open_threshold_percent"],
                             gates["gripper_closed_threshold_percent"])
        for event in events:
            # a transition at frame f is visible from a window that starts at f-9 .. f-1, because
            # detection needs both the frame before and the frame after the change
            window = next((start for start in starts if start <= event["frame"] < start + 10), None)
            event["window_start"] = window
            event["planned_windows"] = len(starts)
        rows.append({"episode": index, "split": episode["split"],
                     "category": episode.get("category"), "mechanism": episode.get("mechanism"),
                     "planned_windows": len(starts), "frames": len(actions),
                     "events": len(events),
                     "covered": sum(1 for event in events if event["window_start"] is not None),
                     "by_kind": {kind: {"events": sum(1 for e in events if e["kind"] == kind),
                                         "covered": sum(1 for e in events
                                                        if e["kind"] == kind and e["window_start"] is not None)}
                                 for kind in ("close", "release")},
                     "event_detail": events})
    events_total = sum(row["events"] for row in rows)
    covered_total = sum(row["covered"] for row in rows)
    summary = {
        "schema_version": 1, "branch": "so101", "phase": "E3",
        "substage": "E4-T-T2-window-coverage", "split": args.split,
        "view_sha256": meta["view_sha256"], "windows_sha256": common.sha256_file(view / "e4t_windows.json"),
        "planned_windows": sum(row["planned_windows"] for row in rows),
        "episodes": len(rows), "events": events_total, "events_inside_a_planned_window": covered_total,
        "coverage": (covered_total / events_total if events_total else None),
        "thresholds": {key: gates[key] for key in ("gripper_open_threshold_percent",
                                                   "gripper_closed_threshold_percent")},
        "reading": ("the key-event timing gate can only be measured on the covered transitions; the "
                    "rest are reported as unmeasurable rather than as a pass"),
        "plan_is_not_modified": True,
    }
    text = json.dumps({**summary, "per_episode": rows}, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("episodes", "planned_windows", "events",
                                              "events_inside_a_planned_window", "coverage")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
