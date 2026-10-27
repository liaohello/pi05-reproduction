#!/usr/bin/env python3
"""Check the open-loop activity floor against the expert action stream it is meant to classify.

The floor decides which validation windows count as "active" in the gate, so it must sit above
recording noise and below every deliberate expert motion.  Both sides are measured here on the
40 train episodes only: the per-frame joint and gripper deltas of the expert actions.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402

REPO = common.REPO_ROOT
JOINT = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
GRIPPER = [5, 11]


def distribution(values: np.ndarray) -> dict:
    return {
        "frames": int(values.size),
        "max": float(values.max()) if values.size else None,
        "p50": float(np.percentile(values, 50)) if values.size else None,
        "p90": float(np.percentile(values, 90)) if values.size else None,
        "p99": float(np.percentile(values, 99)) if values.size else None,
        "exact_zero_share": float(np.mean(values == 0.0)) if values.size else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--training-config", type=Path,
                        default=common.REPO_ROOT / "configs/so101/e4t_training.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    view = args.view.expanduser().resolve()
    meta = common.load_json(view / "meta.json")
    gate = common.load_json(args.training_config)["validation_gate"]
    joint_floor = float(gate["activity_joint_deg_per_frame"])
    gripper_floor = float(gate["activity_gripper_pct_per_frame"])

    joint_deltas, gripper_deltas, exact = [], [], 0
    episodes = [row for row in meta["episodes"] if row["split"] == "train"]
    for row in episodes:
        path = view / "episodes" / f"episode_{int(row['episode_index']):03d}" / "episode.npz"
        with np.load(path, allow_pickle=False) as data:
            action = np.asarray(data["action"], dtype=np.float64)
        delta = np.abs(np.diff(action, axis=0))
        joint_deltas.append(delta[:, JOINT].max(axis=1))
        gripper_deltas.append(delta[:, GRIPPER].max(axis=1))
        exact += int(np.count_nonzero(delta[:, JOINT].max(axis=1) == 0.0))
    if not joint_deltas:
        raise SystemExit("the train split contributed no frames")
    joint = np.concatenate(joint_deltas)
    gripper = np.concatenate(gripper_deltas)
    control_interval_s = 1.0 / float(meta.get("fps", 50))
    report = {
        "schema_version": 1, "branch": "so101", "phase": "E3", "substage": "E4-T-gate-calibration",
        "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "scope": "40 train episodes only", "train_episodes": len(episodes),
        "view_sha256": meta["view_sha256"],
        "floors": {"joint_deg_per_frame": joint_floor, "gripper_pct_per_frame": gripper_floor,
                   "declared_in": str(args.training_config.relative_to(REPO))},
        "expert_joint_delta_per_frame_deg": distribution(joint),
        "expert_gripper_delta_per_frame_pct": distribution(gripper),
        "frames_with_no_joint_change": exact,
        "share_of_frames_below_joint_floor": float(np.mean(joint < joint_floor)),
        "share_of_frames_below_gripper_floor": float(np.mean(gripper < gripper_floor)),
        "joint_floor_as_deg_per_second": joint_floor / control_interval_s,
        "reading": ("the floor is compared with the largest non-zero expert delta and with the share of "
                    "truly frozen frames: it must be far below deliberate motion and far above zero"),
        "geometric_margin_note": gate.get("fk_tcp_error_mib", "no FK margin reported this round"),
        "conclusion": "",
    }
    nonzero_joint = joint[joint > 0.0]
    above_floor = float(np.mean(joint >= joint_floor))
    report["share_of_moving_frames_above_joint_floor"] = (
        float(np.mean(nonzero_joint >= joint_floor)) if nonzero_joint.size else None)
    report["conclusion"] = (
        "accepted" if (nonzero_joint.size and above_floor > 0.0
                       and report["expert_joint_delta_per_frame_deg"]["p50"] > joint_floor
                       and report["share_of_moving_frames_above_joint_floor"] >= 0.5)
        else "review: the floor does not separate frozen from moving expert frames")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["conclusion"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
