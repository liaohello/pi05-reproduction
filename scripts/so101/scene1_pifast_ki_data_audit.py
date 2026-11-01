#!/usr/bin/env python3
"""CPU-only Scene1 data and quantile-normalization audit for the PI-FAST plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_FEATURES = [
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
]
EXPECTED_UNITS = ["degree"] * 5 + ["percent"] + ["degree"] * 5 + ["percent"]
STAT_FIELDS = ("mean", "std", "q01", "q99")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def per_dimension_stats(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 12 or not np.isfinite(array).all():
        raise ValueError("expected a finite [frames, 12] array")
    return {
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action_horizon < 1:
        raise ValueError("action horizon must be positive")

    meta_path = args.view / "meta.json"
    norm_path = args.view / "assets/so101/norm_stats.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    recorded = json.loads(norm_path.read_text(encoding="utf-8"))["norm_stats"]
    episodes = sorted(meta["episodes"], key=lambda row: int(row["episode_index"]))
    train = [row for row in episodes if row["split"] == "train"]
    validation = [row for row in episodes if row["split"] == "validation"]
    expected_counts = {
        "episodes": 64,
        "train": 52,
        "validation": 12,
        "train_red": 26,
        "train_blue": 26,
        "validation_red": 6,
        "validation_blue": 6,
    }
    actual_counts = {
        "episodes": len(episodes),
        "train": len(train),
        "validation": len(validation),
        "train_red": sum(row["color"] == "red" for row in train),
        "train_blue": sum(row["color"] == "blue" for row in train),
        "validation_red": sum(row["color"] == "red" for row in validation),
        "validation_blue": sum(row["color"] == "blue" for row in validation),
    }
    if actual_counts != expected_counts:
        raise ValueError(f"unexpected episode counts: {actual_counts}")
    if meta["state_action_dim"] != 12:
        raise ValueError("state/action dimension is not 12")
    if meta["feature_names"] != EXPECTED_FEATURES:
        raise ValueError("feature order differs from the SO101 contract")
    if meta["feature_units"] != EXPECTED_UNITS:
        raise ValueError("feature units differ from degree/percent")
    if int(meta["split"]["family_crosses"]) != 0:
        raise ValueError("scene families cross the train/validation split")

    def load(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        states, actions = [], []
        for row in rows:
            path = (
                args.view / "episodes"
                / f"episode_{int(row['episode_index']):03d}" / "episode.npz"
            )
            with np.load(path, allow_pickle=False) as data:
                state = np.asarray(data["state"], dtype=np.float64)
                action = np.asarray(data["action"], dtype=np.float64)
            expected_frames = int(row["frames"])
            if state.shape != (expected_frames, 12) or action.shape != (expected_frames, 12):
                raise ValueError(f"episode {row['episode_index']}: bad frame shape")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"episode {row['episode_index']}: non-finite values")
            states.append(state)
            actions.append(action)
        return np.concatenate(states), np.concatenate(actions)

    train_state, train_action = load(train)
    validation_state, validation_action = load(validation)
    recomputed = {
        "state": per_dimension_stats(train_state),
        "actions": per_dimension_stats(train_action),
    }
    matches = {
        key: all(
            np.allclose(recorded[key][field], recomputed[key][field],
                        rtol=1e-6, atol=1e-6)
            for field in STAT_FIELDS
        )
        for key in ("state", "actions")
    }
    windows = {
        "train": sum(max(0, int(row["frames"]) - args.action_horizon + 1) for row in train),
        "validation": sum(
            max(0, int(row["frames"]) - args.action_horizon + 1)
            for row in validation
        ),
    }
    by_color = {}
    for split, state, action in (
        ("train", train_state, train_action),
        ("validation", validation_state, validation_action),
    ):
        rows = train if split == "train" else validation
        for color in ("red", "blue"):
            selected = np.asarray([row["color"] == color for row in rows])
            frame_mask = np.concatenate([
                np.full(int(row["frames"]), row["color"] == color, dtype=bool)
                for row in rows
            ])
            by_color[f"{split}/{color}"] = {
                "episodes": int(selected.sum()),
                "frames": int(frame_mask.sum()),
                "windows": sum(
                    max(0, int(row["frames"]) - args.action_horizon + 1)
                    for row in rows if row["color"] == color
                ),
                "state": per_dimension_stats(state[frame_mask]),
                "actions": per_dimension_stats(action[frame_mask]),
            }
    payload = {
        "schema_version": 1,
        "status": "pass" if all(matches.values()) else "fail",
        "claim": "CPU-only per-frame train audit; chunk windows are counted but never used to weight norm stats",
        "view_sha256": meta["view_sha256"],
        "meta_sha256": sha256_file(meta_path),
        "norm_stats_sha256": sha256_file(norm_path),
        "counts": actual_counts,
        "family_crosses_split": 0,
        "frames": {
            "train": int(len(train_action)),
            "validation": int(len(validation_action)),
        },
        "action_horizon": args.action_horizon,
        "windows": windows,
        "feature_order": meta["feature_names"],
        "feature_units": meta["feature_units"],
        "normalization": {
            "grouping": "combined_train_red_blue_single_group",
            "method": "quantile_0.01_0.99",
            "fit_scope": "all 52 train episodes at original frame rate",
            "overlapping_chunks_reused_for_stats": False,
            "recomputed": recomputed,
            "recorded": recorded,
            "recorded_matches_recomputed": matches,
            "validation_used": False,
        },
        "per_color_distribution_audit": by_color,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "windows": windows,
        "norm_matches": matches,
    }, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
