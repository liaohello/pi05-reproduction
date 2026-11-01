#!/usr/bin/env python3
"""Build the all-49 LIBERO goal/task-8 training view and Norm Stats.

The source view already identifies the 49 target episodes and records the
Parquet paths.  This builder deliberately does not inherit its 39/10 split:
the requested experiment has no validation split and computes normalization
statistics from all 49 selected episodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_EPISODES = [
    379, 422, 426, 431, 433, 447, 448, 451, 459, 466,
    481, 483, 488, 507, 511, 513, 522, 532, 537, 549,
    551, 563, 568, 582, 607, 615, 620, 621, 626, 634,
    639, 642, 646, 653, 655, 670, 679, 708, 716, 718,
    726, 727, 749, 750, 768, 770, 801, 803, 806,
]
TASK = "put the bowl on the plate"
TASK_SUITE = "libero_goal"
TASK_ID = 8
DATASET_TASK_INDEX = 10
ACTION_HORIZON = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_source_view(path: Path, expected_sha256: str) -> dict:
    document = json.loads((path / "view.json").read_text(encoding="utf-8"))
    claimed = document.get("view_sha256")
    payload = {key: value for key, value in document.items() if key != "view_sha256"}
    if not claimed or canonical_sha256(payload) != claimed:
        raise ValueError("source view SHA256 is invalid")
    if claimed != expected_sha256:
        raise ValueError("source view differs from the expected SHA256")
    return document


def json_stats(values) -> dict:
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": np.asarray(values.mean(axis=0), dtype=float).tolist(),
        "std": np.asarray(values.std(axis=0), dtype=float).tolist(),
        "q01": np.asarray(np.quantile(values, 0.01, axis=0), dtype=float).tolist(),
        "q99": np.asarray(np.quantile(values, 0.99, axis=0), dtype=float).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-view", type=Path, required=True)
    parser.add_argument("--expected-source-view-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import pyarrow.parquet as pq

    source = load_source_view(args.source_view, args.expected_source_view_sha256)
    if (
        source.get("suite") != TASK_SUITE
        or int(source.get("task_id", -1)) != TASK_ID
        or int(source.get("dataset_task_index", -1)) != DATASET_TASK_INDEX
        or source.get("task") != TASK
    ):
        raise ValueError("source view is not the registered LIBERO task")

    rows = sorted(source.get("episodes", []), key=lambda row: int(row["episode_index"]))
    episode_ids = [int(row["episode_index"]) for row in rows]
    if episode_ids != EXPECTED_EPISODES:
        raise ValueError("source view episode IDs differ from the requested 49 episodes")
    if source.get("train_episode_ids") is not None:
        source_ids = {
            int(value)
            for value in source.get("train_episode_ids", [])
            + source.get("validation_episode_ids", [])
        }
        if source_ids != set(EXPECTED_EPISODES):
            raise ValueError("source view splits do not cover exactly the requested episodes")

    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {args.out}")
    norm_dir = args.out / "assets" / "libero_goal8"
    norm_dir.mkdir(parents=True, exist_ok=True)

    states = []
    actions = []
    output_rows = []
    total_frames = 0
    for source_row in rows:
        episode = int(source_row["episode_index"])
        parquet = args.data_root / source_row["parquet"]
        if not parquet.is_file():
            raise FileNotFoundError(f"missing Parquet for episode {episode}")
        if sha256_file(parquet) != source_row["parquet_sha256"]:
            raise ValueError(f"Parquet SHA256 changed for episode {episode}")
        table = pq.read_table(
            parquet,
            columns=["episode_index", "frame_index", "task_index", "state", "actions"],
        )
        present_episodes = set(table["episode_index"].to_pylist())
        present_tasks = set(table["task_index"].to_pylist())
        frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        expected_frames = int(source_row["frames"])
        if present_episodes != {episode}:
            raise ValueError(f"episode IDs changed in Parquet for {episode}")
        if present_tasks != {DATASET_TASK_INDEX}:
            raise ValueError(f"task index changed in Parquet for {episode}")
        if state.shape != (expected_frames, 8) or action.shape != (expected_frames, 7):
            raise ValueError(f"state/action shape changed for episode {episode}")
        if not np.array_equal(frame_indices, np.arange(expected_frames)):
            raise ValueError(f"frame indices are not contiguous for episode {episode}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"non-finite state/action value in episode {episode}")
        states.append(state)
        actions.append(action)
        total_frames += expected_frames
        row = dict(source_row)
        row["split"] = "train"
        output_rows.append(row)

    if total_frames != 4563:
        raise ValueError(f"expected 4563 selected frames, found {total_frames}")
    all_state = np.concatenate(states, axis=0)
    all_action = np.concatenate(actions, axis=0)
    norm = {
        "norm_stats": {
            "state": json_stats(all_state),
            "actions": json_stats(all_action),
        }
    }
    norm_path = norm_dir / "norm_stats.json"
    norm_path.write_text(
        json.dumps(norm, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    view = dict(source)
    view.pop("view_sha256", None)
    view["schema_version"] = 2
    view["view_id"] = f"{source['view_id']}-all49-h{ACTION_HORIZON}"
    view["source_view_sha256"] = source["view_sha256"]
    view["action_horizon"] = ACTION_HORIZON
    view["train_episode_ids"] = EXPECTED_EPISODES
    view["validation_episode_ids"] = []
    view["split_mode"] = "all_selected_training"
    view["episodes"] = output_rows
    view["norm_stats"] = {
        "path": "assets/libero_goal8/norm_stats.json",
        "sha256": sha256_file(norm_path),
        "selected_episode_count": len(EXPECTED_EPISODES),
        "selected_frame_count": total_frames,
    }
    view["constraints"] = {
        **source.get("constraints", {}),
        "complete_episode_split": False,
        "validation_split_absent": True,
        "all_selected_episodes_in_norm_stats": True,
        "validation_in_norm_stats": False,
        "extra_delta_transform": False,
        "action_semantics": "delta end-effector",
    }
    view["view_sha256"] = canonical_sha256(view)
    (args.out / "view.json").write_text(
        json.dumps(view, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    stats = {
        "split": "all_selected_training",
        "episodes": len(EXPECTED_EPISODES),
        "frames": total_frames,
        "episode_ids": EXPECTED_EPISODES,
        "state": json_stats(all_state),
        "actions": json_stats(all_action),
        "norm_stats_sha256": sha256_file(norm_path),
    }
    (args.out / "train_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "view": str(args.out),
        "view_sha256": view["view_sha256"],
        "norm_stats_sha256": view["norm_stats"]["sha256"],
        "episodes": len(EXPECTED_EPISODES),
        "frames": total_frames,
        "action_horizon": ACTION_HORIZON,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
