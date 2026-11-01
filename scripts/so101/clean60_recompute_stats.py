#!/usr/bin/env python3
"""Recompute clean60 statistics from the frozen 48-train split only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import clean60_contracts as clean60  # noqa: E402
import so101_common as common  # noqa: E402


def stats(arrays: np.ndarray) -> dict:
    return {
        "mean": np.mean(arrays, axis=0).astype(float).tolist(),
        "std": np.std(arrays, axis=0).astype(float).tolist(),
        "q01": np.quantile(arrays, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(arrays, 0.99, axis=0).astype(float).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=60)
    parser.add_argument("--expected-train", type=int, default=48)
    parser.add_argument("--expected-validation", type=int, default=12)
    parser.add_argument("--dataset-label", default="clean60")
    args = parser.parse_args()
    view = args.view.expanduser().resolve()
    meta = common.load_json(view / "meta.json")
    binding = common.load_json(view / "protocol_binding.json")
    split = common.load_json(view / "splits.json")
    counts = (args.expected_episodes, args.expected_train, args.expected_validation)
    problems = clean60.view_contract_problems(
        meta, protocol_binding=binding, expected_counts=counts, require_clean=True)
    if problems:
        raise SystemExit("refusing to recompute for an invalid view: " + "; ".join(problems))
    train_ids = sorted(int(row["episode_index"]) for row in meta["episodes"]
                       if row["split"] == "train")
    validation_ids = sorted(int(row["episode_index"]) for row in meta["episodes"]
                            if row["split"] == "validation")
    if (len(train_ids), len(validation_ids)) != (args.expected_train,
                                                 args.expected_validation):
        raise SystemExit("statistics scope differs from the registered split counts")
    if counts == (60, 48, 12) and (train_ids != list(range(48))
                                   or validation_ids != list(range(48, 60))):
        raise SystemExit("statistics scope is not the frozen first-48/last-12 split")
    states, actions = [], []
    for episode_id in train_ids:
        with np.load(view / "episodes" / f"episode_{episode_id:03d}" / "episode.npz",
                     allow_pickle=False) as data:
            state = np.asarray(data["state"], dtype=np.float64)
            action = np.asarray(data["action"], dtype=np.float64)
        if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 12:
            raise SystemExit(f"episode {episode_id}: invalid aligned arrays")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise SystemExit(f"episode {episode_id}: non-finite arrays")
        states.append(state)
        actions.append(action)
    validation_frames = sum(
        int(row["frames"]) for row in meta["episodes"] if row["split"] == "validation")
    state = np.concatenate(states)
    action = np.concatenate(actions)
    new_stats = {"state": stats(state), "actions": stats(action)}
    constant = {
        "state": [index for index, value in enumerate(new_stats["state"]["std"])
                  if float(value) == 0.0],
        "actions": [index for index, value in enumerate(new_stats["actions"]["std"])
                    if float(value) == 0.0],
    }
    near_constant = {
        "state": [index for index, value in enumerate(new_stats["state"]["std"])
                  if float(value) < 1e-6],
        "actions": [index for index, value in enumerate(new_stats["actions"]["std"])
                    if float(value) < 1e-6],
    }
    if constant["state"] or constant["actions"] or near_constant["state"] or near_constant["actions"]:
        raise SystemExit("refusing constant/near-constant state or action dimensions")

    old_norm_path = view / "assets/so101/norm_stats.json"
    old_norm = common.load_json(old_norm_path)
    old_stats = old_norm["norm_stats"]
    differences = {
        split_name: {
            field: float(np.max(np.abs(
                np.asarray(old_stats[split_name][field], dtype=np.float64)
                - np.asarray(new_stats[split_name][field], dtype=np.float64))))
            for field in ("mean", "std", "q01", "q99")
        }
        for split_name in ("state", "actions")
    }
    norm_payload = {"norm_stats": new_stats}
    canonical = {
        split_name: {field: new_stats[split_name][field]
                     for field in ("mean", "std", "q01", "q99")}
        for split_name in ("state", "actions")
    }
    common.write_json(old_norm_path, norm_payload)
    norm_sha = common.sha256_file(old_norm_path)
    canonical_sha = common.sha256_json(canonical)
    train_stats = common.load_json(view / "train_stats.json")
    train_stats.update({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "scope": "train split only",
        "train_episodes": train_ids,
        "validation_episodes": validation_ids,
        "train_frames": int(state.shape[0]),
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "constant_dimensions": constant,
        "near_constant_dimensions_std_lt_1e-6": near_constant,
        "norm_stats_sha256": norm_sha,
        "norm_stats_canonical_sha256": canonical_sha,
        "stats": new_stats,
        "source": f"recomputed from the frozen {args.dataset_label} train episodes only",
    })
    common.write_json(view / "train_stats.json", train_stats)
    meta.update({
        "norm_stats_sha256": norm_sha,
        "norm_stats_canonical_sha256": canonical_sha,
        "statistics_revision": (
            "clean60_train48_recomputed_v1" if args.dataset_label == "clean60"
            else f"{args.dataset_label}_train{args.expected_train}_recomputed_v1"),
    })
    common.write_json(view / "meta.json", meta)
    admission = common.load_json(view / "e4t_admission.json")
    admission.update({
        "view_sha256": meta["view_sha256"],
        "train_stats_sha256": common.sha256_file(view / "train_stats.json"),
        "norm_stats_sha256": norm_sha,
        "norm_stats_canonical_sha256": canonical_sha,
    })
    admission.setdefault("data_gate", {})["statistics_source"] = (
        "recomputed from the frozen 48 clean train episodes")
    common.write_json(view / "e4t_admission.json", admission)
    report = {
        "schema_version": 1,
        "view": view.name,
        "view_sha256": meta["view_sha256"],
        "train_episodes": train_ids,
        "validation_episodes_excluded": validation_ids,
        "train_frames": int(state.shape[0]),
        "validation_frames_excluded": validation_frames,
        "old_norm_stats_sha256": clean60.sha256_json(old_stats),
        "new_norm_stats_sha256": norm_sha,
        "new_norm_stats_canonical_sha256": canonical_sha,
        "old_vs_new_max_abs": differences,
        "near_constant_dimensions_std_lt_1e-6": near_constant,
        "stats": new_stats,
        "passed": True,
    }
    common.write_json(view / "stats_recompute.json", report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"stats", "train_episodes", "validation_episodes_excluded"}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
