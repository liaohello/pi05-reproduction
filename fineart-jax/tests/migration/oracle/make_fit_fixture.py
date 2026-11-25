#!/usr/bin/env python3
"""Build the deterministic FAST-fitting fixture.

Writes the same table twice: a LeRobot-shaped parquet shard for the reference
loader and an ``.npz`` for the candidate, which must not need pyarrow at this
stage. ``export_g1_data.py`` cross-checks the two representations before any
golden is trusted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tests.migration import g1_cases as cases  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    shard_dir = out / "dataset" / "data" / "chunk-000"
    shard_dir.mkdir(parents=True, exist_ok=True)

    rows, states, episodes = cases.fit_action_table()
    actions = np.asarray(rows, dtype=np.float32)
    state_array = np.asarray(states, dtype=np.float32)
    episode_array = np.asarray(episodes, dtype=np.int64)
    index_array = np.arange(len(actions), dtype=np.int64)

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "index": pa.array(index_array, type=pa.int64()),
            "episode_index": pa.array(episode_array, type=pa.int64()),
            "action": pa.array([row.tolist() for row in actions], type=pa.list_(pa.float32())),
            "observation.state": pa.array([row.tolist() for row in state_array], type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, shard_dir / "file-000.parquet")

    npz_path = out / "actions.npz"
    np.savez(str(npz_path), action=actions, state=state_array, episode_index=episode_array)

    # Read the parquet back and prove it carries the same numbers.
    read_back = pq.read_table(shard_dir / "file-000.parquet", columns=["episode_index", "action", "observation.state"])
    parquet_actions = np.stack(read_back["action"].to_numpy(zero_copy_only=False)).astype(np.float32)
    parquet_states = np.stack(read_back["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
    parquet_episodes = read_back["episode_index"].to_numpy()
    consistent = (
        np.array_equal(parquet_actions, actions)
        and np.array_equal(parquet_states, state_array)
        and np.array_equal(parquet_episodes, episode_array)
    )
    if not consistent:
        raise SystemExit("parquet fixture does not round-trip to the same arrays")

    manifest = {
        "rows": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "episodes": int(cases.FIT_EPISODES),
        "frames_per_episode": int(cases.FIT_FRAMES_PER_EPISODE),
        "parquet_sha256": hashlib.sha256((shard_dir / "file-000.parquet").read_bytes()).hexdigest(),
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "actions_sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
        "states_sha256": hashlib.sha256(np.ascontiguousarray(state_array).tobytes()).hexdigest(),
        "parquet_matches_npz": bool(consistent),
        "excluded_episodes": list(cases.FIT_EXCLUDE_EPISODES),
    }
    (out / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
