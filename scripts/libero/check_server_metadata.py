#!/usr/bin/env python3
"""Fail the launch unless the ready line advertises the expected policy metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PREFIX = "POLICY_SERVER_READY "


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    args = parser.parse_args()

    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    line = next(
        row for row in args.readiness.read_text(encoding="utf-8").splitlines()
        if row.startswith(PREFIX)
    )
    metadata = json.loads(line[len(PREFIX):])
    expected = {
        "model_id": "pi05_libero_bowl_plate_49_lora",
        "checkpoint_step": args.expected_step,
        "action_horizon": int(protocol["data"]["action_horizon"]),
        "task_id": int(protocol["task_id"]),
        "task": protocol["task"],
        "protocol_sha256": protocol_sha,
        "norm_stats_sha256": protocol["data"]["norm_stats_sha256"],
        "suite": protocol["suite"],
        "action_dim": int(protocol["data"]["action_dim"]),
        "state_dim": int(protocol["data"]["state_dim"]),
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items() if metadata.get(key) != value
    }
    if mismatches:
        print("SERVER_METADATA_MISMATCH " + json.dumps(mismatches, sort_keys=True))
        return 1
    print("SERVER_METADATA_OK " + json.dumps(
        {key: metadata[key] for key in expected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
