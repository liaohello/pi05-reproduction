#!/usr/bin/env python3
"""Read-only current GPU admission check for the Scene1 PI-FAST run."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/deploy"))
sys.path.insert(0, str(REPO / "scripts/so101"))
import common  # noqa: E402
from resource_policy import memory_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-indices", default="0,1,2,3")
    parser.add_argument("--demand-mib", type=int, required=True)
    parser.add_argument("--margin-mib", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indices = [
        int(value) for value in args.gpu_indices.split(",") if value.strip()
    ]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("GPU indices must be non-empty and unique")
    cfg = common.load_config()
    query = ",".join(str(value) for value in indices)
    result = common.run(common.ssh_argv(cfg) + [
        f"nvidia-smi -i {query} --query-gpu=index,uuid,memory.total,"
        "memory.used,memory.free,utilization.gpu "
        "--format=csv,noheader,nounits"
    ], timeout=30)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-1000:])
    devices = []
    for line in result.stdout.strip().splitlines():
        index, uuid, total, used, free, util = [
            value.strip() for value in line.split(",", 5)]
        devices.append({
            "index": int(index),
            "uuid": uuid,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
            "utilization_percent_observed_not_gate": int(util),
        })
    payload = {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "requested_devices": indices,
        "devices": devices,
        "margin_mib": args.margin_mib,
        "demand_mib": args.demand_mib,
        "utilisation_is_not_an_admission_gate": True,
    }
    try:
        plan = memory_plan(
            devices,
            demand_mib=args.demand_mib,
            margin_mib=args.margin_mib,
        )
    except Exception as error:
        payload["status"] = "blocked"
        payload["reason"] = str(error)
    else:
        payload["status"] = "pass"
        payload["memory_plan"] = plan
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "free_mib": [device["memory_free_mib"] for device in devices],
        "required_mib": args.demand_mib + args.margin_mib,
        "reason": payload.get("reason"),
    }, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
