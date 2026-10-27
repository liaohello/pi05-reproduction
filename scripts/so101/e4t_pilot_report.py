#!/usr/bin/env python3
"""Print the retained attempt ledger of an E4-T collection or pilot run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    path = args.run / "attempts.jsonl"
    if not path.is_file():
        print(json.dumps({"run": str(args.run), "state": "no_attempts_recorded"}))
        return 1
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        audit = row.get("recovery_audit") or {}
        event = row.get("perturbation") or {}
        print(json.dumps({
            "profile": row.get("profile"), "seed": row.get("seed"),
            "success": row.get("success"), "failure": row.get("raw_failure_reason")
            or row.get("failure_reason"), "mechanism": event.get("mechanism"),
            "category": event.get("category"), "trigger_frame": event.get("control_frame"),
            "audit_valid": audit.get("valid"), "audit_reasons": audit.get("reasons"),
            "frame_xy_m": None if audit.get("frame_xy_displacement_m") is None
            else round(float(audit["frame_xy_displacement_m"]), 5),
            "correction": audit.get("correction_evidence"), "frames": row.get("frames"),
        }, ensure_ascii=False))
    summary = {
        "run": str(args.run), "attempts": len(rows),
        "successes": sum(1 for row in rows if row.get("success")),
        "audited_recovery_successes": sum(1 for row in rows if row.get("success")
                                          and (row.get("recovery_audit") or {}).get("valid")),
        "protocol_sha256": (common.load_json(args.run / "e4t_manifest.json").get("protocol_sha256")
                            if (args.run / "e4t_manifest.json").is_file() else None),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
