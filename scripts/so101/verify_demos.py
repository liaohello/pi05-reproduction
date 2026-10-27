#!/usr/bin/env python3
"""E1 audit: official reader checks, alignment checks and physical replay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import so101_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/protocol.json")
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, default=None)
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()

    root_value = args.root or __import__("os").environ.get("SO101_SOURCE_ROOT")
    if not root_value:
        print("missing --root or SO101_SOURCE_ROOT", file=sys.stderr)
        return 2
    root = common.setup_external(root_value)
    protocol = common.load_json(args.protocol)
    config = common.load_external_config(root, protocol)
    collection = args.collection.expanduser().resolve()
    from bimanual.recording.collection import verify_collection
    from bimanual.tasks.sorting import regions

    certificate = regions.load_certificate(root / config["evaluation"]["certificate"])
    report = verify_collection(collection, config, repo_id=common.COLLECTION_REPO_ID,
                               collection=config["collection"], replay=not args.no_replay)
    official = report.get("official", {})
    replay = report.get("replay", {})
    checks = {
        "passed": bool(report.get("passed")),
        "episodes": int(report.get("episodes", 0)),
        "frames": int(report.get("bounds", {}).get("frames", 0)),
        "state_shape": official.get("state_shape"),
        "action_chunk_shape": official.get("action_chunk_shape"),
        "image_shapes": official.get("image_shapes"),
        "three_cameras": sorted(official.get("image_shapes", {})) == sorted(protocol["observation"]["cameras"]),
        "state_12": official.get("state_shape") == [12],
        "replay_total": replay.get("total"),
        "replay_passed": replay.get("passed"),
        "alignment_sampled": len(report.get("alignment", [])),
        "alignment_mismatches": len(report.get("alignment_mismatches", [])),
        "frame_count_mismatches": len(report.get("frame_count_mismatches", [])),
        "action_chunk_problems": len(report.get("action_chunk_problems", [])),
        "image_problems": len(report.get("image_problems", [])),
        "expert_source_sha256": report.get("expert_source_sha256"),
        "expert_revisions": report.get("expert_revisions"),
        "problems": report.get("problems", []),
        "certificate_hashes": certificate.hashes(),
    }
    checks["replay_all_passed"] = (replay.get("total") == protocol["collection"]["target_successes"]
                                   and replay.get("passed") == replay.get("total"))
    checks["ok"] = bool(checks["passed"] and checks["three_cameras"] and checks["state_12"]
                        and checks["alignment_mismatches"] == 0
                        and checks["frame_count_mismatches"] == 0
                        and checks["action_chunk_problems"] == 0
                        and checks["image_problems"] == 0
                        and (args.no_replay or checks["replay_all_passed"]))
    if args.public_output:
        common.write_json(args.public_output, {
            "schema_version": 1, "phase": "E1-audit",
            "episodes": checks["episodes"], "frames": checks["frames"],
            "state_shape": checks["state_shape"],
            "action_chunk_shape": checks["action_chunk_shape"],
            "cameras": sorted(official.get("image_shapes", {})),
            "replay_total": checks["replay_total"], "replay_passed": checks["replay_passed"],
            "alignment_mismatches": checks["alignment_mismatches"],
            "frame_count_mismatches": checks["frame_count_mismatches"],
            "action_chunk_problems": checks["action_chunk_problems"],
            "image_problems": checks["image_problems"],
            "expert_source_sha256": checks["expert_source_sha256"],
            "expert_revisions": checks["expert_revisions"],
            "certificate_hashes": checks["certificate_hashes"],
            "ok": checks["ok"],
            "claim": "scripted expert demonstrations audited and replayed; not a policy result",
        })
    print(json.dumps(checks, indent=2, default=str))
    return 0 if checks["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
