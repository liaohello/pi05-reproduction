#!/usr/bin/env python3
"""Rebuild the E4-T admission document from a frozen training view.

The admission is a seal over artifacts that already exist: the exported view meta, the per-shard
audits and the aggregate replay of the reused episodes. This script derives the same document as
the exporter does, from the same function, and refuses to write it when the view, the audits or the
replay evidence contradict each other.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402
import export_e4t_view as exporter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--old-replay", type=Path, required=True)
    parser.add_argument("--new-collection", type=Path, action="append", required=True)
    parser.add_argument("--new-audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--replace-unsigned", action="store_true",
                        help="overwrite an existing admission that the training entry point would refuse")
    args = parser.parse_args()

    if len(args.new_collection) != len(args.new_audit):
        raise SystemExit("every --new-collection needs exactly one --new-audit")
    view = args.view.expanduser().resolve()
    meta = common.load_json(view / "meta.json")
    for field in ("recovery_category_note", "audit_excluded_by_protocol", "old_episode_replay"):
        if field not in meta:
            raise SystemExit(f"view meta lacks '{field}': this view was exported before the replay "
                             f"evidence became an input, so re-export it instead of sealing it")
    replay_path = args.old_replay.expanduser().resolve()
    replay = common.load_json(replay_path)
    if not replay.get("ok") or int(replay.get("replay_passed", -1)) != int(replay.get("replay_total", -1)):
        raise SystemExit(f"reused episodes did not all replay: "
                         f"{replay.get('replay_passed')}/{replay.get('replay_total')}")
    old_replay_record = {
        "episodes": int(replay["replay_total"]), "replayed": int(replay["replay_passed"]),
        "frames": int(replay.get("frames", 0)),
        "verification_sha256": common.sha256_file(replay_path),
        "expert_source_sha256": replay.get("expert_source_sha256"),
        "certificate_hashes": replay.get("certificate_hashes"),
    }
    collections = [path.expanduser().resolve() for path in args.new_collection]
    audit_rows = {}
    for audit_path in args.new_audit:
        document = common.load_json(audit_path.expanduser().resolve())
        if not document.get("passed"):
            raise SystemExit(f"shard audit did not pass: {audit_path}")
        for row in document.get("episodes", []):
            audit_rows[row["attempt_id"]] = row

    admission = exporter.make_admission(
        meta=meta, meta_path=view / "meta.json", collections=collections,
        new_audits=args.new_audit, audit_rows=audit_rows, old_replay_record=old_replay_record)
    problems = exporter.admission_contract_problems(admission)
    if problems:
        raise SystemExit("admission refused by the training entry point: " + "; ".join(problems))
    if admission["view_sha256"] != meta["view_sha256"]:
        raise SystemExit("admission view hash differs from the exported view")
    if admission["totals"]["episodes"] != len(meta["episodes"]):
        raise SystemExit("admission episode total differs from the exported view")

    output = (args.output or (view / "e4t_admission.json")).expanduser().resolve()
    if output.is_file():
        existing = common.load_json(output)
        existing_problems = exporter.admission_contract_problems(existing)
        if not args.replace_unsigned:
            raise SystemExit(f"{output} already exists; pass --replace-unsigned only when the "
                             f"training entry point refuses it (its problems: {existing_problems})")
        if existing_problems:
            admission["replaced_document"] = {"sha256": common.sha256_file(output),
                                              "problems": existing_problems}
        else:
            raise SystemExit(f"{output} already satisfies the training contract; refusing to overwrite")
    common.write_json(output, admission)
    print(json.dumps({"output": str(output), "sha256": common.sha256_file(output),
                      "totals": admission["totals"],
                      "excluded_by_protocol": admission["data_gate"]["excluded_by_protocol"],
                      "old_episode_replay": admission["data_gate"]["old_episode_replay"],
                      "replaced_document": admission.get("replaced_document")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
