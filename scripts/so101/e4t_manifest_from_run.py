#!/usr/bin/env python3
"""Rebuild an E4-T collection manifest for a run that stopped before writing one.

A run only writes ``e4t_manifest.json`` when it reaches its target, but the episodes it already
saved are complete on disk: ``attempts.jsonl`` records every attempt with its scene fingerprint,
frame count, diagnostics directory and the dataset index the writer assigned. This tool turns that
index back into the manifest shape the audit and export steps expect, and marks it as rebuilt so
the provenance is never confused with an original run manifest.

It never invents an episode: only attempts whose writer index exists are included, and the episode
count must match what the LeRobot dataset reports.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True,
                        help="source_fingerprint.py JSON for the snapshot that produced the run")
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/e4t_protocol.json")
    parser.add_argument("--repo-id", default="local/so101_pi05_e4t_30")
    parser.add_argument("--output", type=Path, default=None,
                        help="defaults to <collection>/e4t_manifest.json and refuses to overwrite")
    args = parser.parse_args()

    collection = args.collection.expanduser().resolve()
    protocol = common.load_json(args.protocol)
    fingerprint = common.load_json(args.fingerprint)
    attempts_path = collection / "attempts.jsonl"
    attempts = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    saved = [row for row in attempts if row.get("source_episode_index") is not None]
    if not saved:
        raise SystemExit("the run has no saved episodes to index")

    dataset = collection / "dataset"
    dataset_files = sorted(path for path in dataset.rglob("*") if path.is_file())
    if not dataset_files:
        raise SystemExit(f"no dataset files under {dataset}")
    total_episodes = max(int(row["source_episode_index"]) for row in saved) + 1
    manifest = {
        "schema_version": 1, "phase": "E4-T-collection", "rebuilt_from_attempts": True,
        "rebuilt_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "created": (collection / "attempts.jsonl").stat().st_mtime,
        "repo_id": args.repo_id, "collection": str(collection),
        "protocol_sha256": common.sha256_json(protocol),
        "patch_sha256": protocol["source_patch"]["sha256"],
        "source_fingerprint": {"source_sha256": fingerprint["source_sha256"],
                              "expert_sha256": fingerprint["expert_sha256"],
                              "config_sha256": fingerprint.get("config_sha256"),
                              "model_sha256": fingerprint.get("model_sha256")},
        "dataset_sha256": common.sha256_json(
            [path.relative_to(dataset).as_posix() + ":" + str(path.stat().st_size)
             for path in dataset_files]),
        "dataset_file_count": len(dataset_files), "dataset_episode_count": total_episodes,
        "attempts": len(attempts), "successes": len(saved),
        "episodes": saved,
        "profiles": protocol["collection"]["target_successes_per_profile"],
        "claim": "index rebuilt from the retained attempt ledger; the split is assigned globally at export",
    }
    if len(saved) != total_episodes:
        raise SystemExit(f"saved attempts index {len(saved)} rows but the dataset reaches "
                         f"episode {total_episodes - 1}")
    output = (args.output or (collection / "e4t_manifest.json")).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite an existing manifest: {output}")
    common.write_json(output, manifest)
    counts: dict[str, int] = {}
    for row in saved:
        counts[row["profile"]] = counts.get(row["profile"], 0) + 1
    print(json.dumps({"manifest": str(output), "episodes": len(saved), "by_profile": counts,
                      "rebuilt": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
