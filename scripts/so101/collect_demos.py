#!/usr/bin/env python3
"""E1: collect 20 successful direct-sorting SO101 demonstrations.

Runs the external scripted physical expert in the pinned snapshot, keeps every
attempt, freezes the train/validation split and generates the 20 evaluation
scenes before any learning starts.  All artifacts stay in the run directory;
``--public-manifest`` writes the sanitised hash-only record.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import so101_common as common  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _source_fingerprint(root: Path) -> dict:
    import source_fingerprint

    return source_fingerprint.fingerprint(root, python=sys.executable)


def _public_payload(output: Path, protocol: dict, config: dict, manifest: dict,
                    fingerprint: dict, protocol_sha: str, config_sha: str,
                    dataset_info_sha: str) -> dict:
    return {
        "schema_version": 1,
        "created": manifest["created"],
        "phase": "E1",
        "claim": "20 successful scripted-expert simulations; not a learned-policy success rate",
        "protocol_sha256": protocol_sha,
        "source_sha256": fingerprint["source_sha256"],
        "expert_sha256": fingerprint["expert_sha256"],
        "config_sha256": config_sha,
        "dataset_info_sha256": dataset_info_sha,
        "collection": {
            "direct_seed_range": protocol["collection"]["direct_seed_range"],
            "target_successes": protocol["collection"]["target_successes"],
            "maximum_attempts": protocol["collection"]["maximum_attempts"],
            "attempts_used": manifest["attempts_used"],
            "successes": manifest["successes"],
            "failures": manifest["failures"],
            "failure_reasons": manifest["failure_reasons"],
        },
        "split": manifest["split"],
        "episodes": [
            {key: row[key] for key in ("episode_index", "seed", "scene_sha256", "split",
                                       "source_episode_index", "frames")}
            for row in manifest["episodes"]
        ],
        "evaluation_scenes": manifest["evaluation_scenes"],
        "evaluation_manifest_sha256": manifest["evaluation_manifest_sha256"],
        "control": protocol["control"],
        "cameras": protocol["observation"]["cameras"],
        "state_action_dim": protocol["observation"]["state"]["shape"][0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="external SO101 checkout (default: $SO101_SOURCE_ROOT)")
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, default=None)
    parser.add_argument("--pilot", action="store_true",
                        help="one-success smoke of the collection path; never part of the E1 dataset")
    parser.add_argument("--pilot-seeds", default="8090,8091")
    args = parser.parse_args()

    root_value = args.root or __import__("os").environ.get("SO101_SOURCE_ROOT")
    if not root_value:
        print("missing --root or SO101_SOURCE_ROOT", file=sys.stderr)
        return 2
    root = Path(root_value).expanduser().resolve()
    protocol = common.load_json(args.protocol)
    protocol_sha = common.sha256_file(args.protocol)
    config = common.load_external_config(root, protocol)

    from bimanual.recording.collection import collect
    from bimanual.tasks.sorting import regions
    from bimanual.tasks.sorting.certification import check_certificate_current
    from bimanual.tasks.sorting.scene_sampling import generate_scene
    from bimanual.sim.environment import BimanualEnv

    output = args.output.expanduser().resolve()
    if output.exists():
        print(f"refusing to reuse existing run directory: {output}", file=sys.stderr)
        return 2

    cert_path = root / config["evaluation"]["certificate"]
    certificate = regions.load_certificate(cert_path)
    collection = config["collection"]
    if args.pilot:
        pilot_seeds = [int(value) for value in args.pilot_seeds.split(",") if value.strip()]
        if len(pilot_seeds) != 2:
            raise SystemExit("--pilot-seeds must contain two comma-separated seeds")
        collection = dict(collection)
        collection["direct_seeds"] = pilot_seeds
        plan = [("direct", seed) for seed in pilot_seeds]
        target, attempts = 1, 2
    else:
        start, stop = protocol["collection"]["direct_seed_range"]
        plan = [("direct", seed) for seed in range(start, stop + 1)]
        target = int(protocol["collection"]["target_successes"])
        attempts = int(protocol["collection"]["maximum_attempts"])

    print(json.dumps({"phase": "E1", "pilot": args.pilot, "output": str(output),
                      "plan_attempts": len(plan), "target_successes": target,
                      "seed_range": list(collection["direct_seeds"])}, indent=2))
    summary = collect(config, output, plan=plan, collection=collection,
                      target=target, attempts=attempts,
                      repo_id=common.COLLECTION_REPO_ID, certificate=certificate,
                      diagnostic=True)

    successes = [row for row in summary["attempts"] if row["success"]]
    if args.pilot:
        pilot = {"phase": "E1-pilot", "pilot_seeds": pilot_seeds,
                 "attempts": summary["profiles"]["direct"]["attempts"],
                 "successes": len(successes),
                 "failure_reasons": summary["profiles"]["direct"]["failure_reasons"],
                 "note": "pilot only; never included in the E1 dataset"}
        common.write_json(output / "pilot_summary.json", pilot)
        print(json.dumps(pilot, indent=2))
        return 0 if successes else 1
    target = int(protocol["collection"]["target_successes"])
    if len(successes) != target:
        print(json.dumps({"phase": "E1", "status": "incomplete",
                          "successes": len(successes), "target": target,
                          "failure_reasons": summary["profiles"]["direct"]["failure_reasons"]},
                         indent=2), file=sys.stderr)
        return 1
    if len({row["seed"] for row in successes}) != target:
        raise RuntimeError("duplicate seed among successful demonstrations")

    train_count = int(protocol["collection"]["split"]["train"])
    episodes = []
    for index, row in enumerate(successes):
        split = "train" if index < train_count else "validation"
        episodes.append({
            "episode_index": index,
            "seed": int(row["seed"]),
            "scene_sha256": row["scene_sha256"],
            "source_episode_index": index,
            "split": split,
            "frames": int(row["frames"]),
            "attempt": row,
        })
    split = {
        "rule": protocol["collection"]["split"]["rule"],
        "train": [row["episode_index"] for row in episodes if row["split"] == "train"],
        "validation": [row["episode_index"] for row in episodes if row["split"] == "validation"],
        "train_seeds": [row["seed"] for row in episodes if row["split"] == "train"],
        "validation_seeds": [row["seed"] for row in episodes if row["split"] == "validation"],
    }
    if len(split["train"]) != train_count or len(split["validation"]) != target - train_count:
        raise RuntimeError("split size mismatch")

    with BimanualEnv(config, images=False) as env:
        stale = check_certificate_current(certificate, env)
        if stale:
            raise RuntimeError("certificate is stale: " + "; ".join(stale))
        eval_start, eval_stop = protocol["evaluation"]["seed_range"]
        if not (eval_stop - eval_start + 1 == protocol["evaluation"]["episodes"]):
            raise RuntimeError("evaluation seed range does not match the episode budget")
        if set(range(eval_start, eval_stop + 1)) & set(range(start, stop + 1)):
            raise RuntimeError("evaluation seeds overlap the collection pool")
        evaluation_scenes = []
        for seed in range(eval_start, eval_stop + 1):
            scene = generate_scene(config, certificate, "direct", seed, env=env,
                                   sampling=str(config["sorting"].get("sampling_mode", "legacy")))
            evaluation_scenes.append({"seed": seed, "sha256": scene.sha256,
                                      "summary": scene.summary, "scene": scene.as_dict()})

    evaluation_manifest = {
        "purpose": protocol["evaluation"]["purpose"],
        "seed_range": [eval_start, eval_stop],
        "scenes": evaluation_scenes,
        "certificate_hashes": certificate.hashes(),
        "config_sha256": common.sha256_json(config),
        "not_used_for": protocol["evaluation"]["not_used_for"],
    }
    evaluation_manifest_sha = common.sha256_json(evaluation_manifest)

    fingerprint = _source_fingerprint(root)
    config_sha = common.sha256_json({key: value for key, value in config.items() if key != "root"})
    dataset_info = output / "dataset" / "meta" / "info.json"
    dataset_info_sha = common.sha256_file(dataset_info)
    failure_reasons = summary["profiles"]["direct"]["failure_reasons"]
    manifest = {
        "schema_version": 1,
        "phase": "E1",
        "created": _now(),
        "protocol_path": "configs/so101/protocol.json",
        "protocol_sha256": protocol_sha,
        "source_fingerprint": fingerprint,
        "config_sha256": config_sha,
        "dataset_info_sha256": dataset_info_sha,
        "repo_id": common.COLLECTION_REPO_ID,
        "attempts_used": int(summary["profiles"]["direct"]["attempts"]),
        "successes": len(successes),
        "failures": int(summary["profiles"]["direct"]["failures"]),
        "failure_reasons": failure_reasons,
        "episodes": episodes,
        "split": split,
        "evaluation_scenes": [{"seed": row["seed"], "sha256": row["sha256"],
                               "summary": row["summary"]} for row in evaluation_scenes],
        "evaluation_manifest": evaluation_manifest,
        "evaluation_manifest_sha256": evaluation_manifest_sha,
        "statistics_scope": "train split only",
        "claim": protocol["claims"]["engineering_chain"],
        "not_claimed": protocol["claims"]["explicitly_not"],
    }
    common.write_json(output / "split.json", split)
    common.write_json(output / "eval_scenes.json", evaluation_manifest)
    common.write_json(output / "e1_manifest.json", manifest)
    if args.public_manifest:
        public = _public_payload(output, protocol, config, manifest, fingerprint,
                                 protocol_sha, config_sha, dataset_info_sha)
        common.write_json(args.public_manifest, public)
    print(json.dumps({"phase": "E1", "status": "complete", "successes": len(successes),
                      "attempts": manifest["attempts_used"], "failures": manifest["failures"],
                      "split": {"train": len(split["train"]), "validation": len(split["validation"])},
                      "source_sha256": fingerprint["source_sha256"],
                      "config_sha256": config_sha,
                      "dataset_info_sha256": dataset_info_sha,
                      "evaluation_manifest_sha256": evaluation_manifest_sha}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
