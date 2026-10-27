#!/usr/bin/env python3
"""E3: one short closed loop plus a frozen 20-scene diagnostic.

The external ``PolicyRunner`` remains the only owner of ``env.step``.  This
script owns the protocol: it freezes scenes before rollouts, loads exactly the
E2 checkpoint through the policy server, records every attempt and reports the
success count with a Wilson interval.  It is an engineering diagnostic, not a
benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import so101_common as common  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _scene_from_entry(entry: dict):
    from bimanual.tasks.sorting.scene_sampling import SceneSpec

    scene = SceneSpec.from_dict(entry["scene"])
    if not scene.sha256 or scene.digest() != scene.sha256:
        raise ValueError(f"scene seed {entry['seed']} digest is missing or changed")
    if scene.sha256 != entry["sha256"]:
        raise ValueError(f"scene seed {entry['seed']} differs from the frozen hash")
    return scene


def _trace_checks(directory: Path) -> dict:
    trace_path = directory / "trace.npz"
    if not trace_path.is_file():
        return {"available": False}
    with np.load(trace_path, allow_pickle=False) as trace:
        state = np.asarray(trace["state"])
        action = np.asarray(trace["action"])
        substeps = np.asarray(trace["substeps"]) if "substeps" in trace else None
    return {
        "available": True,
        "frames": int(action.shape[0]),
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "state_finite": bool(np.isfinite(state).all()),
        "action_finite": bool(np.isfinite(action).all()),
        "action_min": action.min(axis=0).tolist() if action.size else None,
        "action_max": action.max(axis=0).tolist() if action.size else None,
        "substeps_total": int(substeps.sum()) if substeps is not None else None,
    }


def _failure_category(row: dict) -> str:
    reason = str(row.get("failure_reason") or "")
    if row.get("success"):
        return "success"
    if "timeout" in reason:
        return "timeout"
    if "unexpected_contact" in reason:
        return "unexpected_contact"
    if "invalid_action" in reason or "policy_error" in reason or "action_error" in reason:
        return "policy_or_action_error"
    if row.get("stop_reason"):
        return str(row["stop_reason"])
    if row.get("invalid_infrastructure"):
        return "invalid_infrastructure"
    return reason.split(":")[0] or "unknown"


def _run_episode(runner, adapter, scene, directory: Path, timeout_s: float) -> dict:
    adapter.reset_seed = int(scene.seed)
    try:
        result = runner.run_episode(scene, output=directory, timeout_s=timeout_s, record=True)
    except Exception as error:  # infrastructure failure before or during the loop
        row = {
            "seed": int(scene.seed), "scene_sha256": scene.sha256, "success": False,
            "failure_reason": f"infrastructure: {type(error).__name__}: {error}",
            "exception": traceback.format_exc(),
            "frames": 0, "pre_action_failure": True, "invalid_infrastructure": True,
            "output": str(directory), "trace": _trace_checks(directory),
        }
        common.write_json(directory / "result.json", row)
        return row
    row = result.as_dict()
    row.update({
        "seed": int(scene.seed),
        "scene_sha256": scene.sha256,
        "output": str(directory),
        "trace": _trace_checks(directory),
        "reset_seed": int(scene.seed),
        "adapter_latency": adapter.latency_summary(),
    })
    pre_action = (int(row.get("frames") or 0) == 0
                  and str(row.get("failure_reason") or "").startswith(
                      ("policy_error", "action_error", "invalid_action")))
    row["pre_action_failure"] = bool(pre_action)
    if pre_action:
        row["invalid_infrastructure"] = True
    common.write_json(directory / "result.json", row)
    return row


def _run_with_retry(runner, adapter, scene, root: Path, *, timeout_s: float) -> dict:
    directory = root / f"direct_{scene.seed:04d}"
    first = _run_episode(runner, adapter, scene, directory, timeout_s)
    first["attempt"] = 1
    if not first.get("invalid_infrastructure"):
        return first
    retry_dir = root / f"direct_{scene.seed:04d}_retry"
    retry = _run_episode(runner, adapter, scene, retry_dir, timeout_s)
    retry["attempt"] = 2
    retry["first_attempt"] = first
    return retry


def _build_manifest(scenes: list[dict], *, certificate, config: dict, checkpoint_meta: dict,
                    e1_manifest_sha: str, protocol_sha: str) -> dict:
    entries = [{"seed": int(item["seed"]), "sha256": item["sha256"],
                "summary": item["summary"], "scene": item["scene"]} for item in scenes]
    manifest = {
        "schema_version": 1,
        "created": _now(),
        "phase": "E3",
        "purpose": "fixed 20-scene engineering diagnostic; not a benchmark",
        "seed_range": [min(item["seed"] for item in entries), max(item["seed"] for item in entries)],
        "episodes": entries,
        "certificate_hashes": certificate.hashes(),
        "config_sha256": common.sha256_json(config),
        "protocol_sha256": protocol_sha,
        "e1_manifest_sha256": e1_manifest_sha,
        "checkpoint": checkpoint_meta,
        "attempts_per_scene": 1,
        "invalid_rule": "pre-action infrastructure failure gets one retry; both attempts are kept",
    }
    manifest["manifest_sha256"] = common.sha256_json(
        {key: value for key, value in manifest.items() if key != "created"})
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="external SO101 checkout")
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/protocol.json")
    parser.add_argument("--eval-config", type=Path,
                        default=common.REPO_ROOT / "configs/so101/eval.json")
    parser.add_argument("--collection", type=Path, required=True,
                        help="E1 run directory containing eval_scenes.json and e1_manifest.json")
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8123)
    parser.add_argument("--checkpoint-name", default=None,
                        help="expected checkpoint directory name reported by the server")
    parser.add_argument("--expected-norm-stats-sha256", default=None)
    parser.add_argument("--mode", choices=("all", "smoke", "diagnostic"), default="all")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--public-manifest", type=Path, default=None)
    args = parser.parse_args()

    root_value = args.root or __import__("os").environ.get("SO101_SOURCE_ROOT")
    if not root_value:
        print("missing --root or SO101_SOURCE_ROOT", file=sys.stderr)
        return 2
    root = common.setup_external(root_value)
    protocol = common.load_json(args.protocol)
    eval_config = common.load_json(args.eval_config)
    config = common.load_external_config(root, protocol)
    collection = args.collection.expanduser().resolve()
    output = args.output.expanduser().resolve()
    train_view = args.train_view.expanduser().resolve()
    if output.exists():
        print(f"refusing to reuse existing evaluation directory: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True)

    e1_manifest = common.load_json(collection / "e1_manifest.json")
    e1_manifest_sha = common.sha256_json(e1_manifest)
    train_stats = common.load_json(train_view / "train_stats.json")
    expected_norm = (args.expected_norm_stats_sha256
                     or train_stats.get("norm_stats_canonical_sha256")
                     or train_stats["norm_stats_sha256"])

    from bimanual.recording.policy_runner import PolicyRunner
    from bimanual.sim.environment import BimanualEnv
    from bimanual.tasks.sorting import regions
    from bimanual.tasks.sorting.certification import check_certificate_current
    from bimanual.tasks.sorting.scene_sampling import generate_scene
    from so101_policy_client import SO101PolicyClient

    certificate = regions.load_certificate(root / config["evaluation"]["certificate"])
    client = SO101PolicyClient(args.server_host, args.server_port,
                               instruction=protocol["task"]["instruction"],
                               timeout_s=args.timeout_s)
    metadata = client.metadata
    print(json.dumps({"server_metadata": metadata}, indent=2, default=str), flush=True)
    if int(metadata.get("action_dim", -1)) != 12 or int(metadata.get("action_horizon", -1)) != 10:
        raise SystemExit("policy server does not expose the SO101 12-D / 10-step contract")
    if int(metadata.get("control_hz", -1)) != int(protocol["control"]["control_hz"]):
        raise SystemExit("policy server control rate differs from the SO101 protocol")
    if sorted(metadata.get("cameras", [])) != sorted(protocol["observation"]["cameras"]):
        raise SystemExit("policy server camera set differs from the SO101 protocol")
    if metadata.get("norm_stats_sha256") != expected_norm:
        raise SystemExit("policy server norm stats differ from the E1 train-only statistics")
    if args.checkpoint_name and metadata.get("checkpoint_name") != args.checkpoint_name:
        raise SystemExit(f"policy server checkpoint {metadata.get('checkpoint_name')} "
                         f"differs from requested {args.checkpoint_name}")
    warmup = client.warmup(timeout_s=600.0)
    print(json.dumps({"server_warmup": warmup}, indent=2), flush=True)

    checkpoint_meta = {
        "path": metadata.get("checkpoint"),
        "name": metadata.get("checkpoint_name"),
        "norm_stats_sha256": metadata.get("norm_stats_sha256"),
        "action_dim": metadata.get("action_dim"),
        "action_horizon": metadata.get("action_horizon"),
        "control_hz": metadata.get("control_hz"),
    }
    provenance = {
        "schema_version": 1,
        "created": _now(),
        "phase": "E3",
        "source_fingerprint": __import__("source_fingerprint").fingerprint(
            root, python=sys.executable),
        "protocol_sha256": common.sha256_file(args.protocol),
        "eval_config_sha256": common.sha256_file(args.eval_config),
        "config_sha256": common.sha256_json(config),
        "certificate_hashes": certificate.hashes(),
        "e1_manifest_sha256": e1_manifest_sha,
        "collection_dataset_info_sha256": e1_manifest["dataset_info_sha256"],
        "norm_stats_sha256": expected_norm,
        "checkpoint": checkpoint_meta,
        "server_metadata": metadata,
        "client_contract": {"state_shape": [12], "action_shape": [10, 12],
                            "cameras": protocol["observation"]["cameras"],
                            "control_hz": protocol["control"]["control_hz"],
                            "note": "PolicyRunner clips and applies every action"},
        "scripts_sha256": {
            "evaluate_so101.py": common.sha256_file(Path(__file__)),
            "so101_policy_client.py": common.sha256_file(
                Path(__file__).resolve().parent / "so101_policy_client.py"),
        },
        "claim": "engineering diagnostic; not a benchmark success rate",
    }

    results = []
    with BimanualEnv(config, images=True) as env:
        stale = check_certificate_current(certificate, env)
        if stale:
            raise SystemExit("certificate is stale: " + "; ".join(stale))
        runner = PolicyRunner(env, client)
        if args.mode in {"all", "smoke"}:
            smoke_seed = int(eval_config["smoke"]["seed"])
            smoke_scene = generate_scene(config, certificate, "direct", smoke_seed, env=env,
                                         sampling=str(config["sorting"].get("sampling_mode", "legacy")))
            smoke_manifest = {
                "seed": smoke_seed, "sha256": smoke_scene.sha256,
                "summary": smoke_scene.summary, "scene": smoke_scene.as_dict(),
                "certificate_hashes": certificate.hashes(),
                "config_sha256": common.sha256_json(config),
                "checkpoint": checkpoint_meta,
                "created": _now(),
            }
            common.write_json(output / "smoke_scene.json", smoke_manifest)
            smoke = _run_episode(runner, client, smoke_scene, output / "smoke",
                                 args.timeout_s)
            smoke["attempt"] = 1
            common.write_json(output / "smoke_result.json", smoke)
            print(json.dumps({"smoke": {key: smoke.get(key) for key in
                                        ("success", "failure_reason", "frames",
                                         "sim_time_s", "wall_time_s", "clipped_values",
                                         "safety_stops")}}), flush=True)
            if smoke.get("invalid_infrastructure") and smoke.get("pre_action_failure"):
                raise SystemExit("smoke closed loop failed before any action; stopping")
            if not smoke.get("trace", {}).get("available"):
                raise SystemExit("smoke produced no analyzable trace; stopping")
            results.append(smoke)

        if args.mode in {"all", "diagnostic"}:
            scenes = common.load_json(collection / "eval_scenes.json")["scenes"]
            episodes = list(scenes[:args.max_episodes] if args.max_episodes else scenes)
            if len(episodes) != 20:
                raise SystemExit(f"diagnostic needs the frozen 20 scenes, got {len(episodes)}")
            manifest = _build_manifest(episodes, certificate=certificate, config=config,
                                       checkpoint_meta=checkpoint_meta,
                                       e1_manifest_sha=e1_manifest_sha,
                                       protocol_sha=provenance["protocol_sha256"])
            common.write_json(output / "manifest.json", manifest)
            rows = []
            for entry in episodes:
                scene = _scene_from_entry(entry)
                row = _run_with_retry(runner, client, scene, output / "episodes",
                                      timeout_s=args.timeout_s)
                rows.append(row)
                print(json.dumps({"seed": scene.seed, "success": row.get("success"),
                                  "failure_reason": row.get("failure_reason"),
                                  "frames": row.get("frames"),
                                  "attempt": row.get("attempt")}), flush=True)
            valid = [row for row in rows if not row.get("invalid_infrastructure")]
            invalid = [row for row in rows if row.get("invalid_infrastructure")]
            successes = sum(1 for row in valid if row.get("success"))
            categories: dict[str, int] = {}
            for row in rows:
                category = _failure_category(row)
                categories[category] = categories.get(category, 0) + 1
            summary = {
                "schema_version": 1,
                "created": _now(),
                "phase": "E3",
                "mode": "diagnostic",
                "scene_count": len(rows),
                "valid_episodes": len(valid),
                "invalid_infrastructure": len(invalid),
                "successes": successes,
                "success_rate": (successes / len(valid)) if valid else None,
                "wilson_95": common.wilson_interval(successes, len(valid)),
                "denominator_note": (
                    "20 frozen scenes; invalid infrastructure episodes are excluded "
                    "only under the pre-registered rule and the denominator is restated"
                ),
                "failure_categories": categories,
                "attempts": rows,
                "manifest_sha256": manifest["manifest_sha256"],
                "checkpoint": checkpoint_meta,
                "claim": "small-sample engineering diagnostic; not a benchmark success rate",
            }
            common.write_json(output / "summary.json", summary)
            common.write_json(output / "provenance.json", provenance)
            public = {
                "schema_version": 1, "created": summary["created"], "phase": "E3",
                "scene_count": summary["scene_count"], "valid_episodes": summary["valid_episodes"],
                "invalid_infrastructure": summary["invalid_infrastructure"],
                "successes": summary["successes"], "success_rate": summary["success_rate"],
                "wilson_95": summary["wilson_95"],
                "failure_categories": summary["failure_categories"],
                "manifest_sha256": summary["manifest_sha256"],
                "scene_hashes": [{"seed": entry["seed"], "sha256": entry["sha256"]}
                                 for entry in episodes],
                "checkpoint_norm_stats_sha256": checkpoint_meta["norm_stats_sha256"],
                "claim": summary["claim"],
            }
            if args.public_manifest:
                common.write_json(args.public_manifest, public)
            print(json.dumps({key: summary[key] for key in
                              ("valid_episodes", "invalid_infrastructure", "successes",
                               "success_rate", "wilson_95", "failure_categories",
                               "manifest_sha256")}, indent=2))
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
