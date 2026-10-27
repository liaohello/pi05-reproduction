#!/usr/bin/env python3
"""E3.1 / G4: one training-scene closed loop for a paired checkpoint.

Uses the same PolicyRunner and SubstepGuard as E3, the fixed first training
scene and its recorded initial state.  This is a training-scene learnability
diagnostic, not a generalisation success rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import so101_common as common  # noqa: E402

JOINT_INDICES = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
GRIPPER_INDICES = [5, 11]


def _mae(values, indices):
    return float(np.abs(np.asarray(values)[:, indices]).mean()) if len(values) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/protocol.json")
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--expected-checkpoint-name", default="999")
    parser.add_argument("--expected-state-mode", choices=("off", "on"), required=True)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--public-output", type=Path, default=None)
    args = parser.parse_args()
    root_value = args.root or __import__("os").environ.get("SO101_SOURCE_ROOT")
    if not root_value:
        raise SystemExit("missing --root or SO101_SOURCE_ROOT")
    root = common.setup_external(root_value)
    protocol = common.load_json(args.protocol)
    config = common.load_external_config(root, protocol)
    collection = args.collection.expanduser().resolve()
    run_config = common.load_json(args.run_config)
    binding = common.load_json(args.binding)
    e1 = common.load_json(collection / "e1_manifest.json")
    episode = e1["episodes"][0]
    diagnostics = Path(episode.get("diagnostics") or episode["attempt"]["diagnostics"])
    scene_dict = common.load_json(diagnostics / "scene.json")
    initial_state = np.load(diagnostics / "initial_state.npz")["state"]
    with np.load(diagnostics / "trace.npz", allow_pickle=False) as trace:
        expert_action = np.asarray(trace["action"], dtype=np.float32)
        expert_state = np.asarray(trace["state"], dtype=np.float32)
        expert_substeps = np.asarray(trace["substeps"]) if "substeps" in trace else None

    from bimanual.recording.policy_runner import PolicyRunner
    from bimanual.sim.environment import BimanualEnv
    from bimanual.tasks.sorting import regions
    from bimanual.tasks.sorting.certification import check_certificate_current
    from bimanual.tasks.sorting.scene_sampling import SceneSpec
    from so101_policy_client import SO101PolicyClient

    scene = SceneSpec.from_dict(scene_dict)
    if not scene.sha256 or scene.digest() != scene.sha256 or scene.sha256 != episode["scene_sha256"]:
        raise SystemExit("scene digest is not comparable")
    if int(scene.seed) != int(episode["seed"]):
        raise SystemExit("scene seed differs from the fixed training episode")
    certificate = regions.load_certificate(root / config["evaluation"]["certificate"])
    client = SO101PolicyClient(args.server_host, args.server_port,
                               instruction=protocol["task"]["instruction"],
                               timeout_s=args.timeout_s)
    metadata = client.metadata
    expected_norm = run_config.get("norm_stats_canonical_sha256")
    expected_config = run_config.get("config_sha256")
    checks = {
        "state_mode": metadata.get("state_mode") == args.expected_state_mode
        and args.expected_state_mode == run_config.get("state_mode"),
        "checkpoint_name": metadata.get("checkpoint_name") == args.expected_checkpoint_name,
        "norm_stats": metadata.get("norm_stats_sha256") == expected_norm,
        "state_step": binding.get("train_state_step") is not None,
        "run_config_sha": metadata.get("run_config_sha256") == common.sha256_file(args.run_config),
        "config_sha": metadata.get("expected_config_sha256") == expected_config,
        "action_contract": (int(metadata.get("action_dim", -1)) == 12
                            and int(metadata.get("action_horizon", -1)) == 10
                            and int(metadata.get("control_hz", -1)) == 50),
        "cameras": sorted(metadata.get("cameras", [])) == sorted(protocol["observation"]["cameras"]),
    }
    if not all(checks.values()):
        raise SystemExit("server/config binding failed: " + json.dumps(checks))
    warmup = client.warmup(timeout_s=600.0)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse {output}")
    output.mkdir(parents=True)
    client.reset_seed = int(scene.seed)
    with BimanualEnv(config, images=True) as env:
        stale = check_certificate_current(certificate, env)
        if stale:
            raise SystemExit("certificate stale: " + "; ".join(stale))
        runner = PolicyRunner(env, client)
        result = runner.run_episode(scene, output=output / "episode", timeout_s=args.timeout_s,
                                    record=True, initial_state=initial_state)
        result_dict = result.as_dict()
        with np.load(output / "episode" / "trace.npz", allow_pickle=False) as trace:
            state = np.asarray(trace["state"], dtype=np.float32)
            action = np.asarray(trace["action"], dtype=np.float32)
            substeps = np.asarray(trace["substeps"]) if "substeps" in trace else None
            failed = np.asarray(trace["failed_frame"], dtype=bool) if "failed_frame" in trace else None
    n = min(len(expert_action), len(action))
    action_error = np.abs(action[:n] - expert_action[:n]) if n else np.zeros((0, 12))
    tracking = (np.abs(state[1:n] - action[:n - 1]) if n > 1 else np.zeros((0, 12)))
    comparison = {
        "expert_frames": len(expert_action),
        "policy_frames": len(action),
        "compared_frames": n,
        "action_vs_expert_mae_joint_deg": _mae(action_error, JOINT_INDICES),
        "action_vs_expert_mae_gripper_pct": _mae(action_error, GRIPPER_INDICES),
        "action_vs_expert_mae_per_dim": (action_error.mean(axis=0).tolist()
                                         if len(action_error) else None),
        "command_tracking_mae_joint_deg": _mae(tracking, JOINT_INDICES),
        "command_tracking_mae_gripper_pct": _mae(tracking, GRIPPER_INDICES),
        "expert_substeps_unique": sorted(set(np.asarray(expert_substeps).tolist()))
        if expert_substeps is not None else None,
        "policy_substeps_unique": sorted(set(np.asarray(substeps).tolist()))
        if substeps is not None else None,
        "failed_frames": int(failed.sum()) if failed is not None else None,
        "trace_available": True,
    }
    report = {
        "schema_version": 1,
        "phase": "E3.1",
        "substage": "G4-paired-rollout",
        "label": args.expected_state_mode,
        "scene_sha256": scene.sha256,
        "seed": int(scene.seed),
        "initial_state_matches_file": True,
        "run_config": str(args.run_config),
        "run_config_sha256": common.sha256_file(args.run_config),
        "binding": binding,
        "train_state_step": binding.get("train_state_step"),
        "parameters_sha256": binding.get("parameters_sha256"),
        "server_metadata": metadata,
        "server_checks": checks,
        "warmup": warmup,
        "result": {key: result_dict.get(key) for key in
                   ("success", "failure_reason", "sorted_count", "frames", "sim_time_s",
                    "wall_time_s", "clipped_values", "clipped_frames", "safety_stops",
                    "stop_reason", "stop_frame", "stop_substeps", "stop_time_s",
                    "partial_frame", "executed_substeps")},
        "stop_contacts": result_dict.get("stop_contacts") or [],
        "comparison": comparison,
        "claim": "training-scene learnability diagnostic; not a generalisation success rate",
    }
    common.write_json(output / "rollout.json", report)
    if args.public_output:
        common.write_json(args.public_output, {
            "schema_version": 1, "phase": "E3.1", "substage": "G4-paired-rollout",
            "label": report["label"], "scene_sha256": report["scene_sha256"],
            "seed": report["seed"], "checkpoint_name": metadata.get("checkpoint_name"),
            "state_mode": metadata.get("state_mode"),
            "train_state_step": report["train_state_step"],
            "parameters_sha256": report["parameters_sha256"],
            "norm_stats_sha256": metadata.get("norm_stats_sha256"),
            "result": report["result"], "stop_contacts": report["stop_contacts"],
            "comparison": report["comparison"], "server_checks": checks,
            "claim": report["claim"],
        })
    print(json.dumps({"label": report["label"], "success": report["result"]["success"],
                      "sorted_count": report["result"]["sorted_count"],
                      "frames": report["result"]["frames"],
                      "stop_reason": report["result"]["stop_reason"],
                      "action_mae_joint": comparison["action_vs_expert_mae_joint_deg"],
                      "action_mae_gripper": comparison["action_vs_expert_mae_gripper_pct"],
                      "tracking_mae_joint": comparison["command_tracking_mae_joint_deg"],
                      "tracking_mae_gripper": comparison["command_tracking_mae_gripper_pct"]}, indent=2))
    client.close()
    return 0 if report["result"]["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
