#!/usr/bin/env python3
"""E3.1 / G1: replay the fixed first expert demonstration through PolicyRunner.

The adapter ignores observations and emits the recorded applied actions.  The
run uses the same physics, control period and SubstepGuard as E3.  Initial
state/action mismatches or extra clipping are reported as incomparable; the
guard is never relaxed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import so101_common as common  # noqa: E402


class OracleSequenceAdapter:
    """Return recorded applied actions; no observation is used."""

    def __init__(self, actions: np.ndarray):
        self.actions = np.asarray(actions, dtype=np.float32)
        self.index = 0
        self.last_telemetry = None

    def reset(self) -> None:
        self.index = 0
        self.last_telemetry = None

    def act(self, observation) -> np.ndarray:
        del observation
        if self.index < len(self.actions):
            action = self.actions[self.index]
            self.index += 1
            self.last_telemetry = {"oracle_index": self.index, "source": "recorded_trace"}
            return action
        self.last_telemetry = {"oracle_index": self.index, "source": "last_action_hold"}
        return self.actions[-1]


def _max_abs(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--protocol", type=Path,
                        default=common.REPO_ROOT / "configs/so101/protocol.json")
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, default=None)
    parser.add_argument("--analyze-existing", action="store_true",
                        help="analyze an existing episode/ run without stepping the simulator again")
    args = parser.parse_args()
    root_value = args.root or __import__("os").environ.get("SO101_SOURCE_ROOT")
    if not root_value:
        raise SystemExit("missing --root or SO101_SOURCE_ROOT")
    root = common.setup_external(root_value)
    protocol = common.load_json(args.protocol)
    config = common.load_external_config(root, protocol)
    collection = args.collection.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.analyze_existing:
        if not (output / "episode" / "summary.json").is_file():
            raise SystemExit(f"no existing episode summary under {output}")
    elif output.exists():
        raise SystemExit(f"refusing to reuse {output}")
    e1 = common.load_json(collection / "e1_manifest.json")
    episodes = e1["episodes"]
    if not 0 <= args.episode_index < len(episodes):
        raise SystemExit("episode index out of range")
    episode = episodes[args.episode_index]
    diagnostics = Path(episode.get("diagnostics") or episode["attempt"]["diagnostics"])
    trace_path = diagnostics / "trace.npz"
    scene_path = diagnostics / "scene.json"
    initial_path = diagnostics / "initial_state.npz"
    for path in (trace_path, scene_path, initial_path):
        if not path.is_file():
            raise SystemExit(f"missing {path}")
    with np.load(trace_path, allow_pickle=False) as trace:
        expert_action = np.asarray(trace["action"], dtype=np.float32)
        expert_state = np.asarray(trace["state"], dtype=np.float32)
        expert_time = np.asarray(trace["time"], dtype=np.float64)
        expert_substeps = np.asarray(trace["substeps"]) if "substeps" in trace else None
        trace_initial = np.asarray(trace["initial_state"], dtype=np.float64)
    initial_state = np.asarray(np.load(initial_path)["state"], dtype=np.float64)
    initial_match = bool(np.array_equal(initial_state, trace_initial))
    scene_dict = common.load_json(scene_path)
    from bimanual.tasks.sorting.scene_sampling import SceneSpec
    scene = SceneSpec.from_dict(scene_dict)
    scene_digest_ok = bool(scene.sha256 and scene.digest() == scene.sha256
                           and scene.sha256 == episode["scene_sha256"])
    if int(scene.seed) != int(episode["seed"]):
        raise SystemExit("scene seed differs from the fixed episode")
    if not initial_match or not scene_digest_ok:
        raise SystemExit("initial state or scene digest is not comparable")

    from bimanual.recording.policy_runner import PolicyRunner
    from bimanual.sim.environment import BimanualEnv
    from bimanual.tasks.sorting import regions
    from bimanual.tasks.sorting.certification import check_certificate_current

    certificate = regions.load_certificate(root / config["evaluation"]["certificate"])
    if args.analyze_existing:
        result_dict = common.load_json(output / "episode" / "summary.json")
    else:
        output.mkdir(parents=True)
        adapter = OracleSequenceAdapter(expert_action)
        with BimanualEnv(config, images=True) as env:
            stale = check_certificate_current(certificate, env)
            if stale:
                raise SystemExit("certificate stale: " + "; ".join(stale))
            runner = PolicyRunner(env, adapter)
            result = runner.run_episode(scene, output=output / "episode", timeout_s=120.0,
                                        record=True, initial_state=initial_state)
            result_dict = result.as_dict()
    with np.load(output / "episode" / "trace.npz", allow_pickle=False) as trace:
        run_state = np.asarray(trace["state"], dtype=np.float32)
        run_action = np.asarray(trace["action"], dtype=np.float32)
        run_time = np.asarray(trace["time"], dtype=np.float64)
        run_substeps = np.asarray(trace["substeps"]) if "substeps" in trace else None
    with np.load(output / "episode" / "initial_state.npz", allow_pickle=False) as saved:
        run_initial = np.asarray(saved["state"], dtype=np.float64)
    n = min(len(expert_action), len(run_action))
    action_error = _max_abs(expert_action[:n], run_action[:n]) if n else float("inf")
    state_error0 = _max_abs(expert_state[0], run_state[0]) if len(run_state) else float("inf")
    period = 1.0 / float(config["runtime"]["control_hz"])
    runner_time0_minus_period = (float(run_time[0]) - period) if len(run_time) else float("inf")
    time_alignment_error = (abs(float(expert_time[0]) - runner_time0_minus_period)
                            if len(run_time) else float("inf"))
    initial_state_error = _max_abs(initial_state, run_initial)
    comparable = bool(
        initial_match and scene_digest_ok
        and int(result_dict.get("clipped_values", 1)) == 0
        and action_error <= 1e-6
        and state_error0 <= 1e-6
        and time_alignment_error <= 1e-6
        and initial_state_error <= 1e-6
    )
    oracle_ok = bool(comparable and result_dict.get("success")
                     and int(result_dict.get("sorted_count", 0)) == int(episode["attempt"]["sorted_count"])
                     and int(result_dict.get("safety_stops", 1)) == 0)
    report = {
        "schema_version": 1,
        "phase": "E3.1",
        "substage": "G1-oracle",
        "episode_index": args.episode_index,
        "seed": int(episode["seed"]),
        "scene_sha256": scene.sha256,
        "source_sha256": e1["source_fingerprint"]["source_sha256"],
        "config_sha256": common.sha256_json({key: value for key, value in config.items()
                                             if key != "root"}),
        "certificate_hashes": certificate.hashes(),
        "control": {"physics_hz": config["runtime"]["physics_hz"],
                    "control_hz": config["runtime"]["control_hz"],
                    "frames": len(expert_action),
                    "physics_substeps_per_full_frame": int(config["runtime"]["physics_hz"]
                                                           // config["runtime"]["control_hz"])},
        "initial_state": {
            "file_matches_trace": initial_match,
            "restored_vs_recorded_max_abs": initial_state_error,
            "frame0_state_error": state_error0,
            "runner_frame0_time_s": float(run_time[0]) if len(run_time) else None,
            "expert_frame0_time_s": float(expert_time[0]),
            "runner_time0_minus_period_s": runner_time0_minus_period,
            "time_alignment_error_s": time_alignment_error,
            "time_semantics": "expert trace stores pre-action time; PolicyRecorder stores post-step time; compare with one control-period offset",
        },
        "actions": {
            "oracle_frames": len(expert_action),
            "runner_frames": len(run_action),
            "compared_frames": n,
            "max_abs_error": action_error,
            "unconsumed_expert_frames": len(expert_action) - n,
            "runner_terminated_by_success": bool(result_dict.get("success")),
            "extra_clipping": int(result_dict.get("clipped_values", 0)),
            "clipped_frames": int(result_dict.get("clipped_frames", 0)),
        },
        "time": {
            "expert_first_last_s": [float(expert_time[0]), float(expert_time[-1])],
            "runner_first_last_s": ([float(run_time[0]), float(run_time[-1])]
                                    if len(run_time) else None),
            "control_period_s": period,
            "substeps_unique_expert": sorted(set(np.asarray(expert_substeps).tolist()))
            if expert_substeps is not None else None,
            "substeps_unique_runner": sorted(set(np.asarray(run_substeps).tolist()))
            if run_substeps is not None else None,
        },
        "result": {key: result_dict.get(key) for key in
                   ("success", "failure_reason", "sorted_count", "frames", "sim_time_s",
                    "wall_time_s", "safety_stops", "clipped_values", "stop_frame",
                    "stop_substeps", "stop_time_s", "partial_frame", "stop_reason")},
        "stop_contacts": result_dict.get("stop_contacts") or [],
        "comparable": comparable,
        "oracle_passed": oracle_ok,
        "claim": "oracle action sequence through the same PolicyRunner/SubstepGuard; "
                 "not a learned-policy result",
    }
    common.write_json(output / "oracle_report.json", report)
    if args.public_output:
        common.write_json(args.public_output, {
            "schema_version": 1, "phase": "E3.1", "substage": "G1-oracle",
            "seed": report["seed"], "scene_sha256": report["scene_sha256"],
            "initial_state_file_matches_trace": report["initial_state"]["file_matches_trace"],
            "restored_vs_recorded_max_abs": report["initial_state"]["restored_vs_recorded_max_abs"],
            "frame0_state_error": report["initial_state"]["frame0_state_error"],
            "action_max_abs_error": report["actions"]["max_abs_error"],
            "extra_clipping": report["actions"]["extra_clipping"],
            "result": report["result"],
            "time_alignment_error_s": report["initial_state"]["time_alignment_error_s"],
            "stop_contacts": report["stop_contacts"],
            "comparable": report["comparable"], "oracle_passed": report["oracle_passed"],
            "claim": report["claim"],
        })
    print(json.dumps(report, indent=2, default=str)[:6000])
    return 0 if oracle_ok else 1


if __name__ == "__main__":
    sys.exit(main())
