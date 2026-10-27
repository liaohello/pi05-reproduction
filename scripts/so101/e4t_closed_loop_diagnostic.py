#!/usr/bin/env python3
"""One gate-waived, single-scene MuJoCo diagnostic for the E4-T state-on checkpoint.

This does not implement the E4-R evaluation protocol. It admits one explicitly bound training
scene, checks the actual first policy observation against that scene's recorded frame zero before
allowing inference, then runs one episode through the existing PolicyRunner and safety guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import so101_common as common  # noqa: E402

CAMERAS = {
    "observation.images.central": "central",
    "observation.images.left_wrist": "left_wrist",
    "observation.images.right_wrist": "right_wrist",
}
CAMERA_ORDER = list(CAMERAS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def low_quality_triptych(frames: dict[str, np.ndarray], destination: Path) -> None:
    tiles = []
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    for key in CAMERA_ORDER:
        image = Image.fromarray(np.asarray(frames[key], dtype=np.uint8), mode="RGB")
        tiles.append(image.resize((320, 240), resample))
    canvas = Image.new("RGB", (960, 240))
    for index, tile in enumerate(tiles):
        canvas.paste(tile, (index * 320, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=38, optimize=True)


class LowQualityVideoRecorder:
    """Write one compact, time-ordered camera triptych per policy request."""

    def __init__(self, destination: Path, *, fps: float = 5.0):
        import cv2

        destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination = destination
        self.fps = float(fps)
        self.width = 480
        self.height = 120
        self.frame_count = 0
        self._cv2 = cv2
        self._writer = cv2.VideoWriter(
            str(destination), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
            (self.width, self.height))
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError("could not initialize low-quality MP4 recording")

    def write(self, frames: dict[str, np.ndarray]) -> None:
        tiles = [self._cv2.resize(np.asarray(frames[key]), (160, 120),
                                  interpolation=self._cv2.INTER_AREA)
                 for key in CAMERA_ORDER]
        triptych = np.concatenate(tiles, axis=1)
        self._writer.write(self._cv2.cvtColor(triptych, self._cv2.COLOR_RGB2BGR))
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def evidence(self) -> dict:
        return {"file": self.destination.name, "format": "mp4v",
                "width": self.width, "height": self.height,
                "fps": self.fps, "frame_count": self.frame_count,
                "control_frames_per_recorded_frame": 10,
                "sha256": sha256_file(self.destination)
                if self.destination.is_file() else None}


def compare_first_frame(observation: dict, expected_state: np.ndarray,
                        expected_images: dict[str, np.ndarray], *, state_tolerance: float,
                        image_mae_tolerance: float, image_p95_tolerance: float) -> dict:
    state = np.asarray(observation.get("observation.state"), dtype=np.float32)
    state_error = np.abs(state - expected_state) if state.shape == expected_state.shape else None
    cameras = {}
    for key in CAMERA_ORDER:
        actual = np.asarray(observation.get(key))
        expected = expected_images[key]
        if actual.shape == expected.shape and actual.dtype == np.uint8:
            error = np.abs(actual.astype(np.int16) - expected.astype(np.int16)).reshape(-1)
            cameras[key] = {"shape": list(actual.shape), "dtype": str(actual.dtype),
                            "mae_pixel": float(error.mean()),
                            "p95_abs_pixel": float(np.percentile(error, 95)),
                            "request_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                            "shape_and_dtype_valid": True}
        else:
            cameras[key] = {"shape": list(actual.shape), "dtype": str(actual.dtype),
                            "mae_pixel": None, "p95_abs_pixel": None,
                            "request_sha256": None, "shape_and_dtype_valid": False}
    state_max = float(state_error.max()) if state_error is not None else None
    return {
        "state_shape": list(state.shape), "state_max_abs_error": state_max,
        "state_mae": float(state_error.mean()) if state_error is not None else None,
        "state_passed": state_error is not None and state_max <= state_tolerance,
        "camera_order": CAMERA_ORDER,
        "cameras": cameras,
        "cameras_passed": all(
            row["shape_and_dtype_valid"]
            and row["mae_pixel"] <= image_mae_tolerance
            and row["p95_abs_pixel"] <= image_p95_tolerance
            for row in cameras.values()),
        "tolerances": {"state_max_abs": state_tolerance,
                       "camera_mae_pixel": image_mae_tolerance,
                       "camera_p95_abs_pixel": image_p95_tolerance},
    }


def gpu_free_memory_mib(gpu_index: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        raise RuntimeError("could not read GPU free memory")
    return int(result.stdout.strip().splitlines()[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/so101/protocol.json")
    parser.add_argument("--episode-dir", type=Path, required=True,
                        help="one frozen E4-T train-view episode directory")
    parser.add_argument("--diagnostics-dir", type=Path, required=True,
                        help="matching external collection diagnostics directory")
    parser.add_argument("--run-config", type=Path, required=True,
                        help="E4-T saved run_config.json")
    parser.add_argument("--binding", type=Path, required=True,
                        help="authoritative checkpoint and view binding JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--state-tolerance", type=float, default=1e-3)
    parser.add_argument("--image-mae-tolerance", type=float, default=12.0)
    parser.add_argument("--image-p95-tolerance", type=float, default=55.0)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--min-free-memory-mib", type=int, default=1500)
    args = parser.parse_args()

    root_value = args.root or os.environ.get("SO101_SOURCE_ROOT")
    if not root_value:
        raise SystemExit("missing --root or SO101_SOURCE_ROOT")
    root = common.setup_external(root_value)
    protocol = common.load_json(args.protocol)
    config = common.load_external_config(root, protocol)
    episode_dir = args.episode_dir.expanduser().resolve()
    diagnostics = args.diagnostics_dir.expanduser().resolve()
    run_config_path = args.run_config.expanduser().resolve()
    binding = common.load_json(args.binding)
    run_config = common.load_json(run_config_path)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit("refusing to reuse diagnostic output directory")
    output.mkdir(parents=True)

    episode_meta = common.load_json(episode_dir / "meta.json")
    view_meta_path = episode_dir.parent.parent / "meta.json"
    view_meta = common.load_json(view_meta_path)
    episode_index = int(episode_meta["episode_index"])
    view_row = next((row for row in view_meta["episodes"]
                     if int(row["episode_index"]) == episode_index), None)
    if view_row is None:
        raise SystemExit("episode is absent from the frozen view manifest")
    if episode_meta.get("split") != "train" or episode_meta.get("category") != "clean":
        raise SystemExit("diagnostic scene must be a frozen clean training episode")
    if view_row.get("split") != "train" or view_row.get("category") != "clean":
        raise SystemExit("frozen view manifest does not bind this to a clean training episode")
    if view_row.get("scene_sha256") != episode_meta.get("scene_sha256"):
        raise SystemExit("episode scene digest differs from the frozen view manifest")
    if view_row.get("attempt_id") != episode_meta.get("attempt_id"):
        raise SystemExit("episode attempt id differs from the frozen view manifest")

    scene_dict = common.load_json(diagnostics / "scene.json")
    initial_state = np.load(diagnostics / "initial_state.npz", allow_pickle=False)["state"]
    with np.load(diagnostics / "trace.npz", allow_pickle=False) as trace:
        expert_state = np.asarray(trace["state"], dtype=np.float32)
        expert_action = np.asarray(trace["action"], dtype=np.float32)
    with np.load(episode_dir / "episode.npz", allow_pickle=False) as episode:
        view_state = np.asarray(episode["state"], dtype=np.float32)
        view_action = np.asarray(episode["action"], dtype=np.float32)
        view_scene_sha = str(episode["scene_sha256"])
    if view_scene_sha != episode_meta.get("scene_sha256"):
        raise SystemExit("episode NPZ scene digest differs from its metadata")
    if expert_state.shape != view_state.shape or expert_action.shape != view_action.shape:
        raise SystemExit("recorded trace and frozen view episode shapes differ")
    trace_state_max = float(np.max(np.abs(expert_state - view_state)))
    trace_action_max = float(np.max(np.abs(expert_action - view_action)))
    if trace_state_max > 1e-6 or trace_action_max > 1e-6:
        raise SystemExit("recorded trace and frozen view episode are not identical")
    if not len(initial_state) or not np.isfinite(initial_state).all():
        raise SystemExit("recorded MuJoCo initial state is empty or non-finite")

    from bimanual.recording.policy_runner import PolicyRunner
    from bimanual.sim.environment import BimanualEnv
    from bimanual.tasks.sorting import regions
    from bimanual.tasks.sorting.certification import check_certificate_current
    from bimanual.tasks.sorting.scene_sampling import SceneSpec
    from so101_policy_client import SO101PolicyClient

    scene = SceneSpec.from_dict(scene_dict)
    scene_hash_ok = bool(scene.sha256 and scene.digest() == scene.sha256
                         and scene.sha256 == episode_meta["scene_sha256"])
    if not scene_hash_ok or int(scene.seed) != int(episode_meta["seed"]):
        raise SystemExit("SceneSpec digest or seed differs from the audited episode")

    run_config_sha = common.sha256_file(run_config_path)
    checks = {
        "authoritative_train_state_step": int(binding.get("train_state_step", -1)) == 30000,
        "orbax_directory": int(binding.get("orbax_step_directory", -1)) == 29999,
        "parameter_hash_present": bool(binding.get("parameters_sha256")),
        "state_mode": run_config.get("state_mode") == "on",
        "run_config_hash": binding.get("run_config_sha256") == run_config_sha,
        "config_hash": binding.get("config_sha256") == run_config.get("config_sha256"),
        "norm_stats_hash": binding.get("norm_stats_canonical_sha256")
        == run_config.get("norm_stats_canonical_sha256"),
        "view_hash": binding.get("view_sha256") == view_meta.get("view_sha256")
        == run_config.get("view_sha256"),
        "step_and_parameter_record": int(binding.get("train_state_step", -1)) == 30000
        and binding.get("parameters_sha256") == "cf636c6eee7729902b6eda63a23d96894e84e6068ad8dbf9b2377361a9e2004d",
    }
    if not all(checks.values()):
        raise SystemExit("checkpoint/run/view binding failed: " + json.dumps(checks))

    client = SO101PolicyClient(args.server_host, args.server_port,
                               instruction=protocol["task"]["instruction"],
                               timeout_s=args.timeout_s)
    metadata = client.metadata
    checks.update({
        "server_state_mode": metadata.get("state_mode") == "on",
        "server_checkpoint_name": metadata.get("checkpoint_name") == "29999",
        "server_norm_stats": metadata.get("norm_stats_sha256")
        == run_config.get("norm_stats_canonical_sha256"),
        "server_expected_norm_stats": metadata.get("expected_norm_stats_canonical_sha256")
        == run_config.get("norm_stats_canonical_sha256"),
        "server_run_config": metadata.get("run_config_sha256") == run_config_sha,
        "server_config": metadata.get("expected_config_sha256") == run_config.get("config_sha256"),
        "server_state_key": metadata.get("state_key") == "observation.state",
        "server_action_contract": (int(metadata.get("action_dim", -1)) == 12
                                    and int(metadata.get("action_horizon", -1)) == 10
                                    and int(metadata.get("control_hz", -1)) == 50),
        "server_cameras": sorted(metadata.get("cameras", [])) == CAMERA_ORDER,
    })
    if not all(checks.values()):
        client.close()
        raise SystemExit("policy-server binding failed: " + json.dumps(checks))

    warmup = client.warmup(timeout_s=600.0)
    free_after_warmup = gpu_free_memory_mib(args.gpu_index)
    if free_after_warmup < args.min_free_memory_mib:
        client.close()
        raise SystemExit("GPU free memory fell below the registered runtime margin after warmup")
    expected_images = {
        key: np.asarray(Image.open(episode_dir / camera / "000000.jpg").convert("RGB"))
        for key, camera in CAMERAS.items()
    }
    captured = {"done": False, "audit": None, "frame_sha256": None,
                "observations_seen": 0}
    video = None
    original_act = client.act

    def checked_act(observation: dict) -> np.ndarray:
        frames = {key: np.asarray(observation[key]) for key in CAMERA_ORDER}
        if video is None:
            raise RuntimeError("low-quality video recorder was not initialized")
        if captured["observations_seen"] % client.action_steps == 0:
            video.write(frames)
        captured["observations_seen"] += 1
        if not captured["done"]:
            audit = compare_first_frame(
                observation, view_state[0], expected_images,
                state_tolerance=args.state_tolerance,
                image_mae_tolerance=args.image_mae_tolerance,
                image_p95_tolerance=args.image_p95_tolerance)
            low_quality_triptych(frames, output / "frame0_lowq.jpg")
            captured["frame_sha256"] = sha256_file(output / "frame0_lowq.jpg")
            captured["audit"] = audit
            captured["done"] = True
            if not audit["state_passed"] or not audit["cameras_passed"]:
                raise RuntimeError("actual frame-zero observation failed the pre-action comparison")
            if gpu_free_memory_mib(args.gpu_index) < args.min_free_memory_mib:
                raise RuntimeError("GPU free memory fell below the registered runtime margin before action")
        return original_act(observation)

    client.act = checked_act
    client.reset_seed = int(scene.seed)
    certificate = regions.load_certificate(root / config["evaluation"]["certificate"])
    result_dict = None
    run_error = None
    try:
        video = LowQualityVideoRecorder(output / "inference_lowq.mp4")
        with BimanualEnv(config, images=True) as env:
            stale = check_certificate_current(certificate, env)
            if stale:
                raise RuntimeError("certificate stale")
            runner = PolicyRunner(env, client)
            result = runner.run_episode(scene, output=output / "episode",
                                        timeout_s=args.timeout_s, record=True,
                                        initial_state=initial_state)
            result_dict = result.as_dict()
    except Exception as error:  # preserve preflight and policy failures as diagnostic evidence
        run_error = type(error).__name__
    finally:
        if video is not None:
            video.close()
        client.close()

    result_fields = ("success", "failure_reason", "sorted_count", "frames", "sim_time_s",
                     "wall_time_s", "clipped_values", "clipped_frames", "safety_stops",
                     "stop_reason", "stop_frame", "stop_substeps", "stop_time_s",
                     "partial_frame", "executed_substeps")
    report = {
        "schema_version": 1,
        "phase": "E4-T",
        "substage": "single_scene_gate_waived_closed_loop_diagnostic",
        "claim": "one authorized diagnostic MuJoCo rollout; not E4-R, not a success-rate estimate, and not an open-loop gate pass",
        "user_authorized_gate_waiver": True,
        "closed_loop_actions_executed": bool(
            result_dict and int(result_dict.get("executed_substeps") or 0) > 0),
        "policy_inference_chunks": int(client.requests),
        "real_follower_started": False,
        "checkpoint": {"train_state_step": 30000, "orbax_step_directory": 29999,
                       "parameters_sha256": binding["parameters_sha256"],
                       "run_config_sha256": run_config_sha,
                       "norm_stats_canonical_sha256": run_config["norm_stats_canonical_sha256"],
                       "view_sha256": view_meta["view_sha256"]},
        "scene": {"episode_index": episode_index, "split": episode_meta["split"],
                  "category": episode_meta["category"], "seed": int(scene.seed),
                  "scene_sha256": scene.sha256,
                  "attempt_id": episode_meta.get("attempt_id"),
                  "trace_vs_view_state_max_abs": trace_state_max,
                  "trace_vs_view_action_max_abs": trace_action_max},
        "server_checks": checks,
        "warmup": warmup,
        "gpu_free_memory_after_warmup_mib": free_after_warmup,
        "frame_zero_pre_action_audit": captured["audit"],
        "frame_zero_lowq_sha256": captured["frame_sha256"],
        "video": video.evidence() if video is not None else None,
        "result": ({key: result_dict.get(key) for key in result_fields}
                   if result_dict else None),
        "exception_type": run_error,
        "stop_contacts": (result_dict.get("stop_contacts") or []) if result_dict else [],
    }
    common.write_json(output / "rollout.json", report)
    print(json.dumps({"closed_loop_actions_executed": report["closed_loop_actions_executed"],
                      "frame_zero_audit_passed": bool(captured["audit"]
                          and captured["audit"]["state_passed"]
                          and captured["audit"]["cameras_passed"]),
                      "success": (result_dict or {}).get("success"),
                      "sorted_count": (result_dict or {}).get("sorted_count"),
                      "frames": (result_dict or {}).get("frames"),
                      "stop_reason": (result_dict or {}).get("stop_reason"),
                      "exception_type": run_error,
                      "frame0_lowq_sha256": captured["frame_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
