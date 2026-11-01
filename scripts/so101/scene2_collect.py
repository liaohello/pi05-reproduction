#!/usr/bin/env python3
"""Physical pilot: four frozen layouts x (red target, blue target) = 8 episodes.

Each episode is reset from its SceneSpec, the post-settle initial fingerprint is
recorded, and the episode is judged by the registered physical predicate.  The
three cameras are written as H.264 streams; the trace, events and applied action
per control frame are kept beside them.  No LeRobot dataset and no training are
started here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene2_common as s2  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class CameraSink:
    """Write one pre-action frame per control tick for each recorded camera."""

    def __init__(self, output: Path, env, camera_keys: list[str], fps: int):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.env = env
        self.camera_keys = list(camera_keys)
        self.fps = int(fps)
        self.index = 0
        self.errors = []
        self.fallback_stride = 5
        self.ffmpeg = shutil.which("ffmpeg")
        self.processes = {}
        self.frames = (self.output / "frames.jsonl").open("w", encoding="utf-8")
        if self.ffmpeg:
            for key in self.camera_keys:
                name = key.removeprefix("observation.images.")
                profile = env.config["camera"][name]
                width = int(profile.get("width", env.config["camera"]["central"]["width"]))
                height = int(profile.get("height", env.config["camera"]["central"]["height"]))
                path = self.output / f"{name}.mp4"
                command = [
                    self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                    "-r", str(self.fps), "-i", "-", "-an", "-c:v", "libx264",
                    "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-threads", "1", str(path),
                ]
                try:
                    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                               stderr=subprocess.PIPE)
                    self.processes[key] = {"process": process, "path": path,
                                           "name": name, "width": width, "height": height}
                except OSError as error:
                    self.errors.append({"camera": name, "error": f"{type(error).__name__}: {error}"})
        else:
            for key in self.camera_keys:
                name = key.removeprefix("observation.images.")
                (self.output / f"{name}_jpeg").mkdir(parents=True, exist_ok=True)

    def frame(self, observation, applied_action):
        row = {"index": self.index, "sim_time": float(self.env.data.time),
               "wall_time": time.time(),
               "applied_action": [float(value) for value in applied_action]}
        self.frames.write(json.dumps(row) + "\n")
        for key in self.camera_keys:
            image = np.asarray(observation[key], dtype=np.uint8)
            if self.ffmpeg:
                entry = self.processes.get(key)
                if entry is None:
                    continue
                height, width = entry["height"], entry["width"]
                if image.shape != (height, width, 3):
                    self.errors.append({"camera": entry["name"], "frame": self.index,
                                        "error": f"shape {image.shape} != {(height, width, 3)}"})
                    continue
                try:
                    entry["process"].stdin.write(image.tobytes())
                except (BrokenPipeError, OSError) as error:
                    self.errors.append({"camera": entry["name"], "frame": self.index,
                                        "error": f"{type(error).__name__}: {error}"})
            elif self.index % self.fallback_stride == 0:
                from PIL import Image

                name = key.removeprefix("observation.images.")
                Image.fromarray(image).save(
                    self.output / f"{name}_jpeg" / f"frame_{self.index:06d}.jpg", quality=92)
        self.index += 1

    def close(self) -> dict:
        videos = {}
        if self.ffmpeg:
            for key, entry in self.processes.items():
                process = entry["process"]
                try:
                    process.stdin.close()
                except OSError:
                    pass
                try:
                    returncode = process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = -1
                stderr = process.stderr.read().decode(errors="replace")[-800:]
                videos[entry["name"]] = {
                    "path": entry["path"].name, "returncode": returncode,
                    "bytes": entry["path"].stat().st_size if entry["path"].is_file() else None,
                    "stderr_tail": stderr if returncode else "",
                }
                if returncode != 0:
                    self.errors.append({"camera": entry["name"], "returncode": returncode,
                                        "error": stderr})
        self.frames.close()
        summary = {
            "frames": self.index, "fps": self.fps, "camera_keys": self.camera_keys,
            "backend": "ffmpeg" if self.ffmpeg else "jpeg",
            "fallback_stride": None if self.ffmpeg else self.fallback_stride,
            "videos": videos, "errors": self.errors,
        }
        (self.output / "sink.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary


def _load_scenes(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _action_divergence(first_dir: Path, second_dir: Path) -> dict:
    def rows(directory):
        path = directory / "cameras" / "frames.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    first, second = rows(first_dir), rows(second_dir)
    count = min(len(first), len(second))
    if count == 0:
        return {"common_frames": 0, "first_divergence_index": None,
                "max_abs_diff_degrees": None, "frames_over_1deg": None,
                "per_arm_max_abs_diff_degrees": None}
    left = np.asarray([row["applied_action"][:6] for row in first[:count]], dtype=float)
    right = np.asarray([row["applied_action"][:6] for row in second[:count]], dtype=float)
    left_other = np.asarray([row["applied_action"][6:] for row in first[:count]], dtype=float)
    right_other = np.asarray([row["applied_action"][6:] for row in second[:count]], dtype=float)
    left_diff = np.abs(left - right)
    right_diff = np.abs(left_other - right_other)
    combined = np.maximum(left_diff.max(axis=1), right_diff.max(axis=1))
    over = np.where(combined > 1.0)[0]
    return {
        "common_frames": int(count),
        "first_divergence_index": int(over[0]) if over.size else None,
        "max_abs_diff_degrees": float(combined.max()),
        "frames_over_1deg": int(over.size),
        "per_arm_max_abs_diff_degrees": {
            "left": float(left_diff.max()), "right": float(right_diff.max())},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", type=Path,
                        default=s2.REPO_ROOT / "configs/so101/scene2_dual_random_protocol.json")
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-clock-limit", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="engineering smoke only: stop after this many episodes")
    args = parser.parse_args()

    root = s2.configure(args.root)
    protocol = s2.load_protocol(args.protocol)
    s2.set_protocol_id(protocol["protocol_id"])
    certificate_path = (root / protocol["source_binding"]["certificate"]).resolve()
    config = s2.load_config(root, protocol, certificate=certificate_path)
    certificate, certificate_path = s2.load_certificate(config)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse existing run directory: {output}")
    output.mkdir(parents=True)
    provenance = s2.provenance(root, args.protocol, config, certificate_path)
    scenes_path = args.scenes.expanduser().resolve()
    frozen = _load_scenes(scenes_path)
    checks = {
        "protocol_id": frozen.get("protocol_id") == protocol["protocol_id"],
        "protocol_sha256": frozen.get("protocol_sha256") == s2.common.sha256_file(args.protocol),
        "adapter_sha256": frozen.get("adapter_sha256") == provenance["adapter_sha256"],
        "certificate_sha256": frozen.get("certificate_sha256") == provenance["certificate_sha256"],
        "source_sha256": (frozen.get("source_fingerprint") or {}).get("source_sha256")
        == provenance["source_sha256"],
        "complete": bool(frozen.get("complete")),
    }
    if not all(checks.values()):
        raise SystemExit("frozen scene registry does not match the current code/assets: "
                         + json.dumps(checks, sort_keys=True))
    layouts = frozen["layouts"]
    scenes = frozen["scenes"]
    settings = protocol["collection"]["physical_pilot"]
    wall_limit = float(args.wall_clock_limit or settings["episode_wall_clock_limit_s"])
    protocol_cap = int(settings["episode_attempt_cap"])
    episode_cap = int(args.limit) if args.limit is not None else protocol_cap

    BimanualEnv = s2._b("BimanualEnv")
    certification = s2._b("certification")
    EpisodeRunner = s2._b("EpisodeRunner")
    SceneSpec = s2._b("SceneSpec")
    camera_keys = list(config["recording"]["camera_keys"])
    if camera_keys != list(protocol["observation"]["cameras"]):
        raise SystemExit("camera order differs from the protocol")
    attempts_path = output / "attempts.jsonl"
    rows, pairs = [], []
    started = time.time()
    episode_index = 0
    stop = False
    with BimanualEnv(config, images=True) as env:
        stale = certification.check_certificate_current(certificate, env)
        if stale:
            raise SystemExit("certificate stale: " + "; ".join(stale))
        runtime = s2.install_runtime(config, env)
        runner = EpisodeRunner(env, runtime["runner_config"], certificate)
        for layout_index, layout in enumerate(layouts):
            if stop:
                break
            physical = layout["physical_sha256"]
            condition_rows = {}
            for color in ("red", "blue"):
                if episode_index >= episode_cap:
                    stop = True
                    break
                if time.time() - started > wall_limit:
                    rows.append({"attempt": episode_index + 1, "phase": "physical_pilot",
                                 "layout_index": layout_index,
                                 "physical_sha256": physical,
                                 "target_color": color,
                                 "success": False,
                                 "failure_reason": "wall_clock_exceeded",
                                 "failure_stage": None, "frames": 0,
                                 "wall_time_s": time.time() - started})
                    episode_index += 1
                    stop = True
                    break
                scene = SceneSpec.from_dict(scenes[physical][color])
                scene_dir = output / f"layout_{layout_index:02d}_{layout['arrangement']}" / color
                scene_dir.mkdir(parents=True, exist_ok=True)
                sink = CameraSink(scene_dir / "cameras", env, camera_keys,
                                  int(config["recording"]["fps"]))
                episode_started = time.time()
                result = None
                failure = None
                try:
                    result = runner.run_episode(scene, output=scene_dir, record=True, sink=sink)
                except Exception as error:  # keep the pilot ledger even on an adapter failure
                    failure = f"{type(error).__name__}: {error}"
                finally:
                    try:
                        sink_summary = sink.close()
                    except Exception as error:
                        sink_summary = {"frames": sink.index, "errors": [
                            {"error": f"{type(error).__name__}: {error}"}]}
                if result is None:
                    row = {
                        "attempt": episode_index + 1, "phase": "physical_pilot",
                        "layout_index": layout_index, "layout_seed": layout["seed"],
                        "arrangement": layout["arrangement"], "target_color": color,
                        "target_name": scene.extra["target_name"],
                        "distractor_name": scene.extra["distractor_name"],
                        "active_arm": scene.extra["active_arm"],
                        "physical_sha256": physical,
                        "layout_sha256": layout["layout_sha256"],
                        "condition_sha256": scene.extra["condition_sha256"],
                        "scene_sha256": scene.sha256, "success": False,
                        "raw_success": False, "failure_reason": "adapter_exception",
                        "failure_stage": None, "failure_message": failure,
                        "frames": sink_summary.get("frames", 0),
                        "sim_time_s": 0.0, "wall_time_s": time.time() - episode_started,
                        "initial_fingerprint": None, "initial_metrics_passed": None,
                        "initial_metrics": None, "target_status": None,
                        "distractor_status": None, "distractor_check": None,
                        "arm_clearance": None, "cameras": sink_summary,
                        "diagnostics": str(scene_dir.relative_to(output)),
                        "scene": scene.as_dict(),
                    }
                    rows.append(row)
                    condition_rows[color] = row
                    with attempts_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, default=str) + "\n")
                    print(json.dumps({"attempt": row["attempt"], "layout": layout_index,
                                      "color": color, "success": False,
                                      "reason": "adapter_exception", "message": failure}),
                          flush=True)
                    episode_index += 1
                    continue
                status = result.detail.get("status") or {}
                fingerprint = env.task.initial_fingerprint
                initial_metrics = env.task.initial_metrics
                row = {
                    "attempt": episode_index + 1,
                    "phase": "physical_pilot",
                    "layout_index": layout_index,
                    "layout_seed": layout["seed"],
                    "arrangement": layout["arrangement"],
                    "target_color": color,
                    "target_name": scene.extra["target_name"],
                    "distractor_name": scene.extra["distractor_name"],
                    "active_arm": scene.extra["active_arm"],
                    "physical_sha256": physical,
                    "layout_sha256": layout["layout_sha256"],
                    "condition_sha256": scene.extra["condition_sha256"],
                    "scene_sha256": scene.sha256,
                    "success": bool(result.success),
                    "raw_success": bool(status.get("success")),
                    "failure_reason": result.failure_reason or status.get("failure_reason"),
                    "failure_stage": result.detail.get("failure_stage"),
                    "failure_message": result.detail.get("failure_message"),
                    "frames": int(result.frames),
                    "sim_time_s": float(result.sim_time_s),
                    "wall_time_s": float(result.wall_time_s),
                    "initial_fingerprint": fingerprint["fingerprint_sha256"] if fingerprint else None,
                    "initial_metrics_passed": (initial_metrics or {}).get("passed"),
                    "initial_metrics": initial_metrics,
                    "target_status": (status.get("objects") or {}).get(scene.extra["target_name"]),
                    "distractor_status": (status.get("objects") or {}).get(scene.extra["distractor_name"]),
                    "distractor_check": status.get("distractor"),
                    "arm_clearance": status.get("arm_clearance"),
                    "cameras": sink_summary,
                    "diagnostics": str(scene_dir.relative_to(output)),
                    "scene": scene.as_dict(),
                }
                rows.append(row)
                condition_rows[color] = row
                with attempts_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, default=str) + "\n")
                print(json.dumps({"attempt": row["attempt"], "layout": layout_index,
                                  "color": color, "arm": row["active_arm"],
                                  "success": row["success"],
                                  "reason": row["failure_reason"],
                                  "frames": row["frames"],
                                  "wall_s": round(row["wall_time_s"], 2)}), flush=True)
                episode_index += 1
            if len(condition_rows) == 2:
                first, second = condition_rows["red"], condition_rows["blue"]
                same = bool(first["initial_fingerprint"] and second["initial_fingerprint"]
                            and first["initial_fingerprint"] == second["initial_fingerprint"])
                first_dir = output / first["diagnostics"]
                second_dir = output / second["diagnostics"]
                divergence = _action_divergence(first_dir, second_dir)
                pairs.append({
                    "layout_index": layout_index,
                    "physical_sha256": physical,
                    "arrangement": layout["arrangement"],
                    "initial_fingerprints": {"red": first["initial_fingerprint"],
                                             "blue": second["initial_fingerprint"]},
                    "initial_fingerprint_match": same,
                    "action_divergence": divergence,
                    "red_success": first["success"], "blue_success": second["success"],
                })
    layouts_success = sum(1 for pair in pairs
                          if pair["red_success"] and pair["blue_success"])
    manifest = {
        "schema_version": 1,
        "created": _now(),
        "phase": "physical_pilot",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": s2.common.sha256_file(args.protocol),
        "scenes_file": scenes_path.name,
        "scenes_sha256": s2.common.sha256_file(scenes_path),
        "profile_sha256": provenance["profile_sha256"],
        "adapter_sha256": provenance["adapter_sha256"],
        "certificate_sha256": provenance["certificate_sha256"],
        "source_fingerprint": {key: provenance[key] for key in
                               ("source_sha256", "source_files", "expert_sha256",
                                "model_sha256", "config_sha256", "packages")},
        "layouts": len(layouts),
        "episodes_executed": len(rows),
        "successes": sum(1 for row in rows if row["success"]),
        "failures": sum(1 for row in rows if not row["success"]),
        "layout_successes_both_conditions": layouts_success,
        "conditions": {color: sum(1 for row in rows if row["target_color"] == color)
                       for color in ("red", "blue")},
        "arm_balance": {
            color: {arm: sum(1 for row in rows if row["target_color"] == color
                             and row["active_arm"] == arm)
                    for arm in ("left", "right")}
            for color in ("red", "blue")},
        "arrangement_balance": {value: sum(1 for row in rows if row["arrangement"] == value)
                                for value in ("red_positive_y", "red_negative_y")},
        "distinct_initial_states": len({row["initial_fingerprint"] for row in rows
                                        if row["initial_fingerprint"]}),
        "pair_checks": pairs,
        "pair_initial_fingerprint_matches": sum(1 for pair in pairs
                                                if pair["initial_fingerprint_match"]),
        "episode_cap": episode_cap,
        "protocol_episode_cap": protocol_cap,
        "engineering_smoke_limit": args.limit,
        "wall_clock_limit_s": wall_limit,
        "wall_time_s": time.time() - started,
        "pilot_is_not_validation": True,
        "claim": ("4 layouts x 2 instructions = 8 condition episodes; "
                  "not 8 independent scenes; pilot data are excluded from training "
                  "and from the formal validation split"),
        "attempts_file": attempts_path.name,
    }
    (output / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if len(rows) == protocol_cap else 1


if __name__ == "__main__":
    sys.exit(main())
