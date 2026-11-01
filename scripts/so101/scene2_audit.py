#!/usr/bin/env python3
"""Audit the pilot traces and produce red/blue visual evidence.

The audit is read-only with respect to the run directory.  It checks control
ticks, state/action alignment, the applied target, all three camera streams,
object trajectories and the recorded physical predicate, then writes
``frame_audit.json`` and one contact sheet per layout pair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene2_common as s2  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _hash_array(array) -> str:
    return hashlib.sha256(np.asarray(array, dtype=np.float64).tobytes()).hexdigest()


def _ffprobe(path: Path) -> dict:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
               "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames,duration",
               "-of", "json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    if result.returncode != 0:
        return {"error": result.stderr[-500:]}
    return json.loads(result.stdout).get("streams", [{}])[0]


def _frame_hashes(path: Path) -> list[str]:
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "framemd5", "-"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return []
    hashes = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 6 and fields[0].isdigit():
            hashes.append(fields[-1])
    return hashes


def _duplicate_runs(hashes: list[str]) -> list[dict]:
    runs = []
    if not hashes:
        return runs
    start, previous = 0, hashes[0]
    for index in range(1, len(hashes) + 1):
        current = hashes[index] if index < len(hashes) else None
        if current != previous:
            length = index - start
            if length > 1:
                runs.append({"start": start, "length": length, "end": index - 1})
            start, previous = index, current
    return runs


def _stage_at(stages, index: int) -> str:
    if index < 0 or index >= len(stages):
        return ""
    return str(stages[index])


def _episode_audit(row: dict, run: Path, protocol: dict) -> dict:
    directory = run / row["diagnostics"]
    trace = np.load(directory / "trace.npz", allow_pickle=False)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    scene = json.loads((directory / "scene.json").read_text(encoding="utf-8"))
    sink = json.loads((directory / "cameras" / "sink.json").read_text(encoding="utf-8"))
    sink_rows = [json.loads(line) for line in
                 (directory / "cameras" / "frames.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    times = np.asarray(trace["time"], dtype=np.float64)
    stages = [str(value) for value in trace["stage"]]
    state = np.asarray(trace["state"], dtype=np.float64)
    action = np.asarray(trace["action"], dtype=np.float64)
    objects = np.asarray(trace["objects_pos"], dtype=np.float64)
    substeps = np.asarray(trace["substeps"], dtype=np.int64)
    expected_dt = 1.0 / float(protocol["control"]["control_hz"])
    deltas = np.diff(times) if times.size > 1 else np.asarray([])
    applied = np.asarray([entry["applied_action"] for entry in sink_rows], dtype=np.float64) \
        if sink_rows else np.zeros((0, 12))
    sink_times = np.asarray([entry["sim_time"] for entry in sink_rows], dtype=np.float64) \
        if sink_rows else np.zeros(0)
    wall_times = np.asarray([entry["wall_time"] for entry in sink_rows], dtype=np.float64) \
        if sink_rows else np.zeros(0)
    common = min(len(action), len(applied))
    applied_error = (np.abs(action[:common] - applied[:common])
                     if common else np.zeros((0, 12)))
    joint_dims = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    gripper_dims = [5, 11]
    action_diff = np.abs(np.diff(action, axis=0)) if len(action) > 1 else np.zeros((0, 12))
    jumps = {
        "joint_max_deg": float(action_diff[:, joint_dims].max()) if len(action_diff) else 0.0,
        "gripper_max_percent": float(action_diff[:, gripper_dims].max()) if len(action_diff) else 0.0,
        "frames_joint_over_15deg": int(np.sum(action_diff[:, joint_dims].max(axis=1) > 15.0))
        if len(action_diff) else 0,
        "frames_gripper_over_30percent": int(np.sum(action_diff[:, gripper_dims].max(axis=1) > 30.0))
        if len(action_diff) else 0,
        "first_joint_jump_index": (int(np.argmax(action_diff[:, joint_dims].max(axis=1) > 15.0))
                                   if len(action_diff) and
                                   np.any(action_diff[:, joint_dims].max(axis=1) > 15.0) else None),
        "note": "expert stage boundaries can legitimately change the gripper target; joints are expected to be continuous",
    }
    tolerance = 1e-6
    expected_substeps = int(round(
        float(protocol["control"]["physics_hz"]) / float(protocol["control"]["control_hz"])))
    incomplete = np.asarray(trace["incomplete"], dtype=bool)
    short = np.where(deltas < expected_dt - tolerance)[0] if deltas.size else []
    long = np.where(deltas > expected_dt + tolerance)[0] if deltas.size else []
    partial_frames = []
    for index in short:
        if substeps[index] < expected_substeps or incomplete[index]:
            partial_frames.append({"index": int(index), "dt_s": float(deltas[index]),
                                   "substeps": int(substeps[index]),
                                   "incomplete": bool(incomplete[index])})
    explained_short = len(partial_frames)
    unexplained_short = int(len(short) - explained_short)
    time_audit = {
        "frames": int(times.size),
        "expected_dt_s": expected_dt,
        "expected_substeps": expected_substeps,
        "dt_min_s": float(deltas.min()) if deltas.size else None,
        "dt_max_s": float(deltas.max()) if deltas.size else None,
        "dt_mean_s": float(deltas.mean()) if deltas.size else None,
        "dt_missing": int(len(long)),
        "dt_duplicate": int(len(short)),
        "dt_short": int(len(short)),
        "dt_long": int(len(long)),
        "dt_nonmonotonic": int(np.sum(deltas <= 0.0)) if deltas.size else 0,
        "partial_frames": partial_frames,
        "explained_short": explained_short,
        "unexplained_short": unexplained_short,
        "substeps": dict(Counter(int(value) for value in substeps.tolist())),
        "incomplete_frames": int(np.sum(incomplete)),
    }
    physics_dt = 1.0 / float(protocol["control"]["physics_hz"])
    if substeps.size and (substeps[-1] < expected_substeps or incomplete[-1]):
        expected_terminal = float(times[-1] + int(substeps[-1]) * physics_dt)
        summary_time = summary.get("sim_time_s")
        time_audit["terminal_partial_frame"] = {
            "index": int(times.size - 1),
            "substeps": int(substeps[-1]),
            "incomplete": bool(incomplete[-1]),
            "trace_time_s": float(times[-1]),
            "expected_terminal_time_s": expected_terminal,
            "summary_sim_time_s": float(summary_time) if summary_time is not None else None,
            "gap_s": (float(summary_time) - expected_terminal) if summary_time is not None else None,
        }
    else:
        time_audit["terminal_partial_frame"] = None
    camera_audit = {}
    for key in sink.get("camera_keys", []):
        name = key.removeprefix("observation.images.")
        path = directory / "cameras" / f"{name}.mp4"
        probe = _ffprobe(path)
        hashes = _frame_hashes(path)
        runs = _duplicate_runs(hashes)
        longest = max(runs, key=lambda item: item["length"]) if runs else None
        camera_audit[name] = {
            "path": path.name, "probe": probe, "decoded_frames": len(hashes),
            "duplicate_runs": len(runs), "longest_duplicate_run": longest,
            "longest_run_stage": _stage_at(stages, longest["start"]) if longest else None,
            "longest_run_time_s": (float(times[min(longest["start"], times.size - 1)])
                                   if longest and times.size else None),
            "frame_count_matches_trace": len(hashes) == int(times.size),
        }
    object_audit = {}
    roles = {scene["target_name"]: "target",
             scene["distractor_name"]: "distractor"}
    for index, entry in enumerate(scene.get("objects", [])):
        name = entry["name"]
        positions = objects[:, index, :]
        path_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))) \
            if len(positions) > 1 else 0.0
        object_audit[name] = {
            "role": roles.get(name),
            "color": entry.get("color"),
            "path_length_m": path_length,
            "net_displacement_m": float(np.linalg.norm(positions[-1] - positions[0]))
            if len(positions) else None,
            "initial_xy": positions[0, :2].tolist() if len(positions) else None,
            "final_xy": positions[-1, :2].tolist() if len(positions) else None,
            "moved_over_5mm": bool(len(positions)
                                   and np.linalg.norm(positions[-1] - positions[0]) > 0.005),
        }
    initial_state = np.load(directory / "initial_state.npz", allow_pickle=False)["state"]
    status = summary.get("status") or {}
    target_status = (status.get("objects") or {}).get(scene["target_name"]) or {}
    distractor_status = (status.get("objects") or {}).get(scene["distractor_name"]) or {}
    predicate = {
        "summary_success": bool(summary.get("success")),
        "status_success": bool(status.get("success")),
        "failure_reason": summary.get("failure_reason"),
        "failure_stage": summary.get("failure_stage"),
        "sorted_count": status.get("sorted_count"),
        "target_contained": target_status.get("contained"),
        "target_supported": target_status.get("supported"),
        "target_robot_contacts": target_status.get("robot_contacts"),
        "target_stable_in_tray": target_status.get("stable_in_tray"),
        "distractor_stable_on_table": distractor_status.get("stable_on_table"),
        "distractor_robot_contacts": distractor_status.get("robot_contacts"),
        "distractor_grasped": distractor_status.get("grasped"),
        "distractor_displacement_m": distractor_status.get("displacement_from_initial_m"),
        "distractor_displacement_ok": (status.get("distractor") or {}).get("displacement_ok"),
        "contact_monitor_error_count": summary.get("contact_monitor_error_count"),
        "soft_contact_count": summary.get("soft_contact_count"),
    }
    consistency = {
        "success_matches_status": predicate["summary_success"] == predicate["status_success"],
        "success_sorted_one": (predicate["sorted_count"] == 1) if predicate["summary_success"] else True,
        "success_target_contained": (predicate["target_contained"] is True)
        if predicate["summary_success"] else True,
        "success_target_supported": (predicate["target_supported"] is True)
        if predicate["summary_success"] else True,
        "success_no_target_robot_contact": (not predicate["target_robot_contacts"])
        if predicate["summary_success"] else True,
        "success_no_distractor_contact": (not predicate["distractor_robot_contacts"])
        if predicate["summary_success"] else True,
        "success_no_distractor_grasp": (not predicate["distractor_grasped"])
        if predicate["summary_success"] else True,
        "success_distractor_displacement_ok": (predicate["distractor_displacement_ok"] is True)
        if predicate["summary_success"] else True,
    }
    return {
        "attempt": row["attempt"], "layout_index": row["layout_index"],
        "arrangement": row["arrangement"], "target_color": row["target_color"],
        "active_arm": row["active_arm"],
        "physical_sha256": row["physical_sha256"], "condition_sha256": row["condition_sha256"],
        "success": row["success"], "failure_reason": row["failure_reason"],
        "initial_fingerprint": row["initial_fingerprint"],
        "initial_state_sha256": _hash_array(initial_state),
        "time": time_audit,
        "state_action": {
            "state_shape": list(state.shape), "action_shape": list(action.shape),
            "state_finite": bool(np.isfinite(state).all()),
            "action_finite": bool(np.isfinite(action).all()),
            "action_jumps": jumps,
            "applied_vs_commanded": {
                "compared_frames": int(common),
                "max_abs_diff": float(applied_error.max()) if applied_error.size else None,
                "frames_over_1e_5": int(np.sum(applied_error.max(axis=1) > 1e-5))
                if applied_error.size else 0,
                "note": "applied is the clamped canonical target; commanded is the plan target",
            },
        },
        "sink": {
            "frames": len(sink_rows), "sink_reported_frames": sink.get("frames"),
            "backend": sink.get("backend"), "errors": sink.get("errors", []),
            "sim_time_max_abs_diff_s": float(np.max(np.abs(sink_times - times[:len(sink_times)])))
            if sink_rows and times.size else None,
            "wall_dt_min_s": float(np.diff(wall_times).min()) if len(wall_times) > 1 else None,
            "wall_dt_max_s": float(np.diff(wall_times).max()) if len(wall_times) > 1 else None,
            "wall_rate_hz": (float(len(wall_times) / (wall_times[-1] - wall_times[0]))
                             if len(wall_times) > 1 and wall_times[-1] > wall_times[0] else None),
        },
        "cameras": camera_audit,
        "objects": object_audit,
        "predicate": predicate,
        "consistency": consistency,
        "consistency_passed": all(consistency.values()),
    }


def _contact_sheet(first: Path, second: Path, output: Path, label: str) -> dict:
    from PIL import Image, ImageDraw

    frames = {}
    for role, path in (("red", first), ("blue", second)):
        for position, extra in (("initial", ["-ss", "0"]),
                                ("final", ["-sseof", "-0.05"])):
            target = output.parent / f".{label}_{role}_{position}.png"
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                       *extra, "-i", str(path), "-frames:v", "1", str(target)]
            subprocess.run(command, check=False, capture_output=True, timeout=120)
            if target.is_file():
                with Image.open(target) as image:
                    frames[(role, position)] = image.convert("RGB").copy()
    if len(frames) < 4:
        return {"created": False, "reason": "could not extract four frames"}
    width = max(image.width for image in frames.values())
    height = max(image.height for image in frames.values())
    sheet = Image.new("RGB", (width * 2, height * 2 + 40), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), label, fill=(240, 240, 240))
    positions = {("red", "initial"): (0, 40), ("red", "final"): (width, 40),
                 ("blue", "initial"): (0, height + 40), ("blue", "final"): (width, height + 40)}
    for key, image in frames.items():
        x, y = positions[key]
        sheet.paste(image, (x, y))
        draw.text((x + 6, y + 6), f"{key[0]} {key[1]}", fill=(255, 230, 80))
    sheet.save(output)
    for key in frames:
        (output.parent / f".{label}_{key[0]}_{key[1]}.png").unlink(missing_ok=True)
    return {"created": True, "path": str(output), "size": list(sheet.size)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
                        default=s2.REPO_ROOT / "configs/so101/scene2_dual_random_protocol.json")
    parser.add_argument("--assets", type=Path,
                        default=s2.REPO_ROOT / "reports/assets/so101-scene2-dual-random")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run = args.run.expanduser().resolve()
    protocol = s2.load_protocol(args.protocol)
    manifest = json.loads((run / "pilot_manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in
            (run / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("diagnostics")]
    assets = args.assets.expanduser().resolve()
    assets.mkdir(parents=True, exist_ok=True)
    audits, sheets = [], []
    for row in rows:
        try:
            audits.append(_episode_audit(row, run, protocol))
        except Exception as error:
            audits.append({"attempt": row.get("attempt"), "layout_index": row.get("layout_index"),
                           "target_color": row.get("target_color"),
                           "error": f"{type(error).__name__}: {error}"})
    by_layout = {}
    for row in rows:
        by_layout.setdefault(row["layout_index"], {})[row["target_color"]] = row
    for layout_index, conditions in sorted(by_layout.items()):
        if set(conditions) == {"red", "blue"}:
            red = run / conditions["red"]["diagnostics"] / "cameras" / "central.mp4"
            blue = run / conditions["blue"]["diagnostics"] / "cameras" / "central.mp4"
            if red.is_file() and blue.is_file():
                output = assets / f"layout_{layout_index:02d}_red_vs_blue.png"
                sheets.append({"layout_index": layout_index,
                               **_contact_sheet(red, blue, output,
                                                f"layout {layout_index} {conditions['red']['arrangement']}")})
    initial_pairs = {}
    for layout_index, conditions in by_layout.items():
        if set(conditions) == {"red", "blue"}:
            red_state = next((audit["initial_state_sha256"] for audit in audits
                              if audit.get("layout_index") == layout_index
                              and audit.get("target_color") == "red"), None)
            blue_state = next((audit["initial_state_sha256"] for audit in audits
                               if audit.get("layout_index") == layout_index
                               and audit.get("target_color") == "blue"), None)
            initial_pairs[layout_index] = {
                "red": red_state, "blue": blue_state,
                "match": bool(red_state and blue_state and red_state == blue_state),
            }
    summary = {
        "schema_version": 1,
        "created": _now(),
        "phase": "pilot_frame_audit",
        "protocol_id": protocol["protocol_id"],
        "run": run.name,
        "pilot_manifest": manifest,
        "episodes": len(audits),
        "episodes_with_errors": sum(1 for audit in audits if "error" in audit),
        "all_time_checks": all(audit.get("time", {}).get("unexplained_short") == 0
                               and audit.get("time", {}).get("dt_long") == 0
                               and audit.get("time", {}).get("dt_nonmonotonic") == 0
                               for audit in audits if "error" not in audit),
        "all_state_action_finite": all(audit.get("state_action", {}).get("state_finite")
                                       and audit.get("state_action", {}).get("action_finite")
                                       for audit in audits if "error" not in audit),
        "all_consistency": all(audit.get("consistency_passed") for audit in audits
                               if "error" not in audit),
        "all_video_frame_counts_match": all(
            all(camera.get("frame_count_matches_trace")
                for camera in audit.get("cameras", {}).values())
            for audit in audits if "error" not in audit),
        "camera_order": rows[0].get("cameras", {}).get("camera_keys") if rows else None,
        "initial_pair_states": initial_pairs,
        "initial_pair_matches": sum(1 for value in initial_pairs.values() if value["match"]),
        "manifest_pair_initial_fingerprint_matches": manifest.get("pair_initial_fingerprint_matches"),
        "initial_pair_checks_agree": (sum(1 for value in initial_pairs.values() if value["match"])
                                      == manifest.get("pair_initial_fingerprint_matches")),
        "terminal_partial_frames": [
            {"attempt": audit.get("attempt"), "layout_index": audit.get("layout_index"),
             "target_color": audit.get("target_color"),
             **audit["time"]["terminal_partial_frame"]}
            for audit in audits if "error" not in audit
            and audit.get("time", {}).get("terminal_partial_frame")],
        "dt_failures": [{"attempt": audit.get("attempt"),
                         "layout_index": audit.get("layout_index"),
                         "target_color": audit.get("target_color"),
                         **audit.get("time", {})}
                        for audit in audits if "error" not in audit
                        and (audit["time"]["unexplained_short"] or audit["time"]["dt_long"]
                             or audit["time"]["dt_nonmonotonic"])],
        "action_jump_failures": [{"attempt": audit.get("attempt"),
                                  "layout_index": audit.get("layout_index"),
                                  "target_color": audit.get("target_color"),
                                  **audit["state_action"]["action_jumps"]}
                                 for audit in audits if "error" not in audit
                                 and (audit["state_action"]["action_jumps"]["frames_joint_over_15deg"]
                                      or audit["state_action"]["action_jumps"]["frames_gripper_over_30percent"])],
        "video_duplicate_runs": [{"attempt": audit.get("attempt"),
                                  "camera": name,
                                  **camera}
                                 for audit in audits if "error" not in audit
                                 for name, camera in audit.get("cameras", {}).items()
                                 if camera.get("longest_duplicate_run")],
        "contact_sheets": sheets,
        "episodes_detail": audits,
        "passed": bool(
            audits and all("error" not in audit for audit in audits)
            and all(audit.get("time", {}).get("unexplained_short") == 0
                    and audit.get("time", {}).get("dt_long") == 0
                    and audit.get("time", {}).get("dt_nonmonotonic") == 0
                    for audit in audits)
            and all(audit.get("state_action", {}).get("state_finite")
                    and audit.get("state_action", {}).get("action_finite")
                    for audit in audits)
            and all(audit.get("consistency_passed") for audit in audits)
            and all(all(camera.get("frame_count_matches_trace")
                        for camera in audit.get("cameras", {}).values())
                    for audit in audits)
            and (sum(1 for value in initial_pairs.values() if value["match"])
                 == manifest.get("pair_initial_fingerprint_matches"))),
    }
    output = args.output or (run / "frame_audit.json")
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    public = {key: value for key, value in summary.items() if key != "episodes_detail"}
    public["episodes_detail"] = [{key: value for key, value in audit.items()
                                  if key not in ("cameras",)} for audit in audits]
    (run / "frame_audit_summary.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("episodes_detail", "pilot_manifest")}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
