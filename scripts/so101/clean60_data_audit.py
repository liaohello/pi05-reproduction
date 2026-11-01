#!/usr/bin/env python3
"""Audit every clean60 attempt and every frame of the exported training view."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import clean60_contracts as clean60  # noqa: E402
import clean100_checks  # noqa: E402
import so101_common as common  # noqa: E402


def resolve_diagnostics(collection: Path, row: dict) -> Path:
    recorded = Path(str(row.get("diagnostics") or ""))
    if recorded.is_dir():
        return recorded
    candidate = collection / "diagnostics" / recorded.name
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"missing diagnostics for seed {row.get('seed')}")


def image_problems(episode_dir: Path, frames: int) -> list[str]:
    """Check exact file coverage and decode every RGB frame once."""
    from PIL import Image

    problems = []
    expected = [f"{index:06d}.jpg" for index in range(frames)]

    def check(path: Path) -> str | None:
        try:
            with Image.open(path) as image:
                if image.size != (640, 480):
                    return f"{path.parent.name}: size {image.size}"
                image.verify()
            with Image.open(path) as image:
                if image.mode != "RGB":
                    return f"{path.parent.name}: mode {image.mode}"
        except Exception as error:
            return f"{path.parent.name}: decode failed: {type(error).__name__}"
        return None

    paths = []
    for camera in ("central", "left_wrist", "right_wrist"):
        directory = episode_dir / camera
        actual = sorted(path.name for path in directory.iterdir() if path.is_file())
        if actual != expected:
            problems.append(f"{camera}: file coverage differs from every recorded frame")
        paths.extend(directory / name for name in expected if (directory / name).is_file())
    with ThreadPoolExecutor(max_workers=32) as pool:
        problems.extend(result for result in pool.map(check, paths) if result)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, action="append", required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, default=None,
                        help="optional pinned SO101 source for re-applying every action")
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--expected-train", type=int, default=None)
    parser.add_argument("--expected-validation", type=int, default=None)
    parser.add_argument("--expected-attempts", type=int, default=None)
    parser.add_argument("--expected-successes", type=int, default=None)
    parser.add_argument("--expected-failures", type=int, default=None)
    args = parser.parse_args()

    collections = [path.expanduser().resolve() for path in args.collection]
    collection_by_name = {collection.name: collection for collection in collections}
    if len(collection_by_name) != len(collections):
        raise SystemExit("duplicate collection names")
    view = args.view.expanduser().resolve()
    protocol = common.load_json(args.protocol)
    meta = common.load_json(view / "meta.json")
    binding = common.load_json(view / "protocol_binding.json")
    split = common.load_json(view / "splits.json")
    admission = common.load_json(view / "e4t_admission.json")
    rows_meta = meta["episodes"]
    total = len(rows_meta)
    train_total = sum(row.get("split") == "train" for row in rows_meta)
    validation_total = sum(row.get("split") == "validation" for row in rows_meta)
    expected_counts = (args.expected_episodes if args.expected_episodes is not None
                       else (60 if total == 60 else total),
                       args.expected_train if args.expected_train is not None
                       else (48 if total == 60 else train_total),
                       args.expected_validation if args.expected_validation is not None
                       else (12 if total == 60 else validation_total))
    if (total, train_total, validation_total) != expected_counts:
        raise SystemExit(f"view is {total}/{train_total}/{validation_total}, "
                         f"expected {expected_counts}")
    single = collections[0] if len(collections) == 1 else None
    collection_of_row: dict[int, Path] = {}
    e1_by_collection: dict[str, dict] = {}
    verification_by_collection: dict[str, dict] = {}
    attempts = []
    rows_by_collection: dict[str, list[dict]] = {name: [] for name in collection_by_name}
    for collection in collections:
        e1 = common.load_json(collection / "e1_manifest.json")
        e1_by_collection[collection.name] = e1
        verification_by_collection[collection.name] = common.load_json(
            collection / "verification.json")
        for row in (json.loads(line) for line in
                    (collection / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()):
            row = dict(row)
            row["source_collection"] = collection.name
            attempts.append(row)
    for row in rows_meta:
        name = row.get("source_collection")
        if name is None:
            if single is None:
                raise SystemExit("merged rows must name their source collection")
            name = single.name
        if name not in collection_by_name:
            raise SystemExit(f"view row names an unknown collection: {name}")
        rows_by_collection[name].append(row)
        collection_of_row[int(row["episode_index"])] = collection_by_name[name]
    for name, rows in rows_by_collection.items():
        if len(rows) != len(e1_by_collection[name]["episodes"]):
            raise SystemExit(f"{name}: manifest and view episode counts differ")
    problems = clean60.view_contract_problems(
        meta, protocol_binding=binding, expected_counts=expected_counts, require_clean=True)
    problems.extend(clean60.admission_contract_problems(
        admission, meta=meta, split=split,
        train_stats_sha256=clean60.sha256_file(view / "train_stats.json")))
    tree_sha = common.sha256_tree(view / "episodes")["sha256"]
    if tree_sha != meta.get("view_sha256"):
        problems.append("recomputed episode tree hash differs from metadata")

    success_attempts = [row for row in attempts if row.get("success")]
    failed_attempts = [row for row in attempts if not row.get("success")]
    expected_attempts = (args.expected_attempts if args.expected_attempts is not None
                         else (175 if total == 60 else len(attempts)))
    expected_successes = (args.expected_successes if args.expected_successes is not None
                          else (60 if total == 60 else len(success_attempts)))
    expected_failures = (args.expected_failures if args.expected_failures is not None
                         else (115 if total == 60 else len(failed_attempts)))
    if (len(attempts), len(success_attempts), len(failed_attempts)) != (
            expected_attempts, expected_successes, expected_failures):
        problems.append(f"attempt ledger is not the required "
                        f"{expected_attempts}/{expected_successes}/{expected_failures} accounting")
    if any(not row.get("failure_reason") or not row.get("failure_stage")
           for row in failed_attempts):
        problems.append("one or more failed attempts lack stage/reason")
    if len({row.get("seed") for row in success_attempts}) != expected_successes:
        problems.append("successful attempt seeds are not unique")
    if len({row.get("scene_sha256") for row in success_attempts}) != expected_successes:
        problems.append("successful scene hashes are not unique")
    expected_manifest = {}
    for name, e1 in e1_by_collection.items():
        for row in e1["episodes"]:
            expected_manifest[(name, row["seed"], row["scene_sha256"])] = row
    attempt_by_seed = {int(row["seed"]): row for row in success_attempts}
    if set(attempt_by_seed) != {seed for _, seed, _ in expected_manifest}:
        problems.append("successful attempts and collection manifest seeds differ")
    for name, seed, scene_sha in expected_manifest:
        row = attempt_by_seed.get(seed, {})
        if row.get("scene_sha256") != scene_sha or row.get("source_collection") != name:
            problems.append(f"seed {seed}: attempt/manifest scene hash mismatch")
    for name, verification in verification_by_collection.items():
        if verification.get("passed") is not True or verification.get("problems"):
            problems.append(f"{name}: stored collection verification is not cleanly passed")
        replay = verification.get("replay") or {}
        collection_successes = len(rows_by_collection[name])
        if replay.get("total") != collection_successes or replay.get("passed") != collection_successes:
            problems.append(f"{name}: stored physical replay does not pass all episodes")
        official = verification.get("official") or {}
        if official.get("episodes") != collection_successes or official.get("fps") != 50.0:
            problems.append(f"{name}: official-loader verification does not bind "
                            "every episode at 50 Hz")
        if set(official.get("image_shapes", {})) != set(clean60.CAMERAS):
            problems.append(f"{name}: official-loader verification does not cover all three cameras")

    env = None
    if args.external_root is not None:
        config = common.load_external_config(args.external_root, protocol)
        from bimanual.sim.environment import BimanualEnv
        env = BimanualEnv(config, images=False)

    episode_rows = []
    try:
        for episode in sorted(meta["episodes"], key=lambda row: row["episode_index"]):
            episode_id = int(episode["episode_index"])
            seed = int(episode["seed"])
            episode_dir = view / "episodes" / f"episode_{episode_id:03d}"
            with np.load(episode_dir / "episode.npz", allow_pickle=False) as data:
                required = {"state", "action", "timestamp", "frame_index", "source_episode_index",
                            "seed", "scene_sha256", "source_sha256", "task"}
                missing = sorted(required - set(data.files))
                if missing:
                    problems.append(f"episode {episode_id}: missing arrays {missing}")
                    continue
                state = np.asarray(data["state"])
                action = np.asarray(data["action"])
                timestamp = np.asarray(data["timestamp"])
                frame_index = np.asarray(data["frame_index"])
                source_index = int(data["source_episode_index"])
                npz_seed = int(data["seed"])
                npz_scene = str(data["scene_sha256"])
                npz_source = str(data["source_sha256"])
                task = str(data["task"])
            source_collection = collection_of_row[episode_id]
            diagnostics = resolve_diagnostics(source_collection, attempt_by_seed[seed])
            with np.load(diagnostics / "trace.npz", allow_pickle=False) as trace:
                trace_state = np.asarray(trace["state"])
                trace_action = np.asarray(trace["action"])
                trace_time = np.asarray(trace["time"])
                trace_incomplete = np.asarray(trace["incomplete"])
            physical = common.load_json(diagnostics / "summary.json")
            frames = int(episode["frames"])
            local_problems = []
            if state.shape != (frames, 12) or action.shape != (frames, 12):
                local_problems.append("state/action shape differs from registered [frames,12]")
            if trace_state.shape != (frames, 12) or trace_action.shape != (frames, 12):
                local_problems.append("trace state/action shape differs from view")
            if (not np.isfinite(state).all() or not np.isfinite(action).all()
                    or not np.isfinite(timestamp).all() or not np.isfinite(trace_time).all()):
                local_problems.append("non-finite state/action/time")
            if not np.array_equal(state, trace_state) or not np.array_equal(action, trace_action):
                local_problems.append("dataset state/action differs from diagnostics trace")
            if not clean100_checks.timestamps_nominal(timestamp, frames):
                local_problems.append("dataset timestamps are not exact 50 Hz control frames")
            if not clean100_checks.frame_index_continuous(frame_index, frames):
                local_problems.append("dataset frame indices have gaps or reordering")
            if not clean100_checks.control_clock_regular(trace_time, frames):
                local_problems.append("diagnostics time does not advance by 0.02 s")
            if not clean100_checks.control_clock_aligned(trace_time, timestamp, frames):
                local_problems.append("diagnostics and dataset control clocks are misaligned")
            if np.any(trace_incomplete):
                local_problems.append("an incomplete physics frame entered the view")
            if npz_seed != seed or npz_scene != episode["scene_sha256"]:
                local_problems.append("NPZ seed/scene identity differs from metadata")
            if npz_source != meta.get("source_sha256"):
                local_problems.append("NPZ source fingerprint differs from view metadata")
            if task != protocol["task"]["instruction"]:
                local_problems.append("instruction differs from the red/blue sorting task")
            manifest_row = expected_manifest.get(
                (source_collection.name, seed, episode["scene_sha256"]))
            if manifest_row is None:
                local_problems.append("no collection manifest row binds this episode")
            elif source_index != int(manifest_row["source_episode_index"]):
                local_problems.append("source episode index differs from collection manifest")
            if physical.get("success") is not True or physical.get("sorted_count") != 4:
                local_problems.append("physical original success predicate did not pass")
            if physical.get("status", {}).get("success_mode") != "strict":
                local_problems.append("physical success was not evaluated in strict mode")
            if physical.get("contact_monitor_error_count") != 0:
                local_problems.append("physical contact monitor reported errors")
            if physical.get("soft_contact_count") != 0:
                local_problems.append("physical episode contains soft contacts")
            for name, status in (physical.get("status") or {}).get("objects", {}).items():
                if not (status.get("sorted") and status.get("contained") and status.get("supported")
                        and not status.get("robot_contacts") and not status.get("out_of_bounds")
                        and f"{status.get('color')}_tray_0" in status.get("supported_geoms", [])):
                    local_problems.append(f"{name}: original object predicate failed")
            if len((physical.get("status") or {}).get("objects", {})) != 4:
                local_problems.append("original success status does not contain four objects")
            if env is not None:
                reapplied = np.stack([
                    np.asarray(env.robot.apply_action(env.data, row), dtype=np.float32)
                    for row in action], axis=0)
                if not np.array_equal(reapplied, action):
                    local_problems.append("dataset action differs from re-applied control target")
            if args.verify_images:
                local_problems.extend(image_problems(episode_dir, frames))
            problems.extend(f"episode {episode_id}: {problem}" for problem in local_problems)
            episode_rows.append({
                "episode_index": episode_id,
                "seed": seed,
                "split": episode["split"],
                "scene_sha256": episode["scene_sha256"],
                "frames": frames,
                "actions_checked_as_applied": env is not None,
                "images_checked": args.verify_images,
                "state_action_trace_exact": not any("trace" in item for item in local_problems),
                "strict_predicate_passed": physical.get("success") is True,
                "problems": local_problems,
            })
    finally:
        if env is not None:
            env.close()

    if args.external_root is not None and not episode_rows:
        problems.append("external action audit produced no episode rows")
    payload = {
        "schema_version": 1,
        "collection": [collection.name for collection in collections],
        "view": view.name,
        "view_sha256": meta.get("view_sha256"),
        "recomputed_episode_tree_sha256": tree_sha,
        "attempts": {
            "total": len(attempts), "successes": len(success_attempts),
            "failures": len(failed_attempts),
            "failure_reasons": {
                reason: sum(row.get("failure_reason") == reason for row in failed_attempts)
                for reason in sorted({row.get("failure_reason") for row in failed_attempts})},
            "by_collection": {name: sum(row.get("source_collection") == name for row in attempts)
                              for name in collection_by_name},
            "all_rows": attempts,
        },
        "verification": {
            name: {
                "stored_passed": verification.get("passed"),
                "replay_total": (verification.get("replay") or {}).get("total"),
                "replay_passed": (verification.get("replay") or {}).get("passed"),
                "official": verification.get("official"),
            }
            for name, verification in verification_by_collection.items()
        },
        "images_decoded": args.verify_images,
        "actions_reapplied": env is not None,
        "episodes": episode_rows,
        "problems": problems,
        "passed": not problems,
    }
    common.write_json(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in {"attempts", "episodes"}}, indent=2)[:12000])
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
