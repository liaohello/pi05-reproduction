#!/usr/bin/env python3
"""Audit every selected LIBERO episode, frame stream, and scene fingerprint."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import libero_data  # noqa: E402


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def image_bytes(value) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict):
        payload = value.get("bytes")
        if isinstance(payload, bytes):
            return payload
        path = value.get("path")
        if path:
            return Path(path).read_bytes()
    raise ValueError(f"unsupported image payload: {type(value).__name__}")


def decode_image(value):
    from PIL import Image
    import numpy as np

    payload = image_bytes(value)
    with Image.open(io.BytesIO(payload)) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb)
        gray = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    dhash = 0
    for row in range(8):
        for column in range(8):
            dhash = (dhash << 1) | int(gray[row, column] > gray[row, column + 1])
    return {
        "sha256": sha256_bytes(payload),
        "dhash64": f"{dhash:016x}",
        "shape": list(array.shape),
        "channels": int(array.shape[-1]),
        "dtype": str(array.dtype),
        "mean_rgb": [float(value) for value in array.reshape(-1, 3).mean(axis=0)],
    }


def hamming64(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def union_find_groups(count: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left, right in pairs:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    groups: dict[int, list[int]] = {}
    for value in range(count):
        groups.setdefault(find(value), []).append(value)
    return sorted(groups.values(), key=lambda row: min(row))


OBJECT_REGIONS = {
    "bowl": {"box": (105, 120, 158, 166), "search": (65, 90, 195, 195), "min_score": 0.30},
    "plate": {"box": (98, 170, 180, 225), "search": (65, 145, 200, 250), "min_score": 0.40},
}
SCENE_REFERENCE_EPISODE = 507
SCENE_CENTER_TOLERANCE_PX = 2.0


def object_centers(reference_edges, image):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 100)
    result = {}
    for name, spec in OBJECT_REGIONS.items():
        x0, y0, x1, y1 = spec["box"]
        template = reference_edges[y0:y1, x0:x1]
        sx0, sy0, sx1, sy1 = spec["search"]
        search = edges[sy0:sy1, sx0:sx1]
        response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(response)
        if score < spec["min_score"]:
            raise ValueError(
                f"{name} template match score {score:.4f} is below {spec['min_score']}")
        width, height = x1 - x0, y1 - y0
        result[name] = {
            "score": float(score),
            "center_px": [
                float(sx0 + location[0] + width / 2),
                float(sy0 + location[1] + height / 2),
            ],
            "box_px": [
                sx0 + location[0],
                sy0 + location[1],
                sx0 + location[0] + width,
                sy0 + location[1] + height,
            ],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-view-sha256", required=True)
    args = parser.parse_args()

    import cv2
    import numpy as np
    import pyarrow.parquet as pq

    view_root = args.view.resolve()
    data_root = args.data_root.resolve()
    view = libero_data.load_view(view_root)
    if view["view_sha256"] != args.expected_view_sha256:
        raise ValueError("view differs from the registered SHA256")
    source_hash_checks = {
        "info": sha256_file(data_root / "meta/info.json") == view["source_info_sha256"],
        "tasks": sha256_file(data_root / "meta/tasks.jsonl") == view["source_tasks_sha256"],
        "episodes": sha256_file(data_root / "meta/episodes.jsonl") == view["source_episodes_sha256"],
    }
    if not all(source_hash_checks.values()):
        raise ValueError(f"source metadata hashes differ from the view: {source_hash_checks}")

    selected = {int(row["episode_index"]): row for row in view["episodes"]}
    train_ids = set(int(value) for value in view["train_episode_ids"])
    validation_ids = set(int(value) for value in view["validation_episode_ids"])
    if set(selected) != train_ids | validation_ids:
        raise ValueError("view episode rows differ from the frozen split")
    if train_ids & validation_ids or len(train_ids) != 39 or len(validation_ids) != 10:
        raise ValueError("view is not a disjoint 39/10 full-episode split")

    episodes = []
    train_states = []
    training_schema_columns = set()
    first_images = {}
    for episode in sorted(selected):
        row = selected[episode]
        path = data_root / row["parquet"]
        if sha256_file(path) != row["parquet_sha256"]:
            raise ValueError(f"episode {episode} parquet SHA256 differs from the view")
        table = pq.read_table(
            path,
            columns=[
                "episode_index", "frame_index", "timestamp", "task_index",
                "state", "actions", "image", "wrist_image",
            ],
        )
        training_schema_columns.update(table.column_names)
        count = table.num_rows
        episode_values = table["episode_index"].to_pylist()
        frame_values = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        task_values = table["task_index"].to_pylist()
        state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        if count != int(row["frames"]) or state.shape != (count, 8) or actions.shape != (count, 7):
            raise ValueError(f"episode {episode} frame/state/action shape mismatch")
        if set(episode_values) != {episode} or set(task_values) != {10}:
            raise ValueError(f"episode {episode} source episode/task mismatch")
        if not np.array_equal(frame_values, np.arange(count)):
            raise ValueError(f"episode {episode} frame indices are not contiguous")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError(f"episode {episode} has non-finite state/action values")
        delta_t = np.diff(timestamps)
        dt_error = (
            float(np.max(np.abs(delta_t - 0.1))) if len(delta_t) else 0.0
        )
        if abs(float(timestamps[0])) > 1e-6 or dt_error > 2e-5:
            raise ValueError(f"episode {episode} timestamp spacing differs from 10 Hz")

        base_images = [decode_image(value) for value in table["image"].to_pylist()]
        wrist_images = [decode_image(value) for value in table["wrist_image"].to_pylist()]
        if any(image["shape"] != [256, 256, 3] for image in base_images + wrist_images):
            raise ValueError(f"episode {episode} contains a non-256x256 RGB frame")
        trajectory = hashlib.sha256()
        trajectory.update(state.astype("<f4").tobytes())
        trajectory.update(actions.astype("<f4").tobytes())
        first_payload = image_bytes(table["image"][0].as_py())
        from PIL import Image
        first_images[episode] = np.asarray(
            Image.open(io.BytesIO(first_payload)).convert("RGB"))
        split = "train" if episode in train_ids else "validation"
        if split != row["split"]:
            raise ValueError(f"episode {episode} split differs from metadata")
        if split == "train":
            train_states.append(state)
        episodes.append({
            "episode_index": episode,
            "split": split,
            "frames": count,
            "parquet_sha256": row["parquet_sha256"],
            "trajectory_state_action_sha256": trajectory.hexdigest(),
            "timestamp_start": float(timestamps[0]),
            "timestamp_end": float(timestamps[-1]),
            "timestamp_max_abs_dt_error_s": dt_error,
            "frame_indices_contiguous": True,
            "task_index": 10,
            "state_shape": list(state.shape),
            "action_shape": list(actions.shape),
            "state_action_finite": True,
            "base_rgb_frames_decoded": count,
            "wrist_rgb_frames_decoded": count,
            "base_image_shape": base_images[0]["shape"],
            "wrist_image_shape": wrist_images[0]["shape"],
            "initial_state": state[0].astype(float).tolist(),
            "final_state": state[-1].astype(float).tolist(),
            "first_base_image_sha256": base_images[0]["sha256"],
            "first_wrist_image_sha256": wrist_images[0]["sha256"],
            "first_base_dhash64": base_images[0]["dhash64"],
            "first_wrist_dhash64": wrist_images[0]["dhash64"],
            "first_base_mean_rgb": base_images[0]["mean_rgb"],
            "first_wrist_mean_rgb": wrist_images[0]["mean_rgb"],
            "action_min": actions.min(axis=0).astype(float).tolist(),
            "action_max": actions.max(axis=0).astype(float).tolist(),
        })

    reference = first_images[SCENE_REFERENCE_EPISODE]
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    reference_edges = cv2.Canny(reference_gray, 40, 100)
    for episode in episodes:
        episode["scene_objects"] = object_centers(
            reference_edges, first_images[int(episode["episode_index"])])

    train_state_matrix = np.concatenate(train_states, axis=0)
    train_std = train_state_matrix.std(axis=0)
    train_std = np.where(train_std > 1e-12, train_std, np.inf)
    nearest = []
    near_pairs = []
    for left in range(len(episodes)):
        for right in range(left + 1, len(episodes)):
            first, second = episodes[left], episodes[right]
            base_hamming = hamming64(
                first["first_base_dhash64"], second["first_base_dhash64"])
            wrist_hamming = hamming64(
                first["first_wrist_dhash64"], second["first_wrist_dhash64"])
            state_delta = (
                np.asarray(first["initial_state"], dtype=np.float64)
                - np.asarray(second["initial_state"], dtype=np.float64)
            )
            state_rms_std = float(np.sqrt(np.mean((state_delta / train_std) ** 2)))
            bowl_center_distance = float(np.linalg.norm(
                np.asarray(first["scene_objects"]["bowl"]["center_px"])
                - np.asarray(second["scene_objects"]["bowl"]["center_px"])))
            plate_center_distance = float(np.linalg.norm(
                np.asarray(first["scene_objects"]["plate"]["center_px"])
                - np.asarray(second["scene_objects"]["plate"]["center_px"])))
            nearest.append({
                "left_episode": first["episode_index"],
                "right_episode": second["episode_index"],
                "base_dhash_hamming": base_hamming,
                "wrist_dhash_hamming": wrist_hamming,
                "bowl_center_distance_px": bowl_center_distance,
                "plate_center_distance_px": plate_center_distance,
                "initial_state_rms_in_train_std": state_rms_std,
            })
            exact_visual = (
                first["first_base_image_sha256"] == second["first_base_image_sha256"]
                and first["first_wrist_image_sha256"] == second["first_wrist_image_sha256"]
            )
            same_object_geometry = (
                bowl_center_distance <= SCENE_CENTER_TOLERANCE_PX
                and plate_center_distance <= SCENE_CENTER_TOLERANCE_PX
            )
            if exact_visual or same_object_geometry:
                near_pairs.append((left, right))
    nearest.sort(
        key=lambda row: (
            max(
                row["bowl_center_distance_px"],
                row["plate_center_distance_px"],
            ),
            row["left_episode"],
            row["right_episode"],
        )
    )
    groups = union_find_groups(len(episodes), near_pairs)
    families = []
    for family_id, positions in enumerate(groups):
        members = [episodes[position] for position in positions]
        splits = sorted({row["split"] for row in members})
        families.append({
            "family_id": family_id,
            "episode_ids": [row["episode_index"] for row in members],
            "splits": splits,
            "cross_split": len(splits) > 1,
        })
    exact_duplicate_trajectories = []
    for left in range(len(episodes)):
        for right in range(left + 1, len(episodes)):
            if (
                episodes[left]["trajectory_state_action_sha256"]
                == episodes[right]["trajectory_state_action_sha256"]
            ):
                exact_duplicate_trajectories.append([
                    episodes[left]["episode_index"], episodes[right]["episode_index"]
                ])
    checks = {
        "source_metadata_hashes_match": all(source_hash_checks.values()),
        "single_task_only": all(row["task_index"] == 10 for row in episodes),
        "episode_count_49": len(episodes) == 49,
        "split_39_10_disjoint": (
            len(train_ids) == 39
            and len(validation_ids) == 10
            and not train_ids & validation_ids
        ),
        "all_frame_indices_contiguous": True,
        "all_timestamps_10hz": True,
        "all_state_action_finite": True,
        "all_rgb_frames_decode_256x256": True,
        "no_exact_duplicate_trajectories": not exact_duplicate_trajectories,
        "no_scene_family_crosses_split": not any(row["cross_split"] for row in families),
    }
    success_columns = sorted(
        name for name in training_schema_columns
        if any(token in name.lower() for token in ("success", "reward", "done", "terminal"))
    )
    payload = {
        "schema_version": 1,
        "view_sha256": view["view_sha256"],
        "task": view["task"],
        "dataset_task_index": view["dataset_task_index"],
        "source_hash_checks": source_hash_checks,
        "episodes": episodes,
        "frame_totals": {
            "train": sum(row["frames"] for row in episodes if row["split"] == "train"),
            "validation": sum(
                row["frames"] for row in episodes if row["split"] == "validation"),
        },
        "scene_families": {
            "method": (
                "First-frame bowl and plate centers are tracked by edge-template "
                "matching from reference episode 507. Pairs are one scene family "
                "when both object-center distances are <= 2.0 pixels, covering "
                "+/-1 pixel template-match quantization; connected components."
            ),
            "reference_episode": SCENE_REFERENCE_EPISODE,
            "center_tolerance_px": SCENE_CENTER_TOLERANCE_PX,
            "object_regions": OBJECT_REGIONS,
            "families": families,
            "near_pairs": [
                [episodes[left]["episode_index"], episodes[right]["episode_index"]]
                for left, right in near_pairs
            ],
            "nearest_pairs": nearest[:12],
        },
        "success_label_columns_present": success_columns,
        "success_label_evidence": (
            "Absent from the selected LeRobot parquet schema; recordings are treated as "
            "source demonstrations, not independently re-scored benchmark successes."
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "sha256": sha256_file(args.out),
        "passed": payload["passed"],
        "frames": payload["frame_totals"],
        "families": len(families),
        "near_pairs": len(near_pairs),
        "nearest": nearest[:3],
        "success_label_columns_present": success_columns,
        "checks": checks,
    }, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
