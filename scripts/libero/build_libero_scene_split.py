#!/usr/bin/env python3
"""Create a versioned LIBERO view whose split respects scene families."""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audit_libero_data as data_audit  # noqa: E402
import libero_data  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: int, episode_ids) -> str:
    text = f"libero-scene-family|{seed}|" + ",".join(str(x) for x in sorted(episode_ids))
    return hashlib.sha256(text.encode()).hexdigest()


def stats(values):
    import numpy as np

    return {
        "mean": np.asarray(values.mean(axis=0), dtype=float).tolist(),
        "std": np.asarray(values.std(axis=0), dtype=float).tolist(),
        "q01": np.asarray(np.quantile(values, 0.01, axis=0), dtype=float).tolist(),
        "q99": np.asarray(np.quantile(values, 0.99, axis=0), dtype=float).tolist(),
    }


def choose_validation_families(families: list[list[int]], target: int, seed: int):
    ranked = sorted(families, key=lambda ids: stable_key(seed, ids))
    # DP maximises the number of untouched validation families. Ties use the
    # stable family order so retries choose the same split.
    states = {0: (0, ())}
    for position, family in enumerate(ranked):
        size = len(family)
        for total, (count, chosen) in list(states.items())[::-1]:
            new_total = total + size
            if new_total > target:
                continue
            candidate = (count + 1, chosen + (position,))
            current = states.get(new_total)
            if current is None or candidate > current:
                states[new_total] = candidate
    if target not in states:
        raise ValueError(f"cannot select {target} validation episodes by whole family")
    chosen_positions = states[target][1]
    selected = [ranked[position] for position in chosen_positions]
    remainder = list(families)
    for family in selected:
        remainder.remove(family)
    return selected, remainder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-view", type=Path, required=True)
    parser.add_argument("--expected-source-view-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene-center-tolerance-px", type=float, default=2.0)
    parser.add_argument("--action-horizon", type=int, default=25)
    args = parser.parse_args()

    import cv2
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image

    source_view = libero_data.load_view(args.source_view)
    if source_view["view_sha256"] != args.expected_source_view_sha256:
        raise ValueError("source view differs from the registered hash")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"refusing non-empty output view directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "assets" / "libero_goal8").mkdir(parents=True)

    rows = sorted(source_view["episodes"], key=lambda row: int(row["episode_index"]))
    first_images = {}
    states = {}
    actions = {}
    for row in rows:
        episode = int(row["episode_index"])
        table = pq.read_table(
            args.data_root / row["parquet"], columns=["image", "state", "actions"])
        image_file = table["image"][0].as_py()
        payload = data_audit.image_bytes(image_file)
        first_images[episode] = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
        states[episode] = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        actions[episode] = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
    reference = first_images[data_audit.SCENE_REFERENCE_EPISODE]
    reference_edges = cv2.Canny(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), 40, 100)
    centers = {
        episode: data_audit.object_centers(reference_edges, image)
        for episode, image in first_images.items()
    }
    episode_ids = [int(row["episode_index"]) for row in rows]
    near_pairs = []
    for left, right in itertools.combinations(episode_ids, 2):
        first = centers[left]
        second = centers[right]
        bowl_distance = float(np.linalg.norm(
            np.asarray(first["bowl"]["center_px"])
            - np.asarray(second["bowl"]["center_px"])))
        plate_distance = float(np.linalg.norm(
            np.asarray(first["plate"]["center_px"])
            - np.asarray(second["plate"]["center_px"])))
        if (
            bowl_distance <= args.scene_center_tolerance_px
            and plate_distance <= args.scene_center_tolerance_px
        ):
            near_pairs.append((episode_ids.index(left), episode_ids.index(right)))
    groups = data_audit.union_find_groups(len(episode_ids), near_pairs)
    families = [[episode_ids[index] for index in group] for group in groups]
    validation_families, train_families = choose_validation_families(families, 10, args.seed)
    validation_ids = sorted(value for family in validation_families for value in family)
    train_ids = sorted(value for family in train_families for value in family)
    if len(train_ids) != 39 or len(validation_ids) != 10:
        raise ValueError("scene-family split did not produce 39/10 episodes")
    if set(train_ids) & set(validation_ids):
        raise ValueError("scene-family split overlaps")

    train_state = np.concatenate([states[value] for value in train_ids], axis=0)
    train_action = np.concatenate([actions[value] for value in train_ids], axis=0)
    validation_state = np.concatenate([states[value] for value in validation_ids], axis=0)
    validation_action = np.concatenate([actions[value] for value in validation_ids], axis=0)
    norm = {"norm_stats": {"state": stats(train_state), "actions": stats(train_action)}}
    norm_path = args.out / "assets" / "libero_goal8" / "norm_stats.json"
    norm_path.write_text(json.dumps(norm, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    view = json.loads(json.dumps(source_view))
    view.pop("view_sha256", None)
    view["schema_version"] = 2
    view["view_id"] = (
        f"{source_view['view_id']}-scene-v1-h{args.action_horizon}")
    view["source_view_sha256"] = source_view["view_sha256"]
    view["seed"] = args.seed
    view["action_horizon"] = args.action_horizon
    view["train_episode_ids"] = train_ids
    view["validation_episode_ids"] = validation_ids
    for row in view["episodes"]:
        row["split"] = (
            "train" if int(row["episode_index"]) in set(train_ids) else "validation")
    view["norm_stats"] = {
        "path": "assets/libero_goal8/norm_stats.json",
        "sha256": sha256_file(norm_path),
        "train_only": True,
    }
    view["constraints"]["scene_family_atomic"] = True
    family_rank = {id(family): position for position, family in enumerate(families)}
    view["scene_split"] = {
        "method": (
            "First-frame edge-template tracking of bowl and plate; pairs whose "
            "object centers are both within the registered pixel tolerance form "
            "one transitive family."
        ),
        "reference_episode": data_audit.SCENE_REFERENCE_EPISODE,
        "center_tolerance_px": args.scene_center_tolerance_px,
        "selection": (
            "Validation uses whole families and maximises the number of validation "
            "families; remaining families form train."
        ),
        "families": [
            {
                "family_id": family_rank[id(family)],
                "episode_ids": sorted(family),
                "split": "validation" if family in validation_families else "train",
                "first_frame_object_centers_px": {
                    str(episode): centers[episode] for episode in sorted(family)
                },
            }
            for family in families
        ],
    }
    canonical = json.dumps(view, sort_keys=True, separators=(",", ":")).encode("utf-8")
    view["view_sha256"] = hashlib.sha256(canonical).hexdigest()
    (args.out / "view.json").write_text(
        json.dumps(view, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    train_stats = {
        "split": "train",
        "episodes": len(train_ids),
        "frames": int(train_state.shape[0]),
        "state": stats(train_state),
        "actions": stats(train_action),
        "validation_included": False,
        "norm_stats_sha256": sha256_file(norm_path),
    }
    validation_stats = {
        "split": "validation",
        "episodes": len(validation_ids),
        "frames": int(validation_state.shape[0]),
        "state": stats(validation_state),
        "actions": stats(validation_action),
    }
    (args.out / "train_stats.json").write_text(
        json.dumps(train_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "validation_stats.json").write_text(
        json.dumps(validation_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "data_quality.json").write_text(
        json.dumps({
            "source_view_sha256": source_view["view_sha256"],
            "scene_families": view["scene_split"],
            "train_episode_ids": train_ids,
            "validation_episode_ids": validation_ids,
            "split_method": "scene-family-atomic-v1",
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.out),
        "view_sha256": view["view_sha256"],
        "norm_stats_sha256": view["norm_stats"]["sha256"],
        "train_episode_ids": train_ids,
        "validation_episode_ids": validation_ids,
        "train_frames": int(train_state.shape[0]),
        "validation_frames": int(validation_state.shape[0]),
        "families": len(families),
        "validation_families": len(validation_families),
        "train_families": len(train_families),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
