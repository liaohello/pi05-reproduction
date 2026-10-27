#!/usr/bin/env python3
"""Re-derive E4-T open-loop metrics on the sampled trajectories themselves.

The evaluator stores every noise copy in ``arrays.npz`` but the first pass summarised the
arithmetic mean of those copies.  That mean is not a trajectory the sampler emits, it lowers the
continuous error, and it destroys the one-frame gripper transitions that the key-event timing gate
measures.  This script recomputes the metrics per sampled copy, aggregates episodes with equal
weight inside a group (the registered rule), and keeps the old mean-of-copies numbers beside the new
ones so the correction is auditable rather than silent.

It reads only recorded predictions; no simulator, checkpoint or GPU is touched.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import so101_common as common  # noqa: E402

JOINT = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
GRIPPER = [5, 11]
LEFT_JOINTS = [0, 1, 2, 3, 4]
RIGHT_JOINTS = [6, 7, 8, 9, 10]
EVENT_PAIRING_LIMIT_FRAMES = 10

GROUPS = {
    "all": lambda r: True,
    "clean": lambda r: r["category"] == "clean",
    "clean_active": lambda r: r["category"] == "clean" and r["phase"] == "active",
    "clean_static": lambda r: r["category"] == "clean" and r["phase"] == "static",
    "recovery": lambda r: r["category"] != "clean",
    "recovery_pre_active": lambda r: r["category"] != "clean"
    and r["disturbance_phase"] == "pre_event" and r["phase"] == "active",
    "recovery_post_active": lambda r: r["category"] != "clean"
    and r["disturbance_phase"] == "post_event" and r["phase"] == "active",
    "recovery_post_static": lambda r: r["category"] != "clean"
    and r["disturbance_phase"] == "post_event" and r["phase"] == "static",
}
for _key in ("A", "B", "C"):
    GROUPS[f"recovery_{_key}"] = (lambda key: lambda r: r["category"] == key)(_key)
    GROUPS[f"recovery_{_key}_post_active"] = (
        lambda key: lambda r: r["category"] == key
        and r["disturbance_phase"] == "post_event" and r["phase"] == "active")(_key)


def detect_gripper_events(actions, threshold) -> list[dict]:
    result = []
    for arm, column in (("left", 5), ("right", 11)):
        states = np.asarray(actions, dtype=float)[:, column]
        for frame in range(1, len(states)):
            if states[frame - 1] >= threshold["open_percent"] and states[frame] <= threshold["closed_percent"]:
                result.append({"arm": arm, "event": "close", "frame": frame})
            if states[frame - 1] <= threshold["closed_percent"] and states[frame] >= threshold["open_percent"]:
                result.append({"arm": arm, "event": "release", "frame": frame})
    return result


def pair_events(gt_events, pred_events) -> tuple[list[dict], list[dict]]:
    pairs, misses = [], []
    for ground in gt_events:
        candidates = [item for item in pred_events
                      if item["arm"] == ground["arm"] and item["event"] == ground["event"]]
        if not candidates:
            misses.append({"kind": ground["event"], "arm": ground["arm"], "reason": "no prediction"})
            continue
        nearest = min(candidates, key=lambda item: abs(item["frame"] - ground["frame"]))
        offset = int(nearest["frame"]) - int(ground["frame"])
        if abs(offset) > EVENT_PAIRING_LIMIT_FRAMES:
            misses.append({"kind": ground["event"], "arm": ground["arm"],
                           "reason": f"nearest {offset:+d} frames"})
            continue
        pairs.append({"kind": ground["event"], "arm": ground["arm"], "error_frames": offset})
    return pairs, misses


def _group_rows(rows: list[dict], predicate) -> list[dict]:
    return [row for row in rows if predicate(row)]


def _episode_macro(rows: list[dict], selector) -> float | None:
    """Mean over episodes of the episode mean, so a long episode cannot dominate a group."""
    by_episode: dict[int, list[float]] = {}
    for row in rows:
        values = selector(row)
        if values is None:
            continue
        by_episode.setdefault(int(row["episode"]), []).extend(
            [float(value) for value in values if value is not None])
    if not by_episode:
        return None
    return float(np.mean([np.mean(values) for values in by_episode.values() if values]))


def _episode_macro_scalar(rows: list[dict], key: str) -> float | None:
    by_episode: dict[int, list[float]] = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            by_episode.setdefault(int(row["episode"]), []).append(float(value))
    if not by_episode:
        return None
    return float(np.mean([np.mean(values) for values in by_episode.values() if values]))


def _pooled_p95(rows: list[dict], key: str) -> float | None:
    values = [value for row in rows for value in row.get(key, [])]
    return float(np.percentile(values, 95)) if values else None


def _episode_p95_macro(rows: list[dict], key: str) -> float | None:
    by_episode: dict[int, list[float]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode"]), []).extend(row.get(key, []))
    if not by_episode:
        return None
    return float(np.mean([np.percentile(values, 95) for values in by_episode.values() if values]))


def group_metrics(rows: list[dict], legacy: dict | None) -> dict:
    if not rows:
        return {}
    per_copy_joint = []
    per_copy_gripper = []
    for copy_index in range(int(rows[0]["copies"])):
        per_copy_joint.append(_episode_macro(rows, lambda row, c=copy_index: [row["joint_mae"][c]]))
        per_copy_gripper.append(_episode_macro(rows, lambda row, c=copy_index: [row["gripper_mae"][c]]))
    return {
        "windows": len(rows),
        "episodes": len({row["episode"] for row in rows}),
        "joint_macro_mae_deg": _episode_macro(rows, lambda row: row["joint_mae"]),
        "gripper_macro_mae_pct": _episode_macro(rows, lambda row: row["gripper_mae"]),
        "joint_p95_deg": _pooled_p95(rows, "joint_error_pool"),
        "joint_p95_episode_macro_deg": _episode_p95_macro(rows, "joint_error_pool"),
        "joint_p95_deg_legacy_p95_of_window_p95": (
            float(np.percentile([row["legacy_p95"] for row in rows], 95))
            if rows and rows[0].get("legacy_p95") is not None else None),
        "per_copy_joint_macro_mae_deg": per_copy_joint,
        "per_copy_gripper_macro_mae_pct": per_copy_gripper,
        "left_joint_mae_deg": _episode_macro(rows, lambda row: row["left_joint_mae"]),
        "right_joint_mae_deg": _episode_macro(rows, lambda row: row["right_joint_mae"]),
        "left_gripper_mae_pct": _episode_macro(rows, lambda row: row["left_gripper_mae"]),
        "right_gripper_mae_pct": _episode_macro(rows, lambda row: row["right_gripper_mae"]),
        "true12_mse": _episode_macro_scalar(rows, "flow_true12_mse"),
        "padding20_mse": _episode_macro_scalar(rows, "flow_padding20_mse"),
        "action12_mse": _episode_macro_scalar(rows, "action12_mse"),
        "hold_joint_mae_deg": _episode_macro_scalar(rows, "hold_joint_mae_deg"),
        "train_mean_joint_mae_deg": _episode_macro_scalar(rows, "train_mean_joint_mae_deg"),
        "hold_gripper_mae_pct": _episode_macro_scalar(rows, "hold_gripper_mae_pct"),
        "train_mean_gripper_mae_pct": _episode_macro_scalar(rows, "train_mean_gripper_mae_pct"),
        "pred_delta_rms_deg_per_frame": _episode_macro(rows, lambda row: row["pred_delta_rms"]),
        "gt_delta_rms_deg_per_frame": _episode_macro_scalar(rows, "gt_delta_rms"),
        "pred_reversals_per_window": _episode_macro(rows, lambda row: row["pred_reversals"]),
        "legacy_mean_of_copies": legacy or {},
    }


def horizon_metrics_by_group(records: list[dict], ground_truth: np.ndarray,
                             predictions: np.ndarray) -> dict:
    """Compute episode-equal joint error at each action horizon for every registered group."""
    result = {}
    for name, predicate in GROUPS.items():
        indices = [index for index, row in enumerate(records) if predicate(row)]
        if not indices:
            continue
        episodes = sorted({int(records[index]["episode"]) for index in indices})
        mae_by_horizon, p95_by_horizon = {}, {}
        mae_by_joint, p95_by_joint = {}, {}
        gripper_mae_by_horizon, gripper_p95_by_horizon = {}, {}
        gripper_mae_by_joint, gripper_p95_by_joint = {}, {}
        for horizon in range(10):
            pred = predictions[indices, :, horizon, :][:, :, JOINT]
            truth = ground_truth[indices, horizon, :][:, JOINT]
            errors = np.abs(pred - truth[:, None, :])
            episode_joint_means = []
            for episode in episodes:
                mask = [int(records[index]["episode"]) == episode for index in indices]
                episode_joint_means.append(errors[mask].mean(axis=(0, 1)))
            joint_macro = np.mean(np.stack(episode_joint_means), axis=0)
            mae_by_horizon[str(horizon)] = float(joint_macro.mean())
            p95_by_horizon[str(horizon)] = float(np.percentile(errors, 95))
            mae_by_joint[str(horizon)] = [float(value) for value in joint_macro]
            p95_by_joint[str(horizon)] = [
                float(value) for value in np.percentile(errors, 95, axis=(0, 1))]
            pred_gripper = predictions[indices, :, horizon, :][:, :, GRIPPER]
            truth_gripper = ground_truth[indices, horizon, :][:, GRIPPER]
            gripper_errors = np.abs(pred_gripper - truth_gripper[:, None, :])
            episode_gripper_means = []
            for episode in episodes:
                mask = [int(records[index]["episode"]) == episode for index in indices]
                episode_gripper_means.append(gripper_errors[mask].mean(axis=(0, 1)))
            gripper_macro = np.mean(np.stack(episode_gripper_means), axis=0)
            gripper_mae_by_horizon[str(horizon)] = float(gripper_macro.mean())
            gripper_p95_by_horizon[str(horizon)] = float(np.percentile(gripper_errors, 95))
            gripper_mae_by_joint[str(horizon)] = [float(value) for value in gripper_macro]
            gripper_p95_by_joint[str(horizon)] = [
                float(value) for value in np.percentile(gripper_errors, 95, axis=(0, 1))]
        result[name] = {
            "windows": len(indices), "episodes": len(episodes),
            "joint_macro_mae_deg": mae_by_horizon,
            "joint_p95_deg": p95_by_horizon,
            "joint_mae_deg_by_joint": mae_by_joint,
            "joint_p95_deg_by_joint": p95_by_joint,
            "gripper_macro_mae_pct": gripper_mae_by_horizon,
            "gripper_p95_pct": gripper_p95_by_horizon,
            "gripper_mae_pct_by_joint": gripper_mae_by_joint,
            "gripper_p95_pct_by_joint": gripper_p95_by_joint,
        }
    return result


def timing_block(pairs: list[dict], misses: list[dict], kind: str, maximum_copies: int) -> dict:
    kind_pairs = [pair for pair in pairs if pair["kind"] == kind]
    kind_misses = [miss for miss in misses if miss["kind"] == kind]
    offsets = [abs(int(pair["error_frames"])) for pair in kind_pairs]
    per_copy = []
    for copy_index in range(maximum_copies):
        copy_offsets = [abs(int(pair["error_frames"])) for pair in kind_pairs
                        if pair.get("copy") == copy_index]
        copy_misses = [miss for miss in kind_misses if miss.get("copy") == copy_index]
        per_copy.append({
            "copy": copy_index,
            "paired": len(copy_offsets),
            "unpaired": len(copy_misses),
            "max_abs_frames": int(max(copy_offsets)) if copy_offsets else None,
            "mean_abs_frames": float(np.mean(copy_offsets)) if copy_offsets else None,
        })
    return {
        "ground_truth_events": int(rows_ground_count(kind_pairs, kind_misses, maximum_copies)),
        "paired": len(kind_pairs),
        "unpaired": len(kind_misses),
        "max_abs_frames": int(max(offsets)) if offsets else None,
        "mean_abs_frames": float(np.mean(offsets)) if offsets else None,
        "within_3_frames": int(sum(1 for value in offsets if value <= 3)),
        "paired_offset_distribution": {str(offset): offsets.count(offset)
                                       for offset in sorted(set(offsets))},
        "measurable": bool(kind_pairs),
        "per_copy": per_copy,
    }


def rows_ground_count(pairs: list[dict], misses: list[dict], copies: int) -> int:
    """Every copy is compared with the same recorded ground-truth events."""
    unique = {(pair.get("episode"), pair.get("start"), pair.get("arm"))
              for pair in pairs} | {(miss.get("episode"), miss.get("start"), miss.get("arm"))
                                    for miss in misses}
    return len(unique) * copies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="one checkpoint evaluation directory (report.json, arrays.npz, records.jsonl.gz)")
    parser.add_argument("--output", type=Path, default=None,
                        help="default: <evaluation>/per_copy.json")
    parser.add_argument("--gates", type=Path,
                        default=common.REPO_ROOT / "configs/so101/e4t_training.json")
    args = parser.parse_args()

    evaluation = args.evaluation.expanduser().resolve()
    output = (args.output or evaluation / "per_copy.json").expanduser().resolve()
    report = common.load_json(evaluation / "report.json")
    with np.load(evaluation / "arrays.npz", allow_pickle=False) as archive:
        ground_truth = np.asarray(archive["ground_truth"], dtype=np.float64)
        predictions = np.asarray(archive["predictions"], dtype=np.float64)
    records = []
    with gzip.open(evaluation / "records.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))

    if ground_truth.ndim != 3 or predictions.ndim != 4:
        raise SystemExit(f"unexpected array shapes {ground_truth.shape} / {predictions.shape}")
    if len(records) != predictions.shape[0]:
        raise SystemExit(f"{len(records)} records but {predictions.shape[0]} prediction windows")
    copies = int(predictions.shape[1])
    threshold = {
        "open_percent": report["thresholds_pretrained_from_train_config"]["open_percent"],
        "closed_percent": report["thresholds_pretrained_from_train_config"]["closed_percent"],
    }

    rows = []
    pairs, misses = [], []
    horizon_values: dict[int, list[float]] = {h: [] for h in range(10)}
    per_episode: dict[int, dict] = {}
    for window_index, record in enumerate(records):
        gt = ground_truth[window_index]
        preds = predictions[window_index]
        if gt.shape != (10, 12) or preds.shape != (copies, 10, 12):
            raise SystemExit(f"window {window_index}: shapes {gt.shape} / {preds.shape}")
        gt_events = detect_gripper_events(gt, threshold)
        errors = np.abs(preds - gt)
        joint_errors = errors[:, :, JOINT]
        gripper_errors = errors[:, :, GRIPPER]
        row = {
            "episode": int(record["episode"]), "start": int(record["start"]),
            "category": record.get("category", "clean"), "subgroup": record.get("subgroup", "clean"),
            "disturbance_phase": record.get("disturbance_phase"), "phase": record.get("phase"),
            "copies": copies,
            "joint_mae": [float(value) for value in joint_errors.mean(axis=(1, 2))],
            "gripper_mae": [float(value) for value in gripper_errors.mean(axis=(1, 2))],
            "left_joint_mae": [float(value) for value in errors[:, :, LEFT_JOINTS].mean(axis=(1, 2))],
            "right_joint_mae": [float(value) for value in errors[:, :, RIGHT_JOINTS].mean(axis=(1, 2))],
            "left_gripper_mae": [float(value) for value in errors[:, :, [5]].mean(axis=(1, 2))],
            "right_gripper_mae": [float(value) for value in errors[:, :, [11]].mean(axis=(1, 2))],
            "joint_error_pool": [float(value) for value in joint_errors.reshape(-1)],
            "true12_mse": record.get("true12_mse"),
            "padding20_mse": record.get("padding20_mse"),
            "flow_true12_mse": record.get("true12_mse"),
            "flow_padding20_mse": record.get("padding20_mse"),
            "action12_mse": float(errors[:, :, :12].mean()),
            "hold_joint_mae_deg": record.get("hold_joint_mae_deg"),
            "train_mean_joint_mae_deg": record.get("mean_joint_mae_deg"),
            "hold_gripper_mae_pct": record.get("hold_gripper_mae_pct"),
            "train_mean_gripper_mae_pct": record.get("mean_gripper_mae_pct"),
            "pred_delta_rms": [float(np.sqrt(np.mean(np.diff(preds[copy][:, JOINT], axis=0) ** 2)))
                               for copy in range(copies)],
            "gt_delta_rms": float(np.sqrt(np.mean(np.diff(gt[:, JOINT], axis=0) ** 2))),
            "pred_reversals": [float(np.count_nonzero(
                (np.diff(preds[copy][:, JOINT], axis=0)[1:] * np.diff(preds[copy][:, JOINT], axis=0)[:-1]) < 0))
                for copy in range(copies)],
            "legacy_p95": record.get("p95_joint_deg"),
        }
        rows.append(row)
        for copy_index in range(copies):
            copy_pred_events = detect_gripper_events(preds[copy_index], threshold)
            copy_pairs, copy_misses = pair_events(gt_events, copy_pred_events)
            for pair in copy_pairs:
                pairs.append({**pair, "copy": copy_index, "episode": row["episode"],
                              "start": row["start"]})
            for miss in copy_misses:
                misses.append({**miss, "copy": copy_index, "episode": row["episode"],
                               "start": row["start"]})
            for horizon in range(10):
                horizon_values[horizon].append(
                    float(joint_errors[copy_index, horizon].mean()))
        episode = per_episode.setdefault(row["episode"], {
            "episode": row["episode"], "windows": 0, "active_windows": 0,
            "joint_mae": [], "gripper_mae": [], "p95_pool": []})
        episode["windows"] += 1
        episode["active_windows"] += int(row["phase"] == "active")
        episode["joint_mae"].extend(row["joint_mae"])
        episode["gripper_mae"].extend(row["gripper_mae"])
        episode["p95_pool"].extend(row["joint_error_pool"])

    aggregates = {}
    for name, predicate in GROUPS.items():
        group_rows = _group_rows(rows, predicate)
        legacy = (report.get("aggregates") or {}).get(name) or {}
        if group_rows:
            aggregates[name] = group_metrics(group_rows, legacy)
    horizon = {str(key): float(np.mean(values)) if values else None
               for key, values in horizon_values.items()}
    timing = {
        "pairing_limit_frames": EVENT_PAIRING_LIMIT_FRAMES,
        "operator": "per sampled trajectory; nearest same-arm event; maximum frame offset",
        "close": timing_block(pairs, misses, "close", copies),
        "release": timing_block(pairs, misses, "release", copies),
    }
    for episode in per_episode.values():
        episode["joint_mae_deg"] = float(np.mean(episode["joint_mae"]))
        episode["gripper_mae_pct"] = float(np.mean(episode["gripper_mae"]))
        episode["joint_p95_deg"] = (float(np.percentile(episode["p95_pool"], 95))
                                    if episode["p95_pool"] else None)
        episode.pop("p95_pool")

    document = {
        "schema_version": 1, "substage": "E4-T-T2-per-copy",
        "label": report.get("label"), "split": report.get("split"),
        "checkpoint_train_state_step": report.get("checkpoint_train_state_step"),
        "windows": len(rows), "noise_copies": copies,
        "metric_source": "per_noise_copy_episode_equal",
        "aggregation": {
            "mae": "mean over sampled copies, then mean over windows within an episode, then equal weight per episode",
            "p95": "95th percentile of every elementwise joint error on the sampled trajectories",
            "event_timing": "per sampled trajectory, maximum absolute frame offset after nearest same-arm pairing",
        },
        "aggregates": aggregates,
        "horizon_metrics_by_group": horizon_metrics_by_group(records, ground_truth, predictions),
        "horizon_mae_joint_deg": horizon,
        "event_timing": timing,
        "per_episode": [per_episode[key] for key in sorted(per_episode)],
        "legacy_mean_of_copies": {
            "aggregates": report.get("aggregates"),
            "horizon_mae_joint_deg": report.get("horizon_mae_joint_deg"),
            "event_timing": report.get("event_timing"),
            "note": "the first pass averaged the noise copies into a trajectory the sampler never emits; kept for audit only",
        },
        "gates_sha256": common.sha256_json(common.load_json(args.gates).get("validation_gate")),
        "action_executed_in_simulation": False,
        "noise_set_sha256": report.get("noise_set_sha256"),
    }
    common.write_json(output, document)
    print(json.dumps({
        "label": document["label"], "step": document["checkpoint_train_state_step"],
        "clean_active": aggregates.get("clean_active", {}).get("joint_macro_mae_deg"),
        "recovery_post_active": aggregates.get("recovery_post_active", {}).get("joint_macro_mae_deg"),
        "event_timing_close": timing["close"]["max_abs_frames"],
        "event_timing_release": timing["release"]["max_abs_frames"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
