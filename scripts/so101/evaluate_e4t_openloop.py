#!/usr/bin/env python3
"""E4-T / T2: open-loop action prediction for one saved checkpoint.

Validation data is never read by the training loader.  This process loads the saved
run config and one checkpoint, then samples complete 10x12 actions from real flow
noise on fixed windows.  No action is applied to MuJoCo.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import openpi_so101  # noqa: E402
import openpi.training.checkpoints as _checkpoints  # noqa: E402
from openpi.models import model as _model  # noqa: E402
from openpi.models.pi0 import make_attn_mask  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402

JOINT = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
GRIPPER = [5, 11]


def sha(value) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()


def raw_row(view: Path, episode: int, frame: int, action=None) -> dict:
    directory = view / "episodes" / f"episode_{episode:03d}"
    with np.load(directory / "episode.npz", allow_pickle=False) as data:
        state = np.asarray(data["state"][frame], dtype=np.float32)
    from PIL import Image
    images = {}
    for key, name in openpi_so101.CAMERA_FILES.items():
        with Image.open(directory / name / f"{frame:06d}.jpg") as image:
            images[key] = np.asarray(image.convert("RGB"))
    row = {
        "observation.state": state,
        "observation.images.central": images["observation.images.central"],
        "observation.images.left_wrist": images["observation.images.left_wrist"],
        "observation.images.right_wrist": images["observation.images.right_wrist"],
        "prompt": openpi_so101.TASK,
    }
    if action is not None:
        row["action"] = np.asarray(action, dtype=np.float32)
    return row


def flow_error(model, observation, actions, rng):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=False)
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * .999 + .001
    expanded = time[..., None, None]
    x_t = expanded * noise + (1 - expanded) * actions
    u_t = noise - actions
    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    suffix, suffix_mask, suffix_ar, adarms = model.embed_suffix(observation, x_t, time)
    mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
    attn = make_attn_mask(mask, ar)
    positions = jnp.cumsum(mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix, suffix], mask=attn, positions=positions, adarms_cond=[None, adarms])
    v = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    return jnp.square(v - u_t), noise, time


def p95(values):
    return float(np.percentile(values, 95)) if values else None


def macro(values):
    return float(np.mean(values)) if values else None


def phase(row, start, threshold) -> str:
    """Active means the ground truth moves more than the declared floor on its own channel.

    The joint floor is in degrees and the gripper floor in percentage points, so each is compared
    only against its own columns: a gripper creep must not make frozen joints count as active.
    """
    gt = np.asarray(row["ground_truth"], dtype=float)
    derivative = np.abs(np.diff(gt, axis=0, prepend=gt[:1]))
    active = bool(derivative[:, JOINT].max() >= threshold["joint_deg_per_frame"]
                  or derivative[:, GRIPPER].max() >= threshold["gripper_pct_per_frame"])
    return "active" if active else "static"


def detect_gripper_events(actions, threshold) -> list[dict]:
    result = []
    for arm_index, arm in (("left", 5), ("right", 11)):
        states = np.asarray(actions, dtype=float)[:, arm]
        for frame in range(1, len(states)):
            if states[frame - 1] >= threshold["open_percent"] and states[frame] <= threshold["closed_percent"]:
                result.append({"arm": arm, "event": "close", "frame": frame})
            if states[frame - 1] <= threshold["closed_percent"] and states[frame] >= threshold["open_percent"]:
                result.append({"arm": arm, "event": "release", "frame": frame})
    return result


EVENT_PAIRING_LIMIT_FRAMES = 10


def anchored_windows(actions, perturbation_frame, threshold) -> list[dict]:
    """One window per recorded gripper transition, for the supplementary timing pass.

    The frozen plan only sees the transitions that happen to fall inside a planned window
    (experiments/so101/e4t-window-coverage.json), so this anchors a window on every transition
    instead: the start is placed five frames before the change and moved to the nearest legal start
    when that window would straddle the perturbation frame or leave the episode.
    """
    out = []
    for event in detect_gripper_events(actions, threshold):
        frame = int(event["frame"])
        # the transition must sit inside the slice, so the window starts between frame-9 and frame-1
        candidates = range(max(0, frame - 9), min(len(actions) - 10, frame - 1) + 1)
        legal = [start for start in candidates
                 if perturbation_frame is None
                 or not (int(perturbation_frame) - 9 <= start <= int(perturbation_frame))]
        if not legal:
            out.append({"anchor": event, "start": None})
            continue
        out.append({"anchor": event, "start": min(legal, key=lambda s: abs(s - (frame - 5)))})
    return out


def pair_events(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pair every ground-truth gripper event with the nearest predicted event of the same arm.

    A ground-truth event with no predicted event inside EVENT_PAIRING_LIMIT_FRAMES is a miss and is
    reported as one; it is never silently dropped, and a far-away prediction is not allowed to count
    as a match.
    """
    pairs, misses = [], []
    for kind in ("close", "release"):
        for row in records:
            for ground in row[f"gt_{kind}_events"]:
                candidates = [item for item in row[f"pred_{kind}_events"] if item["arm"] == ground["arm"]]
                if not candidates:
                    misses.append({"kind": kind, "episode": row["episode"], "start": row["start"],
                                   "arm": ground["arm"], "reason": "no predicted event for this arm"})
                    continue
                nearest = min(candidates, key=lambda item: abs(item["frame"] - ground["frame"]))
                offset = int(nearest["frame"]) - int(ground["frame"])
                if abs(offset) > EVENT_PAIRING_LIMIT_FRAMES:
                    misses.append({"kind": kind, "episode": row["episode"], "start": row["start"],
                                   "arm": ground["arm"],
                                   "reason": f"nearest prediction {offset:+d} frames away, beyond the "
                                             f"{EVENT_PAIRING_LIMIT_FRAMES}-frame pairing window"})
                    continue
                pairs.append({"kind": kind, "episode": row["episode"], "start": row["start"],
                              "arm": ground["arm"], "error_frames": offset})
    return pairs, misses


def aggregate_group(rows: list[dict]) -> dict:
    """One group's metrics.

    ``joint_p95_deg`` is the 95th percentile of the per-window elementwise 95th percentiles, so a
    single wild joint cannot be diluted by averaging it across horizons first; the older, weaker
    definition is kept beside it under an explicit name.
    """
    if not rows:
        return {}
    return {
        "windows": len(rows),
        "joint_macro_mae_deg": macro([row["joint_mae_deg"] for row in rows]),
        "joint_p95_deg": p95([row["p95_joint_deg"] for row in rows]),
        "joint_p95_of_horizon_means_deg": p95(
            np.concatenate([row["horizon_mae_joint"] for row in rows]).tolist()),
        "gripper_macro_mae_pct": macro([row["gripper_mae_pct"] for row in rows]),
        "true12_mse": macro([row["true12_mse"] for row in rows]),
        "padding20_mse": macro([row["padding20_mse"] for row in rows]),
        "hold_joint_mae_deg": macro([row["hold_joint_mae_deg"] for row in rows]),
        "train_mean_joint_mae_deg": macro([row["mean_joint_mae_deg"] for row in rows]),
        "hold_gripper_mae_pct": macro([row["hold_gripper_mae_pct"] for row in rows]),
        "train_mean_gripper_mae_pct": macro([row["mean_gripper_mae_pct"] for row in rows]),
        "pred_delta_rms_deg_per_frame": macro([row["pred_delta_rms_deg_per_frame"] for row in rows]),
        "gt_delta_rms_deg_per_frame": macro([row["gt_delta_rms_deg_per_frame"] for row in rows]),
        "pred_reversals_per_window": macro([float(row["pred_sign_reversals"]) for row in rows]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--training-config", type=Path,
                        default=Path("configs/so101/e4t_training.json").resolve())
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--noise-copies", type=int, default=3,
                        help="paired flow-noise copies per window; the driver and the "
                             "registered plan both use 3")
    parser.add_argument("--noise-seed-base", type=int, default=40000)
    parser.add_argument("--flow-seed-base", type=int, default=50000)
    parser.add_argument("--dense-stride", type=int, default=0)
    parser.add_argument("--anchor-events", action="store_true",
                        help="supplementary timing pass: evaluate one window per recorded gripper "
                             "transition instead of the frozen plan; reported as its own window set")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--train-state-step", type=int, required=True)
    parser.add_argument("--assets-root", type=Path, default=None,
                        help="evaluate a checkpoint that carries no assets of its own, e.g. the "
                             "initialisation weights, by reading norm stats from the view")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    view = args.view.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse {output}")
    run_config = json.loads(args.run_config.read_text(encoding="utf-8"))
    training = json.loads(args.training_config.read_text(encoding="utf-8"))
    windows_document = json.loads(args.windows.read_text(encoding="utf-8"))
    meta = json.loads((view / "meta.json").read_text(encoding="utf-8"))
    state_mode = run_config.get("state_mode")
    if state_mode != "on":
        raise SystemExit("E4-T open-loop evaluation requires state_mode=on")
    episodes = [row for row in meta["episodes"] if row["split"] == args.split]
    gate = training["validation_gate"]
    threshold = {
        "joint_deg_per_frame": float(gate["activity_joint_deg_per_frame"]),
        "gripper_pct_per_frame": float(gate["activity_gripper_pct_per_frame"]),
        "open_percent": gate["gripper_open_threshold_percent"],
        "closed_percent": gate["gripper_closed_threshold_percent"],
    }
    anchors: dict[int, list[dict]] = {}
    if args.anchor_events:
        # supplementary pass: one window per recorded gripper transition, see anchored_windows
        for row in episodes:
            with np.load(view / "episodes" / f"episode_{int(row['episode_index']):03d}" / "episode.npz",
                         allow_pickle=False) as data:
                values = np.asarray(data["action"], dtype=np.float32)
            anchors[int(row["episode_index"])] = anchored_windows(
                values, row.get("perturbation_frame"), threshold)
        expected_windows = sum(len({item["start"] for item in anchors[int(row["episode_index"])]
                                    if item["start"] is not None}) for row in episodes)
    elif windows_document:
        def planned_starts(row):
            values = [int(value) for value in
                      windows_document.get(str(int(row["episode_index"])), [])]
            length = int(row["frames"])
            if args.dense_stride:
                # the dense pass unions a strided sweep with the frozen plan, so the completeness
                # check has to count the same set the loop will walk
                values = sorted(set(range(0, length - 9, args.dense_stride)) | set(values))
            return [value for value in values if value + 10 <= length]
        expected_windows = sum(len(planned_starts(row)) for row in episodes)
    else:
        expected_windows = None
    config = openpi_so101.make_train_config(
        mode="serve", run_root=output, train_view=view, base_checkpoint=checkpoint,
        exp_name="e4t_eval", state_mode=state_mode)
    data_config = config.data.create(config.assets_dirs, config.model)
    assets_root = (args.assets_root.expanduser().resolve() if args.assets_root
                   else checkpoint / "assets")
    norm_stats = _checkpoints.load_norm_stats(assets_root, data_config.asset_id)
    policy = _policy_config.create_trained_policy(config, checkpoint, norm_stats=norm_stats,
                                                 default_prompt=openpi_so101.TASK)
    model = policy._model
    graphdef, model_state = nnx.split(model)

    @jax.jit
    def flow_jit(state, observation, actions, rng):
        return flow_error(nnx.merge(graphdef, state), observation, actions, rng)[0]

    train_rows = [row for row in meta["episodes"] if row.get("split") == "train"]
    train_windows = []
    for row in train_rows:
        with np.load(view / "episodes" / f"episode_{int(row['episode_index']):03d}" / "episode.npz",
                     allow_pickle=False) as data:
            values = np.asarray(data["action"], dtype=np.float32)
        for start in row.get("fixed_windows", []):
            if start + 10 <= len(values):
                train_windows.append(values[start:start + 10])
    train_mean_action = (np.mean(np.stack(train_windows), axis=0).astype(np.float32)
                         if train_windows else None)
    records = []
    arrays = {"ground_truth": [], "predictions": [], "noise_copies": args.noise_copies}
    for episode_row in episodes:
        episode = int(episode_row["episode_index"])
        with np.load(view / "episodes" / f"episode_{episode:03d}" / "episode.npz", allow_pickle=False) as data:
            actions_all = np.asarray(data["action"], dtype=np.float32)
        plan_windows = [int(value) for value in windows_document.get(str(episode), [])]
        if args.anchor_events:
            starts = sorted({item["start"] for item in anchors[episode] if item["start"] is not None})
        elif args.dense_stride:
            starts = sorted(set(range(0, len(actions_all) - 9, args.dense_stride)) | set(plan_windows))
        elif plan_windows:
            # the frozen plan already is fixed_windows union event_windows; using it keeps the
            # pre-registered key-phase windows in the evaluation instead of dropping them
            starts = plan_windows
        else:
            starts = list(episode_row["event_windows"] or episode_row["fixed_windows"])
        starts = [start for start in starts if start + 10 <= len(actions_all)]
        for window_number, start in enumerate(starts):
            end = start + 10
            if end > len(actions_all):
                continue
            gt = actions_all[start:end].copy()
            raw_flow = raw_row(view, episode, start, gt)
            inputs = policy._input_transform(copy.deepcopy(raw_flow))
            inputs = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], inputs)
            observation = _model.Observation.from_dict(inputs)
            actions_j = jnp.asarray(inputs["actions"], dtype=jnp.float32)
            flow_rng = jax.random.key(args.flow_seed_base + episode * 1000 + window_number)
            error = np.asarray(flow_jit(model_state, observation, actions_j, flow_rng))[0]
            predictions = []
            noise_hashes = []
            policy._rng = jax.random.key(0)
            for copy_index in range(args.noise_copies):
                noise = np.random.default_rng(
                    args.noise_seed_base + episode * 10000 + window_number * 100 + copy_index).standard_normal(
                    (10, 32)).astype(np.float32)
                inferred = np.asarray(policy.infer(copy.deepcopy(raw_flow), noise=noise)["actions"], dtype=np.float32)
                if inferred.shape != (10, 12):
                    raise SystemExit(f"prediction shape {inferred.shape}")
                predictions.append(inferred)
                noise_hashes.append(sha(noise))
            mean_prediction = np.mean(predictions, axis=0)
            error_mean = np.abs(mean_prediction - gt)
            pred_delta = np.diff(mean_prediction[:, JOINT], axis=0)
            gt_delta = np.diff(gt[:, JOINT], axis=0)
            reversal = (((pred_delta[1:] * pred_delta[:-1]) < 0)
                        & (np.maximum(np.abs(pred_delta[1:]),
                                      np.abs(pred_delta[:-1])) > threshold["joint_deg_per_frame"]))
            pred_sign_reversals = int(np.count_nonzero(reversal))
            with np.load(view / "episodes" / f"episode_{episode:03d}" / "episode.npz",
                         allow_pickle=False) as data:
                state_at_start = np.asarray(data["state"][start], dtype=np.float32)
            hold = np.repeat(state_at_start[None, :], 10, axis=0)
            if train_mean_action is None:
                raise SystemExit("no train-only mean action: the baseline would silently become "
                                 "the ground truth, so the comparison is refused instead")
            train_mean = train_mean_action
            event = episode_row.get("perturbation_frame")
            category = episode_row.get("category", "clean")
            subgroup = episode_row.get("subgroup", "clean")
            disturbance_phase = None
            if event is not None:
                disturbance_phase = "pre_event" if start < int(event) else "post_event"
            records.append({
                "episode": episode, "window": window_number, "start": start,
                "category": category, "subgroup": subgroup, "mechanism": episode_row.get("mechanism"),
                "disturbance_phase": disturbance_phase,
                "phase": phase({"ground_truth": gt}, start, threshold),
                "true12_mse": float(error[:, :12].mean()),
                "padding20_mse": float(error[:, 12:].mean()),
                "all32_mse": float(error.mean()),
                "joint_mae_deg": float(error_mean[:, JOINT].mean()),
                "gripper_mae_pct": float(error_mean[:, GRIPPER].mean()),
                "mae_per_dim": error_mean.mean(axis=0).tolist(),
                "horizon_mae_joint": error_mean[:, JOINT].mean(axis=1).tolist(),
                "horizon_mae_gripper": error_mean[:, GRIPPER].mean(axis=1).tolist(),
                "pred_delta_rms_deg_per_frame": float(np.sqrt(np.mean(pred_delta ** 2))),
                "pred_delta_max_deg_per_frame": float(np.max(np.abs(pred_delta))),
                "gt_delta_rms_deg_per_frame": float(np.sqrt(np.mean(gt_delta ** 2))),
                "pred_sign_reversals": pred_sign_reversals,
                "p95_joint_deg": p95(error_mean[:, JOINT].reshape(-1).tolist()),
                "hold_joint_mae_deg": float(np.abs(hold[:, JOINT] - gt[:, JOINT]).mean()),
                "hold_gripper_mae_pct": float(np.abs(hold[:, GRIPPER] - gt[:, GRIPPER]).mean()),
                "mean_joint_mae_deg": float(np.abs(train_mean[:, JOINT] - gt[:, JOINT]).mean()),
                "mean_gripper_mae_pct": float(np.abs(train_mean[:, GRIPPER] - gt[:, GRIPPER]).mean()),
                "gt_close_events": [row for row in detect_gripper_events(gt, threshold) if row["event"] == "close"],
                "gt_release_events": [row for row in detect_gripper_events(gt, threshold) if row["event"] == "release"],
                "pred_close_events": [row for row in detect_gripper_events(mean_prediction, threshold) if row["event"] == "close"],
                "pred_release_events": [row for row in detect_gripper_events(mean_prediction, threshold) if row["event"] == "release"],
                "noise_hashes": noise_hashes,
            })
            arrays["ground_truth"].append(gt)
            arrays["predictions"].append(np.stack(predictions))
            if args.max_records and len(records) >= args.max_records:
                break
        if args.max_records and len(records) >= args.max_records:
            break

    timing, timing_misses = pair_events(records)
    unanchored = ([item["anchor"] for rows in anchors.values() for item in rows
                   if item["start"] is None] if args.anchor_events else [])
    # One hash for the whole per-window noise set, so a curve across checkpoints can be shown to
    # differ only by weights.  Order-independent: the same draws always give the same digest.
    noise_set_sha256 = (hashlib.sha256(json.dumps(
        {f"{row['episode']}:{row['start']}": sorted(row["noise_hashes"]) for row in records},
        sort_keys=True).encode()).hexdigest() if records else None)
    group = aggregate_group
    aggregates = {
        "all": group(records),
        "clean": group([r for r in records if r["category"] == "clean"]),
        "clean_active": group([r for r in records if r["category"] == "clean" and r["phase"] == "active"]),
        "recovery": group([r for r in records if r["category"] != "clean"]),
        "recovery_pre_active": group([r for r in records if r["category"] != "clean"
                                      and r["disturbance_phase"] == "pre_event" and r["phase"] == "active"]),
        "recovery_post_active": group([r for r in records if r["category"] != "clean"
                                       and r["disturbance_phase"] == "post_event" and r["phase"] == "active"]),
    }
    for key in ("A", "B", "C"):
        rows = [r for r in records if r["category"] == key]
        aggregates[f"recovery_{key}"] = group(rows)
        aggregates[f"recovery_{key}_post_active"] = group(
            [r for r in rows if r["disturbance_phase"] == "post_event" and r["phase"] == "active"])
    for key in sorted({row.get("mechanism") for row in records if row.get("mechanism")}):
        aggregates[f"mechanism_{key}_post_active"] = group(
            [r for r in records if r.get("mechanism") == key and r["disturbance_phase"] == "post_event"
             and r["phase"] == "active"])
    horizon = {
        str(h): macro([r["horizon_mae_joint"][h] for r in records]) for h in range(10)}
    def timing_block(kind: str) -> dict:
        rows = [row for row in timing if row["kind"] == kind]
        ground_count = sum(len(row[f"gt_{kind}_events"]) for row in records)
        missed = [row for row in timing_misses if row["kind"] == kind]
        return {
            "ground_truth_events": ground_count,
            "paired": len(rows),
            "unpaired": len(missed),
            "max_abs_frames": (int(max(abs(row["error_frames"]) for row in rows)) if rows else None),
            "mean_abs_frames": macro([abs(row["error_frames"]) for row in rows]),
            # an unmeasured group is not a pass: the gate treats None as unmeasured
            "measurable": bool(rows),
        }
    timing_summary = {"pairing_limit_frames": EVENT_PAIRING_LIMIT_FRAMES,
                      "close": timing_block("close"), "release": timing_block("release"),
                      "unpaired_examples": timing_misses[:10]}
    output.mkdir(parents=True)
    with gzip.open(output / "records.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    np.savez_compressed(output / "arrays.npz",
                        ground_truth=np.asarray(arrays["ground_truth"]),
                        predictions=np.asarray(arrays["predictions"]))
    report = {
        "schema_version": 1, "phase": "E4-T", "substage": "T2-open-loop",
        "windows_expected": expected_windows,
        "success": bool(records) and bool(aggregates.get("all"))
                   and len(records) == expected_windows,
        "incomplete": (None if expected_windows is None or not records
                       else len(records) < expected_windows),
        "label": args.label, "split": args.split, "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name, "run_config_sha256": hashlib.sha256(args.run_config.read_bytes()).hexdigest(),
        "checkpoint_train_state_step": args.train_state_step,
        "view_sha256": meta.get("view_sha256"),
        "norm_stats_canonical_sha256": run_config.get("norm_stats_canonical_sha256"),
        "windows": len(records), "noise_copies": args.noise_copies,
        "noise_set_sha256": noise_set_sha256,
        "window_set": ("event_anchored_supplementary" if args.anchor_events
                       else "frozen_plan_plus_dense_sweep" if args.dense_stride
                       else "frozen_plan" if windows_document else "view_metadata"),
        "transitions_without_a_legal_window": unanchored,
        "noise_seed_base": args.noise_seed_base, "flow_seed_base": args.flow_seed_base,
        "thresholds_pretrained_from_train_config": threshold,
        "aggregates": aggregates, "horizon_mae_joint_deg": horizon,
        "event_timing": timing_summary,
        "fk_tcp_error": None,
        "fk_note": "pinned OpenPI model exposes no SO101 FK interface; no kinematic proxy is invented",
        "action_executed_in_simulation": False,
        "validation_used_for_training": False,
        "claim": "open-loop action prediction on recorded expert observations; not closed-loop task success",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"label": args.label, "split": args.split, "windows": len(records),
                      "clean_active": aggregates["clean_active"],
                      "recovery_post_active": aggregates["recovery_post_active"],
                      "event_timing": timing_summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
