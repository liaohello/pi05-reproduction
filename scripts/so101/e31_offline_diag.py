#!/usr/bin/env python3
"""E3.1 / G3: fixed 32-window offline diagnostics for one checkpoint.

Flow-matching error is reported separately for the 12 real action dims and
the 20 padded dims, with the target u = noise - actions.  Action prediction
uses fixed inference noise and reports unnormalised joint-angle and gripper
errors separately.  Fixed noise/time is diagnostic only; training remains
random.
"""
from __future__ import annotations

import argparse
import copy
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

TASK = openpi_so101.TASK
JOINT_INDICES = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
GRIPPER_INDICES = [5, 11]


def _raw(view: Path, frame: int, action: np.ndarray | None = None) -> dict:
    episode = view / "episodes" / "episode_000"
    from PIL import Image
    def image(name):
        with Image.open(episode / name / f"{frame:06d}.jpg") as img:
            return np.asarray(img.convert("RGB"))
    value = {
        "observation.state": np.asarray(np.load(episode / "episode.npz")["state"][frame],
                                        dtype=np.float32),
        "observation.images.central": image("central"),
        "observation.images.left_wrist": image("left_wrist"),
        "observation.images.right_wrist": image("right_wrist"),
        "prompt": TASK,
    }
    if action is not None:
        value["action"] = np.asarray(action, dtype=np.float32)
    return value


def _flow_forward(model, observation, actions, rng):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=False)
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions,
        adarms_cond=[None, adarms_cond])
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    error = jnp.square(v_t - u_t)
    return error, noise, time


def _array_sha(array) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--flow-seed-base", type=int, default=1000)
    parser.add_argument("--infer-seed-base", type=int, default=2000)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    view = args.train_view.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse {output}")
    run_config = json.loads(args.run_config.read_text(encoding="utf-8"))
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    state_mode = str(run_config.get("state_mode", "off"))
    with np.load(view / "episodes/episode_000/episode.npz", allow_pickle=False) as data:
        states = np.asarray(data["state"], dtype=np.float32)
        actions_all = np.asarray(data["action"], dtype=np.float32)
    meta = json.loads((view / "meta.json").read_text(encoding="utf-8"))
    feature_names = meta["feature_names"]
    starts = np.linspace(0, len(states) - 10, int(args.windows), dtype=int).tolist()
    windows = []
    for index, start in enumerate(starts):
        demo = actions_all[start:start + 10].copy()
        if demo.shape != (10, 12):
            raise SystemExit(f"window {index} has bad shape {demo.shape}")
        windows.append({"index": index, "start": int(start), "end": int(start) + 9,
                        "demo_action_sha256": _array_sha(demo)})
    config = openpi_so101.make_train_config(
        mode="serve", run_root=output, train_view=view,
        base_checkpoint=checkpoint, exp_name="offline_diag", state_mode=state_mode)
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)
    policy = _policy_config.create_trained_policy(
        config, checkpoint, norm_stats=norm_stats, default_prompt=TASK)
    model = policy._model
    graphdef, model_state = nnx.split(model)

    @jax.jit
    def flow_jit(state, observation, actions, rng):
        merged = nnx.merge(graphdef, state)
        return _flow_forward(merged, observation, actions, rng)

    @jax.jit
    def loss_jit(state, observation, actions, rng):
        merged = nnx.merge(graphdef, state)
        return merged.compute_loss(rng, observation, actions, train=False)

    flow_records, infer_records = [], []
    model_loss_sanity = None
    for index, start in enumerate(starts):
        demo = actions_all[start:start + 10].copy()
        raw_flow = _raw(view, start, demo)
        inputs = policy._input_transform(copy.deepcopy(raw_flow))
        inputs = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        actions = jnp.asarray(inputs["actions"], dtype=jnp.float32)
        flow_rng = jax.random.key(int(args.flow_seed_base) + index)
        error, noise, time = flow_jit(model_state, observation, actions, flow_rng)
        error = np.asarray(error)[0]
        true12 = float(error[:, :12].mean())
        padding20 = float(error[:, 12:].mean())
        all32 = float(error.mean())
        if index == 0:
            model_loss_sanity = float(np.asarray(
                loss_jit(model_state, observation, actions, flow_rng)).mean())
        raw_infer = _raw(view, start)
        infer_noise = np.random.default_rng(int(args.infer_seed_base) + index).standard_normal(
            (10, 32)).astype(np.float32)
        policy._rng = jax.random.key(0)
        inferred = np.asarray(policy.infer(copy.deepcopy(raw_infer), noise=infer_noise)["actions"],
                              dtype=np.float32)
        if inferred.shape != (10, 12):
            raise SystemExit(f"inference shape {inferred.shape}")
        window_error = np.abs(inferred - demo)
        flow_records.append({
            "window": index, "start": int(start),
            "true12_mse": true12, "padding20_mse": padding20, "all32_mse": all32,
            "model_compute_loss_sanity": model_loss_sanity if index == 0 else None,
            "flow_rng_seed": int(args.flow_seed_base) + index,
            "noise_sha256": _array_sha(noise), "time": np.asarray(time).tolist(),
        })
        infer_records.append({
            "window": index, "start": int(start),
            "inference_noise_sha256": _array_sha(infer_noise),
            "predicted_action_sha256": _array_sha(inferred),
            "demo_action": demo.tolist(), "predicted_action": inferred.tolist(),
            "mae_per_dim": window_error.mean(axis=0).tolist(),
            "mae_joint_deg": float(window_error[:, JOINT_INDICES].mean()),
            "mae_gripper_pct": float(window_error[:, GRIPPER_INDICES].mean()),
            "short_term_change_abs_error": float(np.abs(
                (inferred[-1] - inferred[0]) - (demo[-1] - demo[0])).mean()),
        })
    def mean(values):
        return float(np.mean(values)) if values else None
    aggregate = {
        "windows": len(starts),
        "flow_true12_mse_mean": mean([row["true12_mse"] for row in flow_records]),
        "flow_padding20_mse_mean": mean([row["padding20_mse"] for row in flow_records]),
        "flow_all32_mse_mean": mean([row["all32_mse"] for row in flow_records]),
        "model_compute_loss_sanity": model_loss_sanity,
        "model_loss_matches_all32": (model_loss_sanity is not None
                                     and abs(model_loss_sanity
                                             - flow_records[0]["all32_mse"]) <= 2e-6),
        "flow_u_target": "u = noise - actions; padding actions are zero, so the 20 padded targets are noise",
        "flow_dimension_note": "12 real / 20 padded are reported separately; the 20/32 dimension ratio is not a loss or gradient contribution fraction",
        "mae_joint_deg_mean": mean([row["mae_joint_deg"] for row in infer_records]),
        "mae_gripper_pct_mean": mean([row["mae_gripper_pct"] for row in infer_records]),
        "mae_per_dim_mean": np.mean([row["mae_per_dim"] for row in infer_records],
                                    axis=0).tolist(),
        "short_term_change_abs_error_mean": mean(
            [row["short_term_change_abs_error"] for row in infer_records]),
        "feature_names": feature_names,
        "joint_indices": JOINT_INDICES, "gripper_indices": GRIPPER_INDICES,
    }
    report = {
        "schema_version": 1,
        "phase": "E3.1",
        "substage": "G3-offline",
        "label": args.label,
        "state_mode": state_mode,
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "run_config": str(args.run_config),
        "run_config_sha256": hashlib.sha256(args.run_config.read_bytes()).hexdigest(),
        "binding": binding,
        "train_state_step": binding.get("train_state_step"),
        "parameters_sha256": binding.get("parameters_sha256"),
        "train_view": str(view),
        "train_view_sha256": run_config.get("view_sha256"),
        "norm_stats_canonical_sha256": run_config.get("norm_stats_canonical_sha256"),
        "flow_seed_base": args.flow_seed_base,
        "infer_seed_base": args.infer_seed_base,
        "windows": windows,
        "aggregate": aggregate,
        "per_window": {"flow": flow_records, "inference": infer_records},
        "claim": "fixed-window training-scene fit diagnostic; not a generalization success rate",
    }
    output.mkdir(parents=True)
    (output / "offline.json").write_text(json.dumps(report, indent=2, default=str) + "\n",
                                          encoding="utf-8")
    print(json.dumps({"label": args.label, "train_state_step": report["train_state_step"],
                      "flow_true12": aggregate["flow_true12_mse_mean"],
                      "flow_padding20": aggregate["flow_padding20_mse_mean"],
                      "flow_all32": aggregate["flow_all32_mse_mean"],
                      "model_loss_sanity": aggregate["model_compute_loss_sanity"],
                      "model_loss_matches_all32": aggregate["model_loss_matches_all32"],
                      "joint_mae_deg": aggregate["mae_joint_deg_mean"],
                      "gripper_mae_pct": aggregate["mae_gripper_pct_mean"],
                      "short_term": aggregate["short_term_change_abs_error_mean"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
