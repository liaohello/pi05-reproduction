#!/usr/bin/env python3
"""Scene1 root-cause diagnostic harness.

    *****  NOT EXECUTED BY THE AUDIT — written on a host that cannot fork.  *****

This file exists so that every BLOCKED cell in ../*.md has a command behind it. It has
never been run. Treat it as reviewed-but-untested code: run `--self-check` first, which
only loads the checkpoint and prints tree statistics, and diff `_per_dim_loss` against
`compute_loss` in `openpi/models/pi0.py` before trusting any loss number it prints.

Design constraints honoured:
  * read-only: never writes a checkpoint, never resumes training, never edits config;
  * both weight trees are evaluated from the same restored TrainState;
  * every reported loss is split into real-12 / pad-20 / total-32, because the 32-dim
    mean alone is what allowed "0.0087 looks fine" to survive.

Subcommands:
  self-check   load the checkpoint, verify both trees, print parameter deltas
  ema          live vs EMA: losses, action MAE, paired bootstrap
  padding      per-dim losses + gradient norms/cosine by parameter group
  condition    prompt swap at fixed observation; arm-selection metrics
  phase        per-episode, per-horizon-step error, segmented by gripper transition

Requires: the pinned runtime at /opt/openpi (or $OPENPI_SOURCE_ROOT), a GPU, and
scripts/so101 on sys.path. Invoke from the repository root.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SO101 = REPO / "scripts" / "so101"
sys.path.insert(0, str(SO101))

REAL_DIM = 12
MODEL_DIM = 32
PARAM_GROUPS = {
    "action_expert": r".*llm.*_1.*",
    "projection": r".*(action_out_proj|action_in_proj).*",
    "language_lora": r".*lora.*",
}


# --------------------------------------------------------------------------- loading
def _load_train_module():
    candidates = []
    env = __import__("os").environ.get("OPENPI_SOURCE_ROOT")
    if env:
        candidates.append(Path(env) / "scripts/train.py")
    candidates.append(Path("/opt/openpi/scripts/train.py"))
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("openpi_train_entry", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("pinned OpenPI scripts/train.py not found")


def _split_ckpt(path: Path, step: int | None) -> tuple[Path, int | None]:
    """Accept either <root> or <root>/<step>."""
    path = path.expanduser().resolve()
    if path.name.isdigit() and (path.parent / str(path.name)).is_dir():
        return path.parent, step if step is not None else int(path.name)
    return path, step


def load_state(args):
    import jax
    import openpi.training.checkpoints as _ckpt
    import openpi.training.sharding as _sharding
    import openpi_so101
    from model_scope import resolve_model_scope

    root, step = _split_ckpt(Path(args.checkpoint), args.step)
    saved = json.loads((root.parent / "config_initial.json").read_text(encoding="utf-8"))
    scope = resolve_model_scope(saved)
    config = openpi_so101.make_train_config(
        mode="serve", run_root=root.parent, train_view=Path(args.train_view).resolve(),
        base_checkpoint=Path(args.base_checkpoint or "/base"),
        exp_name="audit", state_mode=saved.get("state_mode", "on"), **scope,
    )
    train = _load_train_module()
    mesh = _sharding.make_mesh(1)
    template, _ = train.init_train_state(config, jax.random.key(0), mesh, resume=True)
    manager, _ = _ckpt.initialize_checkpoint_dir(root, keep_period=1, overwrite=False, resume=True)
    state = _ckpt.restore_state(manager, template, None, step=step)
    return config, state, root, int(np.asarray(state.step))


def pick_tree(state, which: str):
    import flax.nnx as nnx
    if which == "ema":
        if state.ema_params is None:
            raise ValueError("checkpoint has no ema_params")
        return nnx.merge(state.model_def, state.ema_params)
    return nnx.merge(state.model_def, state.params)


# --------------------------------------------------------------------------- transforms
def build_pipeline(config, checkpoint_root: Path):
    import openpi.training.checkpoints as _ckpt
    import openpi.transforms as _t

    norm = _ckpt.load_norm_stats(checkpoint_root / "assets", config.data.asset_id)
    dc = config.data.create(config.assets_dirs, config.model)
    steps = [*dc.repack_transforms.inputs, *dc.data_transforms.inputs,
             _t.Normalize(norm, use_quantiles=dc.use_quantile_norm),
             *dc.model_transforms.inputs]
    return steps, norm


def apply_pipeline(steps, sample: dict):
    import openpi.models.model as _model
    data = dict(sample)
    for t in steps:
        data = t(data)
    return _model.Observation.from_dict(data), data["actions"]


def batch(samples, key_slice=None):
    import jax
    obs = [s[0] for s in samples]
    act = [s[1] for s in samples]
    stacked_obs = jax.tree.map(lambda *x: np.stack(x), *obs)
    stacked_act = np.stack(act)
    if key_slice is not None:
        stacked_act = stacked_act[:, :, key_slice]
    return stacked_obs, stacked_act


# --------------------------------------------------------------------------- loss maths
def _per_dim_loss(model, rng, observation, actions):
    """1:1 copy of Pi0.compute_loss (pi0.py:189-214) returning (b, ah, ad) instead of a mean.

    VERIFY against upstream before use; this duplication is the only place the harness
    departs from the production loss.
    """
    import einops
    import jax
    import jax.numpy as jnp
    import openpi.models.model as _model
    from openpi.models.pi0 import make_attn_mask

    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=False)
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    te = time[..., None, None]
    x_t = te * noise + (1 - te) * actions
    u_t = noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions,
        adarms_cond=[None, adarms_cond])
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    return jnp.square(v_t - u_t), noise, time   # (b, ah, ad)


def _split_losses(sq):
    real = float(np.asarray(sq[..., :REAL_DIM].mean()))
    pad = float(np.asarray(sq[..., REAL_DIM:].mean()))
    total = float(np.asarray(sq.mean()))
    return {"loss_real_12d": real, "loss_pad_20d": pad, "loss_total_32d": total,
            "identity_check_32d": abs(total - (real * REAL_DIM + pad * (MODEL_DIM - REAL_DIM)) / MODEL_DIM)}


def _paired_bootstrap(a, b, n=1000, seed=42):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0:
        return None
    rng = np.random.default_rng(seed)
    d = [float(np.mean(rng.choice(b - a, size=a.size, replace=True))) for _ in range(n)]
    return {"delta_mean": float(np.mean(b - a)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "n": int(a.size)}


# --------------------------------------------------------------------------- subcommands
def cmd_self_check(args):
    import flax.nnx as nnx
    import jax
    config, state, root, step = load_state(args)
    out = {"checkpoint_root": str(root), "train_state_step": step,
           "has_ema": state.ema_params is not None,
           "ema_decay": float(state.ema_decay) if state.ema_decay is not None else None}
    if state.ema_params is not None:
        p = state.params.to_pure_dict()
        e = state.ema_params.to_pure_dict()
        flat_p = jax.tree.leaves(p)
        flat_e = jax.tree.leaves(e)
        num = sum(float(np.sum((np.asarray(x, np.float32) - np.asarray(y, np.float32)) ** 2))
                  for x, y in zip(flat_p, flat_e))
        den = sum(float(np.sum(np.asarray(y, np.float32) ** 2)) for y in flat_e) + 1e-12
        out["relative_l2_live_vs_ema"] = math.sqrt(num / den)
        out["tensors"] = len(flat_p)
    print(json.dumps(out, indent=2))


def _sample_windows(args, config, root, count):
    import openpi_so101
    ds = openpi_so101.SO101EpisodeDataset(
        Path(args.train_view), split=args.split, action_horizon=config.model.action_horizon)
    steps, _ = build_pipeline(config, root)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(count, len(ds)), replace=False)
    return [apply_pipeline(steps, ds[int(i)]) for i in idx]


def cmd_ema(args):
    import jax
    config, state, root, step = load_state(args)
    windows = _sample_windows(args, config, root, args.samples)
    obs, act = batch(windows)
    key = jax.random.key(args.seed)
    rows = {}
    for which in ("current", "ema"):
        model = pick_tree(state, which)
        model.eval()
        sq, _, _ = _per_dim_loss(model, key, obs, act)
        sq = np.asarray(jax.device_get(sq))
        row = _split_losses(sq)
        row["per_window_real_12d"] = sq[..., :REAL_DIM].mean(axis=(1, 2)).tolist()
        rows[which] = row
    out = {"train_state_step": step, "samples": len(windows), "seed": args.seed,
           "noise_shared_across_trees": True, "metrics": rows,
           "bootstrap_real_12d": _paired_bootstrap(
               rows["current"]["per_window_real_12d"], rows["ema"]["per_window_real_12d"],
               seed=args.seed)}
    for r in rows.values():
        r.pop("per_window_real_12d", None)
    _emit(out, args.out)


def cmd_padding(args):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    config, state, root, step = load_state(args)
    windows = _sample_windows(args, config, root, args.batch)
    obs, act = batch(windows)
    model = pick_tree(state, args.tree)
    model.eval()
    key = jax.random.key(args.seed)

    def loss_parts(m, which):
        sq, _, _ = _per_dim_loss(m, key, obs, act)
        if which == "real":
            return sq[..., :REAL_DIM].mean()
        if which == "pad":
            return sq[..., REAL_DIM:].mean()
        return sq.mean()

    grads = {}
    for which in ("real", "pad", "total"):
        g = nnx.value_and_grad(lambda m, w=which: loss_parts(m, w))(model)
        grads[which] = g[1]

    def flat_norm(g):
        st = _pure(g)
        out = {}
        for name, pat in PARAM_GROUPS.items():
            acc = []
            for path, leaf in _flatten(st):
                if re.fullmatch(pat, path):
                    acc.append(float(jnp.sum(jnp.square(leaf))))
            out[name] = math.sqrt(sum(acc))
        return out

    def inner(a, b):
        st_a = _pure(a)
        st_b = _pure(b)
        num = den_a = den_b = 0.0
        for (pa, la), (_, lb) in zip(_flatten(st_a), _flatten(st_b)):
            if re.fullmatch(PARAM_GROUPS["action_expert"], pa):
                num += float(jnp.sum(la * lb))
                den_a += float(jnp.sum(la * la))
                den_b += float(jnp.sum(lb * lb))
        return num / (math.sqrt(den_a * den_b) + 1e-12)

    sq, _, _ = _per_dim_loss(model, key, obs, act)
    out = {"train_state_step": step, "tree": args.tree, "batch": len(windows), "seed": args.seed,
           "losses": _split_losses(np.asarray(jax.device_get(sq))),
           "grad_norm_by_group": {k: flat_norm(v) for k, v in grads.items()},
           "cos_real_pad_action_expert": inner(grads["real"], grads["pad"])}
    _emit(out, args.out)


def cmd_condition(args):
    import jax
    import jax.numpy as jnp

    config, state, root, step = load_state(args)
    steps, norm = build_pipeline(config, root)
    import openpi.policies.policy as _policy
    import openpi.transforms as _t
    import openpi_so101

    ds = openpi_so101.SO101EpisodeDataset(
        Path(args.train_view), split="validation", action_horizon=config.model.action_horizon)
    meta = ds.meta
    names = meta.get("feature_names", [])
    left = slice(*args.left_slice)
    right = slice(*args.right_slice)

    results = []
    for tree in args.tree.split(","):
        model = pick_tree(state, tree)
        model.eval()
        for ep in [int(x) for x in args.episodes.split(",")]:
            for frame in _probe_frames(ds, ep, args.pairs):
                base = ds[ds.samples.index((ep, frame))]
                for colour in ("red", "blue"):
                    prompt = f"Put the {colour} block into the {colour} tray."
                    sample = dict(base)
                    sample["task"] = prompt
                    sample["prompt"] = prompt
                    obs, _ = apply_pipeline(steps, sample)
                    obs = jax.tree.map(lambda x: jnp.asarray(x)[None], obs)
                    acts = model.sample_actions(jax.random.key(args.seed + frame), obs)
                    a = np.asarray(jax.device_get(acts))[0]
                    raw = a * np.asarray(norm["actions"].std)[:a.shape[-1]] \
                        + np.asarray(norm["actions"].mean)[:a.shape[-1]]
                    results.append({
                        "tree": tree, "episode": ep, "frame": frame, "prompt_colour": colour,
                        "left_motion": float(np.abs(np.diff(raw[:, left], axis=0)).sum()),
                        "right_motion": float(np.abs(np.diff(raw[:, right], axis=0)).sum()),
                        "chosen_arm": "left" if np.abs(np.diff(raw[:, left], axis=0)).sum()
                        >= np.abs(np.diff(raw[:, right], axis=0)).sum() else "right",
                        "gripper_delta": float(raw[-1, left][-1] - raw[0, left][-1]),
                        "action_sha256": _sha(a),
                    })
    flips = {}
    for r in results:
        key = (r["tree"], r["episode"], r["frame"])
        flips.setdefault(key, set()).add(r["chosen_arm"])
    summary = {"probes": len(results), "prompt_sensitive": sum(len(v) > 1 for v in flips.values()),
               "total_pairs": len(flips)}
    _emit({"train_state_step": step, "feature_names": names, "summary": summary,
           "rows": results}, args.out)


def cmd_phase(args):
    import jax
    import jax.numpy as jnp

    config, state, root, step = load_state(args)
    steps, _ = build_pipeline(config, root)
    import openpi_so101

    ds = openpi_so101.SO101EpisodeDataset(
        Path(args.train_view), split="validation", action_horizon=config.model.action_horizon)
    model = pick_tree(state, args.tree)
    model.eval()
    out = {"train_state_step": step, "tree": args.tree, "episodes": {}}
    for ep in [int(x) for x in args.episodes.split(",")]:
        if ep not in ds.episodes:
            out["episodes"][str(ep)] = {"status": "not in validation split"}
            continue
        arr = ds.episodes[ep]["arrays"]
        grip = _gripper_signal(arr["action"])
        close = int(np.argmax(grip < 0.5 * np.max(grip))) if grip.size else -1
        rows = []
        span = range(0, max(1, arr["action"].shape[0] - config.model.action_horizon), max(1, args.stride))
        for frame in span:
            sample = ds[ds.samples.index((ep, int(frame)))] if (ep, int(frame)) in ds.samples else None
            if sample is None:
                continue
            obs, expert = apply_pipeline(steps, sample)
            obs = jax.tree.map(lambda x: jnp.asarray(x)[None], obs)
            pred = model.sample_actions(jax.random.key(args.seed + int(frame)), obs)
            err = np.abs(np.asarray(jax.device_get(pred))[0] - np.asarray(expert))
            rows.append({"frame": int(frame),
                         "phase": "pre_grasp" if frame < close else "post_grasp",
                         "mae_per_horizon": err.mean(axis=1).tolist()})
        out["episodes"][str(ep)] = {"gripper_close_frame": close, "frames": len(rows),
                                    "tcp_error": "UNAVAILABLE: forward kinematics not exposed "
                                                 "by the training view; run inside the simulator",
                                    "rows": rows}
    _emit(out, args.out)


# --------------------------------------------------------------------------- helpers
def _pure(obj):
    """Accept an nnx.State, a Module, or a plain dict and return a nested dict."""
    import flax.nnx as nnx
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_pure_dict"):
        return obj.to_pure_dict()
    return nnx.state(obj).to_pure_dict()


def _flatten(tree, prefix=""):
    for k, v in tree.items():
        path = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            yield from _flatten(v, path)
        else:
            yield path, v


def _gripper_signal(actions):
    a = np.asarray(actions, float)
    return np.abs(a[:, 5]) + np.abs(a[:, 11])


def _probe_frames(ds, episode, count):
    frames = [f for (e, f) in ds.samples if e == episode]
    if not frames:
        return []
    pick = np.linspace(0, len(frames) - 1, num=min(count, len(frames)), dtype=int)
    return [frames[i] for i in pick]


def _sha(arr):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(arr, np.float32).tobytes()).hexdigest()


def _emit(obj, path):
    text = json.dumps(obj, indent=2, default=str)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("self-check", "ema", "padding", "condition", "phase"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-view", required=True)
    p.add_argument("--base-checkpoint", default="/base")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--split", default="validation")
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tree", default="current")
    p.add_argument("--episodes", default="62,13,11,2")
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--left-slice", nargs=2, type=int, default=[0, 6])
    p.add_argument("--right-slice", nargs=2, type=int, default=[6, 12])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    {"self-check": cmd_self_check, "ema": cmd_ema, "padding": cmd_padding,
     "condition": cmd_condition, "phase": cmd_phase}[args.command](args)


if __name__ == "__main__":
    main()
