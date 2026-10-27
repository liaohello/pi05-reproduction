#!/usr/bin/env python3
"""E3.1 / G2: verify whether Pi05 consumes the SO101 state.

Runs inside the pinned OpenPI image.  Fixed images, task and sampling noise;
only the legal 12-D state changes.  The state-off policy must be invariant
(negative control).  The state-on policy uses the existing discrete-state
token path and must show token and fixed-noise output changes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import openpi_so101  # noqa: E402
import openpi.training.checkpoints as _checkpoints  # noqa: E402
from openpi import transforms as _transforms  # noqa: E402
from openpi.models import tokenizer as _tokenizer  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402

TASK = openpi_so101.TASK


def _raw_observation(view: Path, state: np.ndarray) -> dict:
    episode = view / "episodes" / "episode_000"
    def image(name):
        from PIL import Image
        with Image.open(episode / name / "000000.jpg") as img:
            return np.asarray(img.convert("RGB"))
    return {
        "observation.state": np.asarray(state, dtype=np.float32),
        "observation.images.central": image("central"),
        "observation.images.left_wrist": image("left_wrist"),
        "observation.images.right_wrist": image("right_wrist"),
        "prompt": TASK,
    }


def _token_report(config, norm_stats, raw: dict, state_mode: str) -> dict:
    inputs = openpi_so101.SO101Inputs()(copy.deepcopy(raw))
    normalized = _transforms.Normalize(norm_stats, use_quantiles=True)(inputs)
    state = np.asarray(normalized["state"], dtype=np.float64)
    tokenizer = _tokenizer.PaligemmaTokenizer(config.model.max_token_len)
    tokens, mask = tokenizer.tokenize(TASK, state if state_mode == "on" else None)
    valid = int(np.asarray(mask, dtype=bool).sum())
    decoded = tokenizer._tokenizer.decode(np.asarray(tokens[:valid]).tolist())
    bins = np.linspace(-1, 1, 256 + 1)[:-1]
    quantized = np.digitize(state, bins=bins) - 1
    return {
        "tokens": tokens,
        "mask": np.asarray(mask, dtype=bool),
        "valid_tokens": valid,
        "max_tokens": int(config.model.max_token_len),
        "truncated": valid >= int(config.model.max_token_len),
        "decoded_prefix": decoded,
        "state_normalized": state.tolist(),
        "state_quantized": quantized.tolist(),
        "state_outside_minus1_plus1": int(np.sum((state < -1.0) | (state > 1.0))),
        "state_saturated_low": int(np.sum(state < -1.0)),
        "state_saturated_high": int(np.sum(state > 1.0)),
    }


def _policy(config, checkpoint: Path, norm_stats):
    return _policy_config.create_trained_policy(
        config, checkpoint, norm_stats=norm_stats, default_prompt=TASK)


def _infer(policy, raw: dict, noise: np.ndarray) -> np.ndarray:
    policy._rng = jax.random.key(0)
    result = policy.infer(copy.deepcopy(raw), noise=noise)
    return np.asarray(result["actions"], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs=2, default=[0, 100])
    parser.add_argument("--noise-seed", type=int, default=1234)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    view = args.train_view.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse {output}")
    with np.load(view / "episodes/episode_000/episode.npz", allow_pickle=False) as data:
        states = np.asarray(data["state"], dtype=np.float32)
    frame_a, frame_b = (int(value) for value in args.frames)
    if not (0 <= frame_a < len(states) and 0 <= frame_b < len(states)):
        raise SystemExit("frame index out of range")
    raw_a = _raw_observation(view, states[frame_a])
    raw_b = _raw_observation(view, states[frame_b])
    noise = np.random.default_rng(args.noise_seed).standard_normal((10, 32)).astype(np.float32)
    reports = {}
    for mode in ("off", "on"):
        config = openpi_so101.make_train_config(
            mode="serve", run_root=output, train_view=view,
            base_checkpoint=checkpoint, exp_name="state_check", state_mode=mode)
        data_config = config.data.create(config.assets_dirs, config.model)
        norm_stats = _checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)
        tokens_a = _token_report(config, norm_stats, raw_a, mode)
        tokens_b = _token_report(config, norm_stats, raw_b, mode)
        policy = _policy(config, checkpoint, norm_stats)
        actions_a = _infer(policy, raw_a, noise)
        actions_b = _infer(policy, raw_b, noise)
        delta = np.abs(actions_a - actions_b)
        token_diff = int(np.sum(tokens_a["tokens"] != tokens_b["tokens"]))
        quantized_changed = int(np.sum(np.asarray(tokens_a["state_quantized"])
                                       != np.asarray(tokens_b["state_quantized"])))
        normalized_delta = float(np.max(np.abs(
            np.asarray(tokens_a["state_normalized"]) - np.asarray(tokens_b["state_normalized"]))))
        def mask_ok(info):
            mask = np.asarray(info["mask"], dtype=bool)
            valid = int(info["valid_tokens"])
            return bool(mask[:valid].all() and not mask[valid:].any()
                        and int(mask.sum()) == valid)
        reports[mode] = {
            "state_mode": mode,
            "discrete_state_input": bool(config.model.discrete_state_input),
            "state_frame_indices": [frame_a, frame_b],
            "fixed_noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
            "actions_shape": list(actions_a.shape),
            "action_max_abs_difference": float(delta.max()) if delta.size else None,
            "action_mean_abs_difference": float(delta.mean()) if delta.size else None,
            "action_a_first": actions_a[0].tolist(),
            "action_b_first": actions_b[0].tolist(),
            "tokens_a": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in tokens_a.items() if k not in ("tokens", "mask")},
            "tokens_b": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in tokens_b.items() if k not in ("tokens", "mask")},
            "token_changed_positions": token_diff,
            "quantized_state_changed": quantized_changed,
            "normalized_state_max_delta": normalized_delta,
            "mask_is_valid_prefix": mask_ok(tokens_a) and mask_ok(tokens_b),
            "valid_tokens_a": int(tokens_a["valid_tokens"]),
            "valid_tokens_b": int(tokens_b["valid_tokens"]),
            "token_valid_equal": bool(tokens_a["valid_tokens"] == tokens_b["valid_tokens"]),
            "token_mask_equal": bool(np.array_equal(tokens_a["mask"], tokens_b["mask"])),
            "token_decoded_a_has_state": "State:" in tokens_a["decoded_prefix"],
            "token_decoded_b_has_state": "State:" in tokens_b["decoded_prefix"],
            "truncated": bool(tokens_a["truncated"] or tokens_b["truncated"]),
        }
        del policy
    off = reports["off"]
    on = reports["on"]
    checks = {
        "state_off_invariant": (off["action_max_abs_difference"] is not None
                                and off["action_max_abs_difference"] <= 1e-6),
        "state_off_tokens_equal": off["token_changed_positions"] == 0,
        "state_on_tokens_changed": on["token_changed_positions"] > 0,
        "state_on_output_changed": (on["action_max_abs_difference"] is not None
                                    and on["action_max_abs_difference"] > 1e-6),
        "state_on_no_truncation": not on["truncated"],
        "state_on_mask_valid": bool(on["mask_is_valid_prefix"]),
        "state_on_quantized_changed": on["quantized_state_changed"] > 0,
        "state_on_decoded_state_present": (on["token_decoded_a_has_state"]
                                            and on["token_decoded_b_has_state"]),
        "state_off_decoded_state_absent": (not off["token_decoded_a_has_state"]
                                           and not off["token_decoded_b_has_state"]),
    }
    report = {
        "schema_version": 1,
        "phase": "E3.1",
        "substage": "G2-state",
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "train_view": str(view),
        "frame_indices": [frame_a, frame_b],
        "noise_seed": args.noise_seed,
        "checks": checks,
        "modes": reports,
        "claim": "fixed-noise state sensitivity diagnostic; state-on is the discrete token path, "
                 "not a continuous state path and not a task result",
    }
    output.mkdir(parents=True)
    (output / "state_check.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "off_delta": off["action_max_abs_difference"],
                      "on_delta": on["action_max_abs_difference"],
                      "on_token_changed": on["token_changed_positions"],
                      "off_token_changed": off["token_changed_positions"],
                      "on_valid_tokens": on["tokens_a"]["valid_tokens"],
                      "on_truncated": on["truncated"]}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
