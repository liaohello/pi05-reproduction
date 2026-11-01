#!/usr/bin/env python3
"""CPU-only full-window FAST audit for the Scene1 PI-FAST plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm
from transformers import AutoProcessor

PALIGEMMA_VOCAB_SIZE = 257_152
FAST_SKIP_TOKENS = 128
FAST_TOKEN_COUNT = 2_048
FAST_TOKEN_START = 254_976
FAST_TOKEN_STOP = 257_024
PAD_ID = 0
EOS_ID = 1
EXPECTED_FEATURES = [
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
]
EXPECTED_UNITS = ["degree"] * 5 + ["percent"] + ["degree"] * 5 + ["percent"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def action_id_to_paligemma(token_id: int) -> int:
    mapped = PALIGEMMA_VOCAB_SIZE - 1 - FAST_SKIP_TOKENS - int(token_id)
    if not FAST_TOKEN_START <= mapped < FAST_TOKEN_STOP:
        raise ValueError("FAST token mapping left the reserved action range")
    return mapped


def paligemma_to_action_id(token_id: int) -> int:
    mapped = PALIGEMMA_VOCAB_SIZE - 1 - FAST_SKIP_TOKENS - int(token_id)
    if not 0 <= mapped < FAST_TOKEN_COUNT:
        raise ValueError("PaliGemma token is outside the FAST action range")
    return mapped


def normalize_quantile(values: np.ndarray, stats: dict) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    if q01.shape != (12,) or q99.shape != (12,) or np.any(q99 <= q01):
        raise ValueError("invalid per-dimension quantile statistics")
    return ((np.asarray(values, dtype=np.float64) - q01)
            / (q99 - q01 + 1e-6) * 2.0 - 1.0).astype(np.float32)


def unnormalize_quantile(values: np.ndarray, stats: dict) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    return ((np.asarray(values, dtype=np.float64) + 1.0) / 2.0
            * (q99 - q01 + 1e-6) + q01)


class FixedFastTokenizer:
    def __init__(
        self,
        fast_tokenizer_dir: Path,
        paligemma_tokenizer_path: Path,
        *,
        max_len: int,
    ):
        self.max_len = int(max_len)
        self.fast = AutoProcessor.from_pretrained(
            str(fast_tokenizer_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        self.language = spm.SentencePieceProcessor(
            model_file=str(paligemma_tokenizer_path)
        )
        if self.fast.vocab_size != FAST_TOKEN_COUNT:
            raise ValueError(f"unexpected FAST vocabulary {self.fast.vocab_size}")
        if self.language.vocab_size() != PALIGEMMA_VOCAB_SIZE:
            raise ValueError(
                f"unexpected PaliGemma vocabulary {self.language.vocab_size()}"
            )

    def encode(
        self, prompt: str, state: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        state = np.asarray(state, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if state.shape != (12,):
            raise ValueError(f"state shape is {state.shape}, expected (12,)")
        if actions.ndim != 2 or actions.shape[1] != 12:
            raise ValueError(f"action shape is {actions.shape}, expected (H, 12)")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("FAST input contains non-finite values")

        cleaned = " ".join(
            str(prompt).lower().strip().replace("_", " ").split()
        )
        discretized = np.digitize(
            state, bins=np.linspace(-1, 1, 256 + 1)[:-1]
        ) - 1
        state_text = " ".join(map(str, discretized))
        prefix = self.language.encode(
            f"Task: {cleaned}, State: {state_text};\n", add_bos=True
        )
        fast_ids = [int(value) for value in self.fast(actions[None])[0]]
        if not fast_ids:
            raise ValueError("FAST returned no action tokens")
        mapped = [action_id_to_paligemma(value) for value in fast_ids]
        postfix = (
            self.language.encode("Action: ")
            + mapped
            + self.language.encode("|", add_eos=True)
        )
        tokens = prefix + postfix
        if len(tokens) > self.max_len:
            raise ValueError(
                f"token sequence {len(tokens)} exceeds max_len {self.max_len}; "
                "truncation is forbidden"
            )
        if postfix[-1] != EOS_ID:
            raise ValueError("FAST suffix does not end in EOS")
        padded = np.asarray(tokens + [PAD_ID] * (self.max_len - len(tokens)),
                            dtype=np.int32)
        return padded, np.asarray(postfix, dtype=np.int32), len(prefix), len(fast_ids)

    def decode(
        self, action_tokens: np.ndarray, *, horizon: int
    ) -> np.ndarray:
        mapped = [
            paligemma_to_action_id(int(token_id))
            for token_id in action_tokens
            if FAST_TOKEN_START <= int(token_id) < FAST_TOKEN_STOP
        ]
        if not mapped:
            raise ValueError("FAST suffix contains no action tokens")
        decoded = self.fast.decode(
            [mapped], time_horizon=horizon, action_dim=12
        )[0]
        if decoded.shape != (horizon, 12) or not np.isfinite(decoded).all():
            raise ValueError("FAST reconstruction has the wrong shape or non-finite values")
        return np.asarray(decoded, dtype=np.float32)


def load_stats(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))["norm_stats"]
    if set(payload) != {"state", "actions"}:
        raise ValueError("norm stats must contain only state and actions groups")
    for group in payload.values():
        for field in ("mean", "std", "q01", "q99"):
            if len(group[field]) != 12:
                raise ValueError(f"norm stats {field} is not 12-dimensional")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--fast-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--paligemma-tokenizer", type=Path, required=True)
    parser.add_argument("--processor-commit", required=True)
    parser.add_argument("--max-token-len", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    meta_path = args.view / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    episodes = sorted(meta["episodes"], key=lambda row: int(row["episode_index"]))
    if len(episodes) != 64:
        raise ValueError(f"expected 64 episodes, found {len(episodes)}")
    if int(meta["split"]["family_crosses"]) != 0:
        raise ValueError("scene family crosses the train/validation split")
    if meta["feature_names"] != EXPECTED_FEATURES:
        raise ValueError("feature order differs from the Scene1 contract")
    if meta["feature_units"] != EXPECTED_UNITS:
        raise ValueError("feature units differ from degree/percent")
    if args.horizon < 1 or args.max_token_len < 2:
        raise ValueError("horizon and max token length must be positive")

    stats = load_stats(args.norm_stats)
    tokenizer = FixedFastTokenizer(
        args.fast_tokenizer_dir,
        args.paligemma_tokenizer,
        max_len=args.max_token_len,
    )
    split_windows = {"train": 0, "validation": 0}
    color_windows = {
        f"{split}/{color}": 0
        for split in ("train", "validation")
        for color in ("red", "blue")
    }
    max_abs = np.zeros(12, dtype=np.float64)
    sum_abs = np.zeros(12, dtype=np.float64)
    min_normalized = np.full(12, np.inf, dtype=np.float64)
    max_normalized = np.full(12, -np.inf, dtype=np.float64)
    sequence_lengths: list[int] = []
    fast_lengths: list[int] = []
    prefix_lengths: list[int] = []
    outside_unit = 0
    episode_rows = []
    error_chunks: list[np.ndarray] = []
    state_forward_inverse_max = 0.0
    action_forward_inverse_max = np.zeros(12, dtype=np.float64)
    padding_nonzero = 0

    def update_stats(error: np.ndarray) -> None:
        nonlocal max_abs, sum_abs
        absolute = np.abs(error.astype(np.float64))
        sum_abs += absolute.sum(axis=0)
        max_abs = np.maximum(max_abs, absolute.max(axis=0))

    for row in episodes:
        episode_path = (
            args.view / "episodes"
            / f"episode_{int(row['episode_index']):03d}" / "episode.npz"
        )
        with np.load(episode_path, allow_pickle=False) as data:
            state = np.asarray(data["state"], dtype=np.float32)
            action = np.asarray(data["action"], dtype=np.float32)
        frames = int(row["frames"])
        if state.shape != (frames, 12) or action.shape != (frames, 12):
            raise ValueError(f"episode {row['episode_index']}: bad array shape")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"episode {row['episode_index']}: non-finite data")

        episode_sequences, episode_fast, episode_prefix = [], [], []
        episode_windows = frames - args.horizon + 1
        if episode_windows < 1:
            raise ValueError(f"episode {row['episode_index']} is shorter than horizon")
        for start in range(episode_windows):
            normalized_state = normalize_quantile(state[start], stats["state"])
            normalized_action = normalize_quantile(
                action[start:start + args.horizon], stats["actions"]
            )
            state_forward_inverse_max = max(
                state_forward_inverse_max,
                float(np.max(np.abs(
                    unnormalize_quantile(
                        normalized_state, stats["state"])
                    - state[start]))),
            )
            action_forward_inverse_max = np.maximum(
                action_forward_inverse_max,
                np.max(np.abs(
                    unnormalize_quantile(
                        normalized_action, stats["actions"])
                    - action[start:start + args.horizon]), axis=0),
            )
            padded_action = np.pad(normalized_action, ((0, 0), (0, 20)))
            padding_nonzero += int(np.count_nonzero(padded_action[:, 12:]))
            outside_unit += int(
                (np.abs(normalized_state) > 1.0).sum()
                + (np.abs(normalized_action) > 1.0).sum()
            )
            min_normalized = np.minimum(
                min_normalized,
                np.minimum(normalized_state, normalized_action.min(axis=0)),
            )
            max_normalized = np.maximum(
                max_normalized,
                np.maximum(normalized_state, normalized_action.max(axis=0)),
            )
            tokens, suffix, prefix_len, fast_count = tokenizer.encode(
                row["task"], normalized_state, normalized_action
            )
            nonpadding = int(np.count_nonzero(tokens))
            mapped_count = sum(
                FAST_TOKEN_START <= int(value) < FAST_TOKEN_STOP
                for value in suffix
            )
            if mapped_count != fast_count:
                raise ValueError("mapped FAST count differs from processor output")
            if int(suffix[-1]) != EOS_ID:
                raise ValueError("EOS is not the final supervised suffix token")
            decoded = tokenizer.decode(suffix, horizon=args.horizon)
            error = decoded - normalized_action
            update_stats(error)
            error_chunks.append(error.astype(np.float32))
            episode_sequences.append(nonpadding)
            episode_fast.append(fast_count)
            episode_prefix.append(prefix_len)
            sequence_lengths.append(nonpadding)
            fast_lengths.append(fast_count)
            prefix_lengths.append(prefix_len)

        split = row["split"]
        color = row["color"]
        split_windows[split] += episode_windows
        color_windows[f"{split}/{color}"] += episode_windows
        episode_rows.append({
            "episode_index": int(row["episode_index"]),
            "split": split,
            "color": color,
            "frames": frames,
            "windows": episode_windows,
            "sequence_length_min": min(episode_sequences),
            "sequence_length_max": max(episode_sequences),
            "fast_token_count_min": min(episode_fast),
            "fast_token_count_max": max(episode_fast),
            "prefix_length_min": min(episode_prefix),
            "prefix_length_max": max(episode_prefix),
        })

    expected_windows = {"train": 35_843, "validation": 8_228}
    expected_colors = {
        "train/red": 17_857,
        "train/blue": 17_986,
        "validation/red": 4_110,
        "validation/blue": 4_118,
    }
    if split_windows != expected_windows or color_windows != expected_colors:
        raise ValueError(
            f"window counts differ: {split_windows}, {color_windows}")
    total = sum(split_windows.values())
    all_errors = np.concatenate(error_chunks, axis=0)
    per_dimension_mae = (sum_abs / (total * args.horizon)).tolist()
    absolute_errors = np.abs(all_errors.astype(np.float64))
    per_dimension_quantiles = {
        str(quantile): np.quantile(
            absolute_errors, quantile, axis=0).tolist()
        for quantile in (0.5, 0.95, 0.99, 0.999)
    }
    acceptance = {
        "per_dimension_mae_max": 0.02,
        "per_dimension_p99_max": 0.05,
        "per_dimension_max_abs_max": 0.15,
        "reason": (
            "FAST is a lossy action tokenizer; thresholds reject gross mapping or "
            "decode failures while allowing its measured quantization error"),
    }
    passed = (
        max(per_dimension_mae) <= acceptance["per_dimension_mae_max"]
        and max(per_dimension_quantiles["0.99"])
        <= acceptance["per_dimension_p99_max"]
        and max(max_abs) <= acceptance["per_dimension_max_abs_max"]
        and max(sequence_lengths) <= args.max_token_len
        and state_forward_inverse_max <= 1e-4
        and float(np.max(action_forward_inverse_max)) <= 1e-4
        and padding_nonzero == 0
    )
    payload = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "claim": "CPU full-window FAST encode/decode audit using fixed tokenizer resources",
        "backend": "cpu",
        "view_sha256": meta["view_sha256"],
        "meta_sha256": sha256_file(meta_path),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "processor": {
            "repository": "physical-intelligence/fast",
            "commit": args.processor_commit,
            "local_tokenizer_sha256": sha256_file(
                args.fast_tokenizer_dir / "tokenizer.json"),
            "paligemma_tokenizer_sha256": sha256_file(
                args.paligemma_tokenizer),
            "vocabulary_size": int(tokenizer.fast.vocab_size),
            "paligemma_vocabulary_size": int(tokenizer.language.vocab_size()),
        },
        "interface": {
            "horizon": args.horizon,
            "physical_action_dim": 12,
            "normalized_before_fast": True,
            "padding_encoded_by_fast": False,
            "max_token_len": args.max_token_len,
            "truncation_performed": False,
            "eos_required": True,
        },
        "windows": split_windows,
        "color_windows": color_windows,
        "token_lengths": {
            "min": min(sequence_lengths),
            "max": max(sequence_lengths),
            "mean": float(np.mean(sequence_lengths)),
            "limit": args.max_token_len,
            "over_limit_count": sum(
                value > args.max_token_len for value in sequence_lengths),
        },
        "fast_token_lengths": {
            "min": min(fast_lengths),
            "max": max(fast_lengths),
            "mean": float(np.mean(fast_lengths)),
        },
        "prefix_lengths": {
            "min": min(prefix_lengths),
            "max": max(prefix_lengths),
            "mean": float(np.mean(prefix_lengths)),
        },
        "normalized_action_roundtrip": {
            "samples": total * args.horizon,
            "per_dimension_mae": per_dimension_mae,
            "per_dimension_abs_error_quantiles": per_dimension_quantiles,
            "per_dimension_max_abs_error": max_abs.tolist(),
            "global_max_abs_error": float(np.max(max_abs)),
            "acceptance": acceptance,
        },
        "quantile_forward_inverse": {
            "state_max_abs_error": state_forward_inverse_max,
            "action_per_dimension_max_abs_error": (
                action_forward_inverse_max.tolist()),
            "action_global_max_abs_error": float(
                np.max(action_forward_inverse_max)),
            "tolerance_max_abs_error": 1e-4,
            "padding_to_32_dimensions": "12 real dimensions followed by 20 zeros",
            "padding_nonzero_values": padding_nonzero,
        },
        "normalized_range_observation": {
            "out_of_unit_values": outside_unit,
            "not_used_as_failure": True,
            "per_dimension_min": min_normalized.tolist(),
            "per_dimension_max": max_normalized.tolist(),
        },
        "episodes": episode_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": payload["status"],
        "windows": split_windows,
        "token_length_max": payload["token_lengths"]["max"],
        "fast_token_count_max": payload["fast_token_lengths"]["max"],
        "roundtrip_max_abs_error": payload[
            "normalized_action_roundtrip"]["global_max_abs_error"],
    }, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
