#!/usr/bin/env python3
"""CPU audit of deterministic step-2 batch replay before and after resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scene1_openpi_so101 as openpi_so101  # noqa: E402
import scene1_pifast_ki_runtime as runtime  # noqa: E402


def sha256_tree(tree) -> dict:
    leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    result = {}
    for path, value in leaves:
        array = np.asarray(jax.device_get(value))
        key = jax.tree_util.keystr(path)
        result[key] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    return result


def batch_digest(batch) -> dict:
    observation, actions = batch
    return {
        "observation": sha256_tree(observation),
        "actions": sha256_tree(actions),
    }


def make_indices(length: int, seed: int, count: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.choice(
        length, size=count, replace=True).astype(np.int64).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--fast-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--paligemma-tokenizer", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["SO101_TRAIN_VIEW"] = str(args.train_view)
    config = runtime.make_train_config(
        mode="probe",
        run_root=args.output.parent,
        train_view=args.train_view,
        base_checkpoint=args.base_checkpoint,
        exp_name="batch-audit",
        steps=1,
        batch_size=args.batch_size,
        save_interval=1,
        warmup_steps=1000,
        peak_lr=2.5e-5,
        decay_steps=40000,
        decay_lr=2.5e-6,
        seed=args.seed,
        fsdp_devices=1,
        num_workers=args.workers,
        fast_ce_weight=0.018,
        fast_tokenizer_dir=args.fast_tokenizer_dir,
        paligemma_tokenizer_path=args.paligemma_tokenizer,
        dataset_label="scene1_single_random_v1",
    )
    raw = openpi_so101.SO101EpisodeDataset(
        args.train_view,
        split="train",
        action_horizon=config.model.action_horizon,
    )
    all_indices = make_indices(
        len(raw), args.seed, args.batch_size * 2)
    loader_a = openpi_so101.create_so101_data_loader(
        config,
        shuffle=False,
        num_batches=2,
        split="train",
        sampler_indices=all_indices,
    )
    iterator_a = iter(loader_a)
    first = next(iterator_a)
    second = next(iterator_a)
    loader_b = openpi_so101.create_so101_data_loader(
        config,
        shuffle=False,
        num_batches=1,
        split="train",
        sampler_indices=all_indices[args.batch_size:],
    )
    resumed_first = next(iter(loader_b))
    second_digest = batch_digest(second)
    resumed_digest = batch_digest(resumed_first)
    same = second_digest == resumed_digest
    train_rng, _ = jax.random.split(jax.random.key(args.seed))
    step1_rng = jax.random.fold_in(train_rng, 1)
    payload = {
        "schema_version": 1,
        "status": "pass" if same else "fail",
        "batch_size": args.batch_size,
        "workers": args.workers,
        "seed": args.seed,
        "sample_indices_sha256": hashlib.sha256(
            json.dumps(all_indices, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "second_batch": second_digest,
        "resume_first_batch": resumed_digest,
        "batches_bitwise_equal": same,
        "resume_step": 1,
        "train_rng_sha256": hashlib.sha256(
            np.asarray(jax.random.key_data(step1_rng)).tobytes()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "batches_bitwise_equal": same,
        "batch_size": args.batch_size,
        "workers": args.workers,
    }, sort_keys=True))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
