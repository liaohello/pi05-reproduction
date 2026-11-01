#!/usr/bin/env python3
"""Probe tokenizer and Pi0.5 VLM prefix embeddings on fixed Scene1 frames."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import scene1_openpi_so101 as sx


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in Path(root).rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        file_digest = hashlib.sha256(path.read_bytes()).digest()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_digest)
    return digest.hexdigest()


def array_stats(value) -> dict:
    array = np.asarray(value)
    numeric = array.astype(np.float32) if array.dtype.kind not in "biu" else array
    flat = numeric.reshape(-1)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(np.mean(numeric)),
        "std": float(np.std(numeric)),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
        "l2_norm": float(np.linalg.norm(flat)),
        "head8": np.asarray(numeric.reshape(-1)[:8], dtype=float).tolist(),
    }


def cosine(left, right) -> float:
    import jax

    left = np.asarray(jax.device_get(left), dtype=np.float64).reshape(-1)
    right = np.asarray(jax.device_get(right), dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="directory containing params/, typically base or an Orbax step dir")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.models import pi0
    from openpi.training import sharding

    if sha256_tree(args.checkpoint) != args.expected_checkpoint_tree_sha256:
        raise ValueError("checkpoint tree SHA256 mismatch")
    config = sx.make_train_config(
        mode="serve",
        run_root=args.out.parent / "vlm_probe_run",
        train_view=args.train_view,
        base_checkpoint=args.checkpoint,
        exp_name="scene1_vlm_probe",
        state_mode="on",
        action_horizon=args.action_horizon,
        batch_size=1,
        fsdp_devices=1,
        seed=args.seed,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
        freeze_vision_encoder=True,
        dtype="bfloat16",
    )
    spec = importlib.util.spec_from_file_location(
        "openpi_train_entry", "/opt/openpi/scripts/train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    mesh = sharding.make_mesh(1)
    state, _ = train.init_train_state(
        config,
        jax.random.split(jax.random.key(args.seed))[1],
        mesh,
        resume=False,
    )
    state = jax.block_until_ready(state)
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    dataset = sx.SO101EpisodeDataset(
        args.train_view, split="train", action_horizon=args.action_horizon)
    sample_index = next(
        index for index, row in enumerate(dataset.sample_metadata)
        if "red" in dataset.episodes[row["episode_id"]]["task"].lower()
    )
    raw_sample = dataset[sample_index]
    prompts = {
        "red": "Put the red block into the red tray.",
        "blue": "Put the blue block into the blue tray.",
    }
    data_config = config.data.create(config.assets_dirs, config.model)
    preprocess = transforms.compose([
        *data_config.data_transforms.inputs,
        transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    tokenizer = next(
        transform.tokenizer for transform in data_config.model_transforms.inputs
        if type(transform).__name__ == "TokenizePrompt"
    )
    outputs = {}
    arrays = {}
    for color, prompt in prompts.items():
        sample = dict(raw_sample)
        sample["task"] = prompt
        processed = preprocess(sample)
        observation = model_lib.Observation.from_dict(
            jax.tree.map(lambda value: jnp.asarray(value)[None], processed))
        image_tokens = {}
        for name, image in observation.images.items():
            tokens, _ = model.PaliGemma.img(image, train=False)
            image_tokens[name] = tokens
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        attention_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_outputs, _ = model.PaliGemma.llm(
            [prefix_tokens, None], mask=attention_mask, positions=positions)
        prefix_out = prefix_outputs[0]
        valid = np.asarray(jax.device_get(prefix_mask[0]))
        image_length = sum(
            int(value.shape[1]) for value in image_tokens.values())
        active_ids = np.asarray(
            jax.device_get(observation.tokenized_prompt[0])
        )[np.asarray(jax.device_get(observation.tokenized_prompt_mask[0]))]
        outputs[color] = {
            "prompt": prompt,
            "token_count": int(active_ids.size),
            "token_ids_active": active_ids.tolist(),
            "decoded_active": tokenizer._tokenizer.decode(active_ids.tolist()),
            "image_token_stats": {
                name: array_stats(jax.device_get(value[0]))
                for name, value in image_tokens.items()
            },
            "prefix_embedding_stats": array_stats(jax.device_get(prefix_out[0])),
            "image_token_count": int(image_length),
            "valid_prefix_tokens": int(valid.sum()),
            "padded_prefix_tokens": int((~valid).sum()),
            "language_context_stats": array_stats(
                jax.device_get(prefix_out[0, image_length:])),
            "language_context_head8": np.asarray(
                jax.device_get(prefix_out[0, image_length]), dtype=float
            )[:8].tolist(),
        }
        arrays[f"{color}_prefix"] = np.asarray(jax.device_get(prefix_out[0]))
        for name, value in image_tokens.items():
            arrays[f"{color}_{name}_tokens"] = np.asarray(jax.device_get(value[0]))
    red_language = arrays["red_prefix"][-200:][:outputs["red"]["token_count"]]
    blue_language = arrays["blue_prefix"][-200:][:outputs["blue"]["token_count"]]
    if red_language.shape != blue_language.shape:
        raise ValueError("red/blue language context shapes differ")
    outputs["red_vs_blue_same_observation"] = {
        "language_cosine": cosine(red_language, blue_language),
        "language_max_abs": float(np.max(np.abs(
            np.asarray(red_language, np.float32)
            - np.asarray(blue_language, np.float32)))),
        "language_mean_abs": float(np.mean(np.abs(
            np.asarray(red_language, np.float32)
            - np.asarray(blue_language, np.float32)))),
        "changed_prompt_token_ids": [
            int(value) for value in outputs["blue"]["token_ids_active"]
            if value not in outputs["red"]["token_ids_active"]
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = args.out.with_suffix(".npz")
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_tree_sha256": sha256_tree(args.checkpoint),
        "view_sha256": dataset.meta["view_sha256"],
        "sample": {
            "episode_id": int(dataset.samples[sample_index][0]),
            "frame": int(dataset.samples[sample_index][1]),
            "raw_state": np.asarray(
                raw_sample["observation.state"], dtype=float).tolist(),
        },
        "state_mode": "on",
        "action_horizon": args.action_horizon,
        "outputs": outputs,
        "arrays": {
            "path": str(arrays_path),
            "sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
            "bytes": arrays_path.stat().st_size,
        },
    }
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "checkpoint_tree_sha256": payload["checkpoint_tree_sha256"],
        "sample": payload["sample"],
        "red": outputs["red"],
        "blue": outputs["blue"],
        "red_vs_blue_same_observation": outputs["red_vs_blue_same_observation"],
        "arrays": payload["arrays"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
