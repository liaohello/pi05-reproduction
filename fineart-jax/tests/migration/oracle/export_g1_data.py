#!/usr/bin/env python3
"""Export G1 goldens from the read-only reference implementation.

Runs under REF_PYTHON with ``PYTHONPATH=$REFERENCE_ROOT/src`` and never imports
candidate code, so a candidate defect cannot become an expected value. Each
group writes an ``.npz`` plus a JSON manifest holding shapes, dtypes and array
hashes; the tests re-export twice and only trust a group whose bytes reproduce.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tests.migration import g1_cases as cases  # noqa: E402


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def dump(path: Path, arrays: dict, manifest: dict) -> None:
    payload = {name: np.asarray(value) for name, value in arrays.items()}
    np.savez(str(path), **payload)
    record = {
        "arrays": {
            name: {
                "shape": list(payload[name].shape),
                "dtype": str(payload[name].dtype),
                "sha256": sha256_array(payload[name]),
            }
            for name in sorted(payload)
        },
        **manifest,
    }
    path.with_suffix(".json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def group_routing(out: Path, args) -> None:
    from lerobot.datasets.language_render import render_sample
    from lerobot.datasets.recipe import TrainingRecipe
    from lerobot.utils.constants import MESSAGES_RENDERED

    manifest: dict = {"cases": {}}
    arrays: dict = {}
    for case in cases.ROUTING_CASES:
        recipe = None if case["recipe"] is None else TrainingRecipe.from_dict(case["recipe"])
        per_case = []
        for sample_idx in cases.ROUTING_SAMPLE_INDICES:
            rendered = None
            messages: list[str] = []
            streams: list[str] = []
            targets: list[int] = []
            if recipe is None:
                # The reference delegates the whole PI0.5 pipeline here; the data-layer
                # contract is the low-level fallback render.
                fallback = {
                    MESSAGES_RENDERED: [{"role": "user", "content": case["task"]}],
                    "message_streams": ["low_level"],
                    "target_message_indices": [],
                }
                rendered = fallback
            else:
                rendered = render_sample(
                    recipe=recipe,
                    persistent=case["persistent"],
                    events=case["events"],
                    t=case["timestamp"],
                    sample_idx=sample_idx,
                    task=case["task"],
                    default_task=case["default_task"],
                )
                if rendered is None:
                    from lerobot.processor.render_messages_processor import _fallback_low_level_render

                    rendered = _fallback_low_level_render(case["task"])
            if rendered is not None:
                messages = [json.dumps(message, sort_keys=True) for message in rendered[MESSAGES_RENDERED]]
                streams = list(rendered["message_streams"])
                targets = list(rendered["target_message_indices"])
            per_case.append(
                {
                    "sample_idx": sample_idx,
                    "rendered": rendered is not None,
                    "messages": messages,
                    "streams": streams,
                    "targets": targets,
                }
            )
            arrays[f"{case['name']}__{sample_idx}__messages"] = np.asarray(len(messages), dtype=np.int64)
            arrays[f"{case['name']}__{sample_idx}__targets"] = np.asarray(targets, dtype=np.int64)
        manifest["cases"][case["name"]] = per_case
    dump(out / "routing.npz", arrays, manifest)


# --------------------------------------------------------------------------- #
# Prompt, tokens, labels
# --------------------------------------------------------------------------- #
def _tokenizer(args, add_eos: bool = False, add_bos: bool | None = None):
    from transformers import AutoTokenizer

    kwargs = {}
    if add_eos:
        kwargs["add_eos_token"] = True
        kwargs["add_bos_token"] = False
    if add_bos is not None and not add_eos:
        kwargs["add_bos_token"] = add_bos
    return AutoTokenizer.from_pretrained(args.text_tokenizer, trust_remote_code=True, **kwargs)


def group_prompt(out: Path, args) -> None:
    import torch
    from lerobot.lerobot_types import TransitionKey
    from lerobot.policies.fineart_vla.processor_fineart_vla import FineARTVLATextTokenizerStep
    from lerobot.utils.constants import OBS_STATE

    step = FineARTVLATextTokenizerStep(tokenizer_name=args.text_tokenizer, max_length=200)
    manifest: dict = {"cases": {}}
    arrays: dict = {}
    for case in cases.PROMPT_CASES:
        complementary = {"messages": case["messages"]}
        if case.get("runtime"):
            # A real runtime query carries no rendering metadata at all; the reference
            # only appends the "Assistant:" tail when those keys are absent.
            pass
        else:
            complementary["message_streams"] = case["streams"]
            complementary["target_message_indices"] = case["targets"]
        observation = {}
        if case.get("state") is not None:
            observation[OBS_STATE] = torch.tensor([case["state"]], dtype=torch.float32)
        transition = {
            TransitionKey.OBSERVATION: observation,
            TransitionKey.COMPLEMENTARY_DATA: complementary,
        }
        out_transition = step(transition)
        obs = out_transition[TransitionKey.OBSERVATION]
        comp = out_transition[TransitionKey.COMPLEMENTARY_DATA]
        ids = obs["observation.language.tokens"].numpy()
        mask = obs["observation.language.attention_mask"].numpy()
        labels = comp["text_labels"].numpy()
        predict = comp["predict_actions"].numpy()
        arrays[f"{case['name']}__ids"] = ids
        arrays[f"{case['name']}__mask"] = mask
        arrays[f"{case['name']}__labels"] = labels
        arrays[f"{case['name']}__predict"] = predict
        nonpad = int(mask.sum())
        manifest["cases"][case["name"]] = {
            "nonpad": nonpad,
            "n_supervised": int((labels != -100).sum()),
            "supervised_first": int(labels[labels != -100][0]) if (labels != -100).any() else None,
            "predict_actions": bool(predict.reshape(-1)[0]),
            "tail_text": _decode_tail(step, ids[0], nonpad),
        }
    dump(out / "prompt.npz", arrays, manifest)


def _decode_tail(step, ids, nonpad):
    tokenizer = step._ensure_tokenizer()
    return tokenizer.decode(ids[:nonpad].tolist())


# --------------------------------------------------------------------------- #
# State discretisation and locations
# --------------------------------------------------------------------------- #
def group_stateloc(out: Path, args) -> None:
    from lerobot.policies.fineart_vla.processor_fineart_vla import (
        _loc_token,
        _vqa_answer_to_loc,
        discretize_state_str,
    )

    tokenizer = _tokenizer(args)
    manifest: dict = {"state": [], "points": [], "boxes": []}
    arrays: dict = {}
    for index, row in enumerate(cases.STATE_BIN_CASES):
        text = discretize_state_str(np.asarray(row, dtype=np.float32))
        manifest["state"].append({"row": row, "text": text})
    for index, answer in enumerate(cases.LOC_POINT_CASES + cases.LOC_BOX_CASES):
        text = _vqa_answer_to_loc(answer)
        target = "points" if answer in cases.LOC_POINT_CASES else "boxes"
        ids = None
        if text is not None:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            arrays[f"loc_{index}__ids"] = np.asarray(ids, dtype=np.int64)
        manifest[target].append({"answer": answer, "text": text, "ids_len": None if ids is None else len(ids)})
    # Individual location tokens, including the clamped edges.
    coords = [0.0, 0.25, 0.5, 1.0, 1.5, -0.5]
    manifest["loc_tokens"] = [{"coord": c, "text": _loc_token(c)} for c in coords]
    arrays["loc_token_ids"] = np.asarray(
        [tokenizer(_loc_token(c), add_special_tokens=False)["input_ids"][0] for c in coords], dtype=np.int64
    )
    dump(out / "stateloc.npz", arrays, manifest)


# --------------------------------------------------------------------------- #
# FAST tokens
# --------------------------------------------------------------------------- #
def group_fast(out: Path, args) -> None:
    import torch
    from lerobot.processor.tokenizer_processor import ActionTokenizerProcessorStep

    manifest: dict = {"cases": {}}
    arrays: dict = {}
    for skip in cases.FAST_SKIP_TOKENS_UNDER_TEST:
        step = ActionTokenizerProcessorStep(
            action_tokenizer_name=args.fast_tokenizer,
            max_action_tokens=256,
            fast_skip_tokens=skip,
            paligemma_tokenizer_name=args.text_tokenizer,
            allow_truncation=True,
        )
        for name, action in cases.FAST_ACTIONS.items():
            tokens, mask, code_mask = step._tokenize_action(torch.tensor([action], dtype=torch.float32))
            key = f"{name}__skip{skip}"
            arrays[f"{key}__tokens"] = tokens.numpy().astype(np.int64)
            arrays[f"{key}__mask"] = mask.numpy()
            arrays[f"{key}__code_mask"] = code_mask.numpy()
            manifest["cases"][key] = {
                "code_tokens": int(code_mask.sum()),
                "real_tokens": int(mask.sum()),
                "first_tokens": tokens.numpy()[0][:8].tolist(),
            }
    dump(out / "fast.npz", arrays, manifest)


# --------------------------------------------------------------------------- #
# Normalisation and relative actions
# --------------------------------------------------------------------------- #
def group_norm(out: Path, args) -> None:
    import torch
    from lerobot.processor.normalize_processor import NormalizerProcessorStep
    from lerobot.processor.relative_action_processor import to_absolute_actions, to_relative_actions

    manifest: dict = {"cases": {}}
    arrays: dict = {}
    modes = ["QUANTILES", "MIN_MAX", "MEAN_STD"]
    for dim in cases.NORMALIZATION_DIMS:
        # The fixture is generated by the shared helper, so the candidate is fed the
        # very same numbers rather than a lookalike table.
        values = np.asarray(cases.norm_case_values(dim), dtype=np.float32)
        dim_stats = {key: list(column) for key, column in cases.norm_case_stats(dim).items()}
        for mode in modes:
            feature_stats = {"action": {key: torch.tensor(value, dtype=torch.float32) for key, value in dim_stats.items()}}
            from lerobot.configs import FeatureType, NormalizationMode

            step = NormalizerProcessorStep(
                features={},
                norm_map={FeatureType.ACTION: NormalizationMode(mode)},
                stats=feature_stats,
            )
            tensor = torch.tensor(values, dtype=torch.float32)
            normalized = step._apply_transform(tensor, "action", FeatureType.ACTION, inverse=False)
            restored = step._apply_transform(normalized, "action", FeatureType.ACTION, inverse=True)
            key = f"norm_d{dim}_{mode}"
            arrays[f"{key}__normalized"] = normalized.numpy()
            arrays[f"{key}__restored"] = restored.numpy()
            manifest["cases"][key] = {
                "max_abs_roundtrip": float(np.abs(restored.numpy() - values).max()),
                "constant_dim_value": float(normalized.numpy()[0, 2]),
            }

    # Relative actions with the gripper excluded.
    from lerobot.configs import FeatureType as _FeatureType

    for dim, names in ((6, cases.ACTION_NAMES_6), (12, None)):
        actions_raw, state_raw = cases.relative_case(dim)
        actions = np.asarray(actions_raw, dtype=np.float32)[None, ...]
        state = np.asarray([state_raw], dtype=np.float32)
        mask = [True] * dim
        if names is not None:
            from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

            step = RelativeActionsProcessorStep(enabled=True, exclude_joints=["gripper"], action_names=names)
            mask = step._build_mask(dim)
        else:
            mask = [True] * dim
        rel = to_relative_actions(torch.tensor(actions), torch.tensor(state), mask)
        back = to_absolute_actions(rel, torch.tensor(state), mask)
        key = f"relative_d{dim}"
        arrays[f"{key}__relative"] = rel.numpy()
        arrays[f"{key}__absolute"] = back.numpy()
        manifest["cases"][key] = {
            "mask": mask,
            "max_abs_roundtrip": float(np.abs(back.numpy() - actions).max()),
        }
    dump(out / "norm.npz", arrays, manifest)


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def group_images(out: Path, args) -> None:
    import torch
    from types import SimpleNamespace
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    class Stub:
        def __init__(self, keys, resolution=(224, 224)):
            self.config = SimpleNamespace(
                image_features={key: None for key in keys},
                image_resolution=resolution,
            )

        def parameters(self):
            yield torch.zeros(1)

    manifest: dict = {"cases": {}}
    arrays: dict = {}
    for case in cases.IMAGE_CASES:
        keys = list(case["keys"])
        present = list(case.get("present", keys))
        batch = {
            key: torch.tensor(value)
            for key, value in cases.image_case_batch(case).items()
            if key in present
        }
        if case.get("expect_error"):
            try:
                PI05Policy._preprocess_images(Stub(keys), batch)
            except ValueError as exc:
                manifest["cases"][case["name"]] = {
                    "error": type(exc).__name__,
                    "present": present,
                    "keys": keys,
                }
            else:
                raise AssertionError(case["name"] + ": the reference did not raise")
            continue

        images, masks = PI05Policy._preprocess_images(Stub(keys), batch)
        missing = [key for key in keys if key not in present]
        # Every image and every mask is exported, not just the first camera. This is
        # what makes swapped non-first cameras and wrong padding masks detectable.
        for index, (image, mask) in enumerate(zip(images, masks)):
            arrays[f"{case['name']}__image{index}"] = image.numpy()
            arrays[f"{case['name']}__mask{index}"] = mask.numpy()
        arrays[f"{case['name']}__n_cameras"] = np.asarray(len(images), dtype=np.int64)
        manifest["cases"][case["name"]] = {
            "keys": present + missing,
            "configured_keys": keys,
            "present": present,
            "n_cameras": len(images),
            "shapes": [list(image.shape) for image in images],
            "mask_shapes": [list(mask.shape) for mask in masks],
            "min": [float(image.min()) for image in images],
            "max": [float(image.max()) for image in images],
        }
    dump(out / "images.npz", arrays, manifest)


# --------------------------------------------------------------------------- #
# FAST fitting, saving and reloading
# --------------------------------------------------------------------------- #
def group_fit(out: Path, args) -> None:
    import shutil

    from lerobot.policies.fineart_vla.fit_fast_tokenizer import fit_fast_tokenizer

    cache_dir = out / "fit_cache"
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)

    fitted_path = fit_fast_tokenizer(
        dataset_repo_id="fixture/local-g1",
        cache_dir=cache_dir,
        base_tokenizer_name=args.fast_tokenizer,
        n_samples=cases.FIT_SAMPLES,
        chunk_size=cases.FIT_CHUNK_SIZE,
        dataset_root=dataset_root,
        exclude_episodes=cases.FIT_EXCLUDE_EPISODES,
        normalization_mode="QUANTILES",
        validation_samples=cases.FIT_VALIDATION_SAMPLES,
        max_reconstruction_rmse=0.10,
        max_dim_rmse=0.20,
    )
    fitted = Path(fitted_path)
    report = json.loads((fitted / "reconstruction_validation.json").read_text())

    from lerobot.processor.tokenizer_processor import ActionTokenizerProcessorStep

    probe = np.asarray(
        [[round(0.4 * ((i * 3 + d) % 11) / 11.0 - 0.2, 6) for d in range(cases.FIT_ACTION_DIM)] for i in range(cases.FIT_CHUNK_SIZE)],
        dtype=np.float32,
    )

    def tokenize(tokenizer_path):
        step = ActionTokenizerProcessorStep(
            action_tokenizer_name=str(tokenizer_path),
            max_action_tokens=256,
            fast_skip_tokens=1152,
            paligemma_tokenizer_name=args.text_tokenizer,
            allow_truncation=True,
        )
        import torch

        tokens, mask, code_mask = step._tokenize_action(torch.tensor([probe.tolist()], dtype=torch.float32))
        return tokens.numpy().astype(np.int64), mask.numpy(), code_mask.numpy()

    tokens_before, mask_before, code_before = tokenize(fitted)

    # Reload from an isolated copy with the original directory gone.
    isolated = out / "fit_isolated"
    shutil.rmtree(isolated, ignore_errors=True)
    shutil.copytree(fitted, isolated)
    moved = out / "fit_moved_away"
    shutil.rmtree(moved, ignore_errors=True)
    fitted.rename(moved)
    try:
        tokens_after, mask_after, code_after = tokenize(isolated)
    finally:
        moved.rename(fitted)

    arrays = {
        "probe": probe,
        "tokens_before": tokens_before,
        "mask_before": mask_before,
        "code_before": code_before,
        "tokens_after": tokens_after,
        "mask_after": mask_after,
        "code_after": code_after,
    }
    manifest = {
        "signature": fitted.name,
        "repair_report": report,
        "tokens_identical_after_reload": bool(
            np.array_equal(tokens_before, tokens_after)
            and np.array_equal(mask_before, mask_after)
            and np.array_equal(code_before, code_after)
        ),
        "files": sorted(path.name for path in fitted.iterdir()),
    }
    dump(out / "fit.npz", arrays, manifest)


# --------------------------------------------------------------------------- #
# Truncation behaviour
# --------------------------------------------------------------------------- #
def group_truncation(out: Path, args) -> None:
    import torch
    from lerobot.lerobot_types import TransitionKey
    from lerobot.policies.fineart_vla.processor_fineart_vla import FineARTVLATextTokenizerStep

    step = FineARTVLATextTokenizerStep(tokenizer_name=args.text_tokenizer, max_length=200)
    manifest = {}
    arrays = {}
    for name, text in (("short", "pick up the cube"), ("long", cases.TRUNCATION_LONG_PROMPT)):
        transition = {
            TransitionKey.OBSERVATION: {},
            TransitionKey.COMPLEMENTARY_DATA: {
                "messages": [{"role": "user", "content": text}],
                "message_streams": ["high_level"],
                "target_message_indices": [0],
            },
        }
        result = step(transition)
        obs = result[TransitionKey.OBSERVATION]
        comp = result[TransitionKey.COMPLEMENTARY_DATA]
        ids = obs["observation.language.tokens"].numpy()
        labels = comp["text_labels"].numpy()
        mask = obs["observation.language.attention_mask"].numpy()
        arrays[f"{name}__ids"] = ids
        arrays[f"{name}__labels"] = labels
        arrays[f"{name}__mask"] = mask
        manifest[name] = {
            "nonpad": int(mask.sum()),
            "n_supervised": int((labels != -100).sum()),
            "truncated": bool(mask.sum() == 200),
            "tail_id": int(ids[0, -1]),
        }
    dump(out / "truncation.npz", arrays, manifest)


GROUPS = {
    "routing": group_routing,
    "prompt": group_prompt,
    "stateloc": group_stateloc,
    "fast": group_fast,
    "norm": group_norm,
    "images": group_images,
    "fit": group_fit,
    "truncation": group_truncation,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--group", required=True, choices=sorted(GROUPS))
    parser.add_argument("--text-tokenizer", required=True)
    parser.add_argument("--fast-tokenizer", required=True)
    parser.add_argument("--dataset-root", default=None)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    GROUPS[args.group](out, args)
    print(json.dumps({"group": args.group, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
