#!/usr/bin/env python3
"""E2 launcher: run the pinned OpenPI train loop on the SO101 training view.

This wraps OpenPI's public ``train.main``; it does not fork the trainer.  The
only injection is the dataset loader, because the pinned OpenPI reads LeRobot
v2.1 while the external writer produces v3.0.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import openpi_so101  # noqa: E402


def _load_train_module():
    candidates = [Path("/opt/openpi/scripts/train.py")]
    source_root = os.environ.get("OPENPI_SOURCE_ROOT")
    if source_root:
        candidates.insert(0, Path(source_root) / "scripts/train.py")
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("openpi_train_entry", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module, path
    raise FileNotFoundError("could not locate the pinned OpenPI scripts/train.py")


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


def _config_snapshot(config) -> dict:
    return _jsonable(config)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("initial", "resume", "smoke", "probe"), required=True)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--exp-name", default="e2_lora_b1_s42")
    parser.add_argument("--state-mode", choices=("off", "on"), default="off")
    parser.add_argument("--train-episodes", default="",
                        help="comma-separated training episode ids; empty means the full split")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr-warmup-steps", type=int, default=10)
    parser.add_argument("--peak-lr", type=float, default=5e-5)
    parser.add_argument("--decay-steps", type=int, default=100)
    parser.add_argument("--decay-lr", type=float, default=5e-5)
    parser.add_argument("--data-loader-workers", type=int, default=0,
                        help="prefetch processes for the training loader; the sampler fixes batch "
                             "contents, so this changes only how fast batches arrive")
    parser.add_argument("--sampling-plan", type=Path, default=None)
    parser.add_argument("--data-admission", type=Path, default=None)
    parser.add_argument("--expected-total-episodes", type=int, default=None)
    parser.add_argument("--expected-train-episodes", type=int, default=None)
    parser.add_argument("--expected-validation-episodes", type=int, default=None)
    parser.add_argument("--dataset-label", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    run_root = args.run_root.expanduser().resolve()
    train_view = args.train_view.expanduser().resolve()
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    meta = json.loads((train_view / "meta.json").read_text(encoding="utf-8"))
    episode_count = len(meta.get("episodes", []))
    train_ids_meta = sorted(int(row["episode_index"]) for row in meta.get("episodes", [])
                            if row.get("split") == "train")
    val_ids_meta = sorted(int(row["episode_index"]) for row in meta.get("episodes", [])
                          if row.get("split") == "validation")
    if meta.get("state_action_dim") != 12:
        raise ValueError("training view is not the 12-D SO101 view")
    phase = args.phase or ("E4-T" if args.sampling_plan else "E3.1" if args.steps else "E2")
    if args.mode in {"smoke", "probe"}:
        if episode_count < 1:
            raise ValueError("training view has no episodes")
    elif phase == "E4-T":
        if args.batch_size <= 0:
            raise ValueError("E4-T requires an explicit batch size")
        if not args.sampling_plan or not args.data_admission:
            raise ValueError("E4-T requires a sampling plan and data admission manifest")
        admission = json.loads(args.data_admission.read_text(encoding="utf-8"))
        if not admission.get("data_gate", {}).get("passed"):
            raise ValueError("E4-T data gate did not pass")
        if admission.get("view_sha256") != meta.get("view_sha256"):
            raise ValueError("E4-T admission view hash differs from training view")
        expected_mapping = {
            "total": (args.expected_total_episodes, episode_count),
            "train": (args.expected_train_episodes, len(train_ids_meta)),
            "validation": (args.expected_validation_episodes, len(val_ids_meta)),
        }
        for key, (expected, actual) in expected_mapping.items():
            if expected is not None and expected != actual:
                raise ValueError(f"E4-T {key} episodes: expected {expected}, got {actual}")
        admission_counts = admission.get("totals", {})
        if admission_counts.get("episodes") != episode_count:
            raise ValueError("E4-T admission total differs from training view")
        if admission_counts.get("train") != len(train_ids_meta):
            raise ValueError("E4-T admission train count differs from training view")
        if admission_counts.get("validation") != len(val_ids_meta):
            raise ValueError("E4-T admission validation count differs from training view")
        if set(train_ids_meta) & set(val_ids_meta):
            raise ValueError("E4-T train/validation episodes overlap")
    else:
        if episode_count != 20:
            raise ValueError(f"training view has {episode_count} episodes; E2/E3.1 requires 20")
    if not (train_view / "assets/so101/norm_stats.json").is_file():
        raise FileNotFoundError("training view has no train-only norm stats")
    os.environ["SO101_TRAIN_VIEW"] = str(train_view)
    if args.train_episodes.strip():
        requested = sorted(int(value) for value in args.train_episodes.split(",") if value.strip())
        if phase == "E4-T" and requested != train_ids_meta:
            raise ValueError("E4-T must train the complete frozen train split")
        os.environ["SO101_TRAIN_EPISODES"] = args.train_episodes.strip()
    else:
        os.environ.pop("SO101_TRAIN_EPISODES", None)

    sampling_summary = None
    if phase == "E4-T" and args.mode not in {"smoke", "probe"}:
        policy = json.loads(args.sampling_plan.read_text(encoding="utf-8"))
        raw = openpi_so101.SO101EpisodeDataset(
            train_view, split="train", action_horizon=10,
            episode_ids=train_ids_meta)
        total = raw.sampling_summary()
        if total["split"] != "train" or len(total["groups"]) < 2:
            raise ValueError("E4-T training sample audit is incomplete")
        weights, audit = raw.weighted_sample_probabilities(policy)
        count = int(args.steps) * int(args.batch_size)
        indices = raw.precomputed_sample_indices(weights, count, seed=42)
        id_path = run_root / "training_sample_ids.jsonl.gz"
        import gzip
        with gzip.open(id_path, "wt", encoding="utf-8") as handle:
            for value in indices:
                handle.write(f"{value}\n")
        unique = len(set(indices))
        group_counts: dict[str, int] = {}
        for value in indices:
            group = raw.sample_metadata[value]["subgroup"]
            group_counts[group] = group_counts.get(group, 0) + 1
        sampling_summary = {
            "policy_sha256": _sha_file(args.sampling_plan),
            "eligible_windows": total,
            "configured_audit": audit,
            "requested_samples": count,
            "unique_samples_covered": unique,
            "empirical_coverage": unique / total["total_eligible_windows"],
            "actual_group_counts": group_counts,
            "actual_group_fractions": {
                key: value / count for key, value in sorted(group_counts.items())},
            "sample_ids_sha256": _sha_file(id_path),
            "sample_ids_file": id_path.name,
            "validation_used_for_training": False,
        }
        os.environ["SO101_TRAINING_SAMPLE_IDS"] = str(id_path)

    config = openpi_so101.make_train_config(
        mode=args.mode, run_root=run_root, train_view=train_view,
        base_checkpoint=base_checkpoint, exp_name=args.exp_name,
        state_mode=args.state_mode, steps_override=args.steps,
        save_interval=args.save_interval, batch_size=args.batch_size,
        lr_warmup_steps=args.lr_warmup_steps, peak_lr=args.peak_lr,
        decay_steps=args.decay_steps, decay_lr=args.decay_lr,
        dataset_label=args.dataset_label or meta.get("dataset_label", "so101"),
        phase=phase, num_workers=args.data_loader_workers,
    )
    snapshot = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "mode": args.mode,
        "run_root": str(run_root),
        "train_view": str(train_view),
        "base_checkpoint": str(base_checkpoint),
        "exp_name": args.exp_name,
        "config": _config_snapshot(config),
        "state_mode": args.state_mode,
        "train_episodes": args.train_episodes,
        "steps": config.num_train_steps,
        "save_interval": config.save_interval,
        "batch_size": config.batch_size,
        "lr_schedule": {"warmup_steps": args.lr_warmup_steps, "peak_lr": args.peak_lr,
                        "decay_steps": args.decay_steps, "decay_lr": args.decay_lr},
        "sampling_plan": str(args.sampling_plan) if args.sampling_plan else None,
        "sampling_summary": sampling_summary,
        "data_admission": str(args.data_admission) if args.data_admission else None,
        "train_view_meta_sha256": _sha_file(train_view / "meta.json"),
        "norm_stats_sha256": _sha_file(train_view / "assets/so101/norm_stats.json"),
    }
    config_path = run_root / f"config_{args.mode}.json"
    config_path.write_text(json.dumps(snapshot, indent=2, default=str) + "\n", encoding="utf-8")
    train_stats = json.loads((train_view / "train_stats.json").read_text(encoding="utf-8"))
    meta = json.loads((train_view / "meta.json").read_text(encoding="utf-8"))
    admission = (json.loads(args.data_admission.read_text(encoding="utf-8"))
                 if args.data_admission else {})
    run_config = {
        "schema_version": 1,
        "phase": phase,
        "state_mode": args.state_mode,
        "train_view": str(train_view),
        "train_episodes": args.train_episodes or f"all:{len(train_ids_meta)}",
        "train_episode_ids": train_ids_meta,
        "validation_episode_ids": val_ids_meta,
        "steps": config.num_train_steps,
        "save_interval": config.save_interval,
        "batch_size": config.batch_size,
        "lr_schedule": snapshot["lr_schedule"],
        "sampling_plan": snapshot["sampling_plan"],
        "sampling_summary": sampling_summary,
        "exp_name": args.exp_name,
        "base_checkpoint": str(base_checkpoint),
        "config_file": config_path.name,
        "config_sha256": _sha_file(config_path),
        "norm_stats_canonical_sha256": train_stats.get("norm_stats_canonical_sha256")
        or train_stats.get("norm_stats_sha256"),
        "view_sha256": meta.get("view_sha256"),
        "train_stats_sha256": _sha_file(train_view / "train_stats.json"),
        "data_admission_sha256": (_sha_file(args.data_admission) if args.data_admission else None),
        "admission_dataset_sha256": admission.get("dataset_sha256"),
        "admission_split_sha256": admission.get("split_sha256"),
    }
    (run_root / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    train_module, train_path = _load_train_module()
    original_loader = train_module._data_loader.create_data_loader
    train_module._data_loader.create_data_loader = openpi_so101.create_so101_data_loader
    print(json.dumps({"phase": phase, "mode": args.mode, "train_entry": str(train_path),
                      "checkpoint_dir": str(config.checkpoint_dir),
                      "steps": config.num_train_steps, "batch_size": config.batch_size,
                      "lr_schedule": snapshot["lr_schedule"],
                      "lora": [config.model.paligemma_variant,
                               config.model.action_expert_variant]}), flush=True)
    try:
        train_module.main(config)
    finally:
        train_module._data_loader.create_data_loader = original_loader
    (run_root / f"done_{args.mode}.json").write_text(
        json.dumps({"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                    "mode": args.mode, "checkpoint_dir": str(config.checkpoint_dir)},
                   indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
