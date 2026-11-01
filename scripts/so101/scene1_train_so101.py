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
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scene1_openpi_so101 as openpi_so101  # noqa: E402
import scene1_validation as validation  # noqa: E402
import clean60_contracts as clean60  # noqa: E402
from scene1_model_scope import resolve_model_scope  # noqa: E402


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


def _runtime_snapshot(config) -> dict:
    """Record the devices, mesh and batch sharding the run actually resolves to."""
    import jax

    from openpi.training import sharding as _sharding

    # OpenPI's mesh is built from jax.devices(), so validate against that same
    # device set before make_mesh can fail with a less useful reshape error.
    devices = list(jax.devices())
    device_count = len(devices)
    fsdp_devices = config.fsdp_devices
    batch_size = config.batch_size
    if device_count < 1:
        raise ValueError("OpenPI runtime requires at least one JAX device")
    if not isinstance(fsdp_devices, int) or isinstance(fsdp_devices, bool) or fsdp_devices < 1:
        raise ValueError("fsdp_devices must be a positive integer")
    if device_count % fsdp_devices != 0:
        raise ValueError(
            f"fsdp_devices ({fsdp_devices}) must divide the OpenPI device count ({device_count})"
        )
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if batch_size % device_count != 0:
        raise ValueError(
            f"global batch_size ({batch_size}) must be divisible by the OpenPI device count "
            f"({device_count})"
        )

    replicas = device_count // fsdp_devices
    mesh = _sharding.make_mesh(config.fsdp_devices)
    return {
        "backend": jax.default_backend(),
        "device_count": device_count,
        "device_ids": [device.id for device in devices],
        "device_kinds": sorted({device.device_kind for device in devices}),
        "fsdp_devices": fsdp_devices,
        "mesh_axis_names": list(mesh.axis_names),
        "mesh_shape": {str(key): int(value) for key, value in mesh.shape.items()},
        "data_parallel_replicas": replicas,
        "global_batch_size": batch_size,
        "batch_per_data_replica": batch_size // replicas,
        "batch_per_device_shard": batch_size // device_count,
        "batch_per_data_shard": batch_size // device_count,
        "batch_partitioned_across_all_devices": True,
        "fsdp_axis_replicates_batch": False,
        "batch_divisible_by_device_count": True,
        "num_train_steps": config.num_train_steps,
    }


def _training_settings(args, run_root: Path) -> dict:
    """Resume the saved model and freeze scope instead of today's defaults."""
    explicit_scope = {
        "paligemma_variant": args.paligemma_variant,
        "action_expert_variant": args.action_expert_variant,
        "freeze_vision_encoder": args.freeze_vision_encoder,
    }
    if args.mode == "resume":
        snapshot_path = run_root / "config_initial.json"
        if not snapshot_path.is_file():
            raise ValueError("resume requires the saved config_initial.json")
        saved = json.loads(snapshot_path.read_text(encoding="utf-8"))
        scope = resolve_model_scope(saved)
        for key, value in explicit_scope.items():
            if value is not None and value != scope[key]:
                raise ValueError(f"explicit {key} differs from the saved training config")
        state_mode = saved.get("state_mode", "off")
        if args.state_mode is not None and args.state_mode != state_mode:
            raise ValueError("explicit state_mode differs from the saved training config")
    else:
        scope = {
            "paligemma_variant": args.paligemma_variant or "gemma_2b_lora",
            "action_expert_variant": args.action_expert_variant or "gemma_300m",
            "freeze_vision_encoder": bool(args.freeze_vision_encoder),
        }
        state_mode = args.state_mode or "off"
    return {"model_scope": scope, "state_mode": state_mode}


class _FiniteOutputGuard(io.TextIOBase):
    """Abort at the per-step log write, before that step can be checkpointed."""

    _metric = re.compile(r"(?:loss|grad_norm|param_norm)=([-+A-Za-z0-9.]+)")
    _step = re.compile(r"Step ([0-9]+):")

    def __init__(self, wrapped, *, require_initial_update: bool, step_offset: int = 0):
        self.wrapped = wrapped
        self.require_initial_update = bool(require_initial_update)
        self.step_offset = int(step_offset)

    def write(self, text):
        text = text or ""
        metrics = {
            match.group(0).split("=", 1)[0]: match.group(1)
            for match in self._metric.finditer(text)
        }
        if any(value.lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}
               for value in metrics.values()):
            raise FloatingPointError(f"non-finite training metrics: {metrics}")
        step = self._step.search(text)
        if step and self.require_initial_update and int(step.group(1)) == 0:
            grad_norm = float(metrics.get("grad_norm", "nan"))
            if not grad_norm > 0.0:
                raise RuntimeError("first optimizer step produced a zero gradient")
        if self.step_offset:
            text = self._step.sub(
                lambda match: f"Step {int(match.group(1)) + self.step_offset}:",
                text,
            )
        return self.wrapped.write(text)

    def flush(self):
        return self.wrapped.flush()

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("initial", "resume", "smoke", "probe"), required=True)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--exp-name", default="e2_lora_b1_s42")
    parser.add_argument("--state-mode", choices=("off", "on"), default=None)
    parser.add_argument("--train-episodes", default="",
                        help="comma-separated training episode ids; empty means the full split")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--optimizer", choices=("AdamW",), default="AdamW")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=None,
                        help="EMA decay; omitted means EMA is disabled")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--lr-warmup-steps", type=int, default=10)
    parser.add_argument("--peak-lr", type=float, default=5e-5)
    parser.add_argument("--decay-steps", type=int, default=100)
    parser.add_argument("--decay-lr", type=float, default=5e-5)
    parser.add_argument("--data-loader-workers", type=int, default=0,
                        help="prefetch processes for the training loader; the sampler fixes batch "
                             "contents, so this changes only how fast batches arrive")
    parser.add_argument("--sampling-plan", type=Path, default=None)
    parser.add_argument("--data-admission", type=Path, default=None)
    parser.add_argument("--registered-config-sha256", default=None)
    parser.add_argument("--registered-training-config", type=Path, default=None)
    parser.add_argument("--pifast-plan", type=Path, default=None)
    parser.add_argument("--pifast-plan-sha256", default=None)
    parser.add_argument("--pifast-data-audit", type=Path, default=None)
    parser.add_argument("--pifast-data-audit-sha256", default=None)
    parser.add_argument("--fast-tokenizer-dir", type=Path, default=None)
    parser.add_argument("--paligemma-tokenizer", type=Path, default=None)
    parser.add_argument("--fast-ce-weight", type=float, default=None)
    parser.add_argument("--require-clean-data", action="store_true")
    parser.add_argument("--require-action-norm-audit", action="store_true")
    parser.add_argument("--registered-base-sha256", default=None)
    parser.add_argument("--checkpoint-probe", action="store_true",
                        help="allow a short isolated full-update/checkpoint probe")
    parser.add_argument("--resume-step", type=int, default=None,
                        help="authoritative TrainState.step restored by a checkpoint probe")
    parser.add_argument("--expected-total-episodes", type=int, default=None)
    parser.add_argument("--expected-train-episodes", type=int, default=None)
    parser.add_argument("--expected-validation-episodes", type=int, default=None)
    parser.add_argument("--validation-every", type=int, default=None,
                        help="compute validation loss after every N optimizer updates")
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--validation-batch-size", type=int, default=16)
    parser.add_argument("--require-validation-loss", action="store_true")
    parser.add_argument("--validation-benchmark", action="store_true",
                        help="probe mode only: validate once after its final short update")
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--fsdp-devices", type=int, default=1,
                        help="devices per FSDP replica; the remaining visible devices are "
                             "data-parallel replicas of the sharded model")
    parser.add_argument("--paligemma-variant", default=None)
    parser.add_argument("--action-expert-variant", default=None)
    parser.add_argument("--freeze-vision-encoder", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--expected-groups", type=int, default=2,
                        help="minimum sampling groups; a clean-only protocol uses 1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.validation_batch_size < 1:
        raise ValueError("--validation-batch-size must be at least 1")
    run_root = args.run_root.expanduser().resolve()
    train_view = args.train_view.expanduser().resolve()
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    settings = _training_settings(args, run_root)
    model_scope = settings["model_scope"]
    args.state_mode = settings["state_mode"]
    run_root.mkdir(parents=True, exist_ok=True)
    meta = json.loads((train_view / "meta.json").read_text(encoding="utf-8"))
    registered_config = None
    pifast_plan = None
    pifast_data_audit = None
    pifast_data_audit_sha256 = None
    pifast_plan_sha256 = None
    if args.pifast_plan is not None:
        if args.registered_training_config is not None:
            raise ValueError(
                "pifast plan and legacy registered training config are mutually exclusive")
        pifast_plan_sha256 = _sha_file(args.pifast_plan)
        if pifast_plan_sha256 != args.pifast_plan_sha256:
            raise ValueError("PI-FAST plan hash mismatch")
        pifast_plan = json.loads(args.pifast_plan.read_text(encoding="utf-8"))
        if pifast_plan.get("plan_id") != "scene1-pifast-ki-v2":
            raise ValueError("unexpected PI-FAST plan identity")
        if pifast_plan.get("supersedes") != (
                "configs/so101/scene1_pifast_ki_training_v1.json"):
            raise ValueError("PI-FAST v2 must explicitly supersede v1")
        if args.mode in {"initial", "resume"} and not pifast_plan.get(
                "execution_authorized"):
            raise ValueError("formal PI-FAST training is not authorized by its plan")
        if args.pifast_data_audit is None:
            raise ValueError("PI-FAST training requires a data/FAST audit")
        pifast_data_audit_sha256 = _sha_file(args.pifast_data_audit)
        if pifast_data_audit_sha256 != args.pifast_data_audit_sha256:
            raise ValueError("PI-FAST data/FAST audit hash mismatch")
        pifast_data_audit = json.loads(
            args.pifast_data_audit.read_text(encoding="utf-8"))
        if pifast_data_audit.get("status") != "pass":
            raise ValueError("PI-FAST data/FAST audit did not pass")
        if pifast_data_audit.get("view_sha256") != meta.get("view_sha256"):
            raise ValueError("PI-FAST data/FAST audit view hash differs")
        expected_norm_sha = _sha_file(
            train_view / "assets/so101/norm_stats.json")
        if pifast_data_audit.get("norm_stats_sha256") != expected_norm_sha:
            raise ValueError("PI-FAST data/FAST audit norm hash differs")
        if args.fast_tokenizer_dir is None or args.paligemma_tokenizer is None:
            raise ValueError("PI-FAST tokenizer resources are required")
        if args.fast_ce_weight is None:
            raise ValueError("fast_ce_weight must be supplied explicitly")
        if args.mode in {"initial", "resume"}:
            registered_weight = pifast_plan["loss"].get("fast_ce_weight")
            if args.fast_ce_weight != registered_weight:
                raise ValueError(
                    "formal fast_ce_weight differs from the frozen plan")
    if args.registered_training_config is not None:
        registered_config = json.loads(
            args.registered_training_config.read_text(encoding="utf-8"))
        actual_hash = _sha_file(args.registered_training_config)
        if not args.registered_config_sha256 or actual_hash != args.registered_config_sha256:
            raise ValueError("registered training configuration hash mismatch")
    elif args.registered_config_sha256:
        raise ValueError("registered config hash requires --registered-training-config")
    checkpoint_steps = ()
    validation_settings = None
    probe_settings = None
    full_settings = None
    if (pifast_plan is None and registered_config is not None
            and registered_config.get("validation_loss")
            and not args.require_validation_loss):
        raise ValueError("registered validation config requires --require-validation-loss")
    if args.require_validation_loss:
        if registered_config is None and pifast_plan is None:
            raise ValueError("Validation Loss requires the registered Scene1 config")
        if pifast_plan is not None:
            registered_updates = int(
                pifast_plan["optimization"]["optimizer_updates"])
            checkpoint_steps = tuple(
                int(value)
                for value in pifast_plan["checkpointing"]["save_steps"])
            validation_config = pifast_plan["validation"]
            validation_settings = {
                "split": "validation",
                "batch_size": int(validation_config["full_batch_size"]),
                "seed": int(pifast_plan["optimization"]["seed"]),
                "norm_stats_source": "train_only",
                "train_false": bool(validation_config["train_false"]),
                "gradient_updates": int(validation_config["gradient_updates"]),
                "sample_weighting": "uniform_per_valid_sample",
                "fixed_flow_time_and_noise": bool(
                    validation_config["fixed_flow_time_and_noise"]),
                "probe": {
                    "metric": "val_loss_probe",
                    "every_updates": int(
                        validation_config["quick_probe_every_updates"]),
                    "samples": int(validation_config["quick_probe_samples"]),
                    "balanced_across_episodes": True,
                    "record_prefix": "val_loss_probe",
                    "requires_checkpoint": False,
                },
                "full": {
                    "metric": "val_loss_full",
                    "samples": int(validation_config["full_samples"]),
                    "record_prefix": "val_loss_full",
                    "requires_checkpoint": True,
                    "checkpoint_steps": list(checkpoint_steps),
                },
            }
        else:
            registered_updates = int(registered_config["optimisation"]["updates"])
            checkpoint_steps = validation.checkpoint_steps(
                registered_config, registered_updates)
        if (args.mode in {"initial", "resume"} and not args.checkpoint_probe
                and args.steps != registered_updates):
            raise ValueError("formal training steps differ from the registered target")
        if pifast_plan is None:
            validation_settings = registered_config.get("validation_loss") or {}
        expected_validation = {
            "split": "validation",
            "batch_size": 16,
            "seed": 42,
            "norm_stats_source": "train_only",
            "train_false": True,
            "gradient_updates": 0,
            "sample_weighting": "uniform_per_valid_sample",
            "fixed_flow_time_and_noise": True,
        }
        for key, value in expected_validation.items():
            if validation_settings.get(key) != value:
                raise ValueError(f"validation_loss.{key} is not {value!r}")
        probe_settings = validation_settings.get("probe") or {}
        full_settings = validation_settings.get("full") or {}
        if probe_settings != {
            "metric": "val_loss_probe",
            "every_updates": 50,
            "samples": 256,
            "balanced_across_episodes": True,
            "record_prefix": "val_loss_probe",
            "requires_checkpoint": False,
        }:
            raise ValueError("validation_loss.probe settings differ from the plan")
        expected_full_samples = (
            int(pifast_plan["validation"]["full_samples"])
            if pifast_plan is not None else 8300
        )
        if full_settings != {
            "metric": "val_loss_full",
            "samples": expected_full_samples,
            "record_prefix": "val_loss_full",
            "requires_checkpoint": True,
            "checkpoint_steps": list(checkpoint_steps),
        }:
            raise ValueError("validation_loss.full settings differ from the plan")
        if pifast_plan is not None:
            if pifast_plan["checkpointing"].get("retain_all") is not True:
                raise ValueError("every registered checkpoint must be retained")
        elif registered_config["checkpointing"].get(
                "retain_all_saved_steps") is not True:
            raise ValueError("every registered checkpoint must be retained")
        for name, actual in (
            ("--validation-every", args.validation_every),
            ("--validation-seed", args.validation_seed),
            ("--validation-batch-size", args.validation_batch_size),
        ):
            if actual is not None:
                expected = {
                    "--validation-every": probe_settings["every_updates"],
                    "--validation-seed": validation_settings["seed"],
                    "--validation-batch-size": validation_settings["batch_size"],
                }[name]
                if actual != expected:
                    raise ValueError(f"{name} differs from the registered validation config")
        args.validation_every = int(probe_settings["every_updates"])
        args.validation_seed = int(validation_settings["seed"])
        args.validation_batch_size = int(validation_settings["batch_size"])
    if args.validation_benchmark and (
        args.mode != "probe" or not args.require_validation_loss
    ):
        raise ValueError("--validation-benchmark requires probe mode and validation loss")
    if args.require_clean_data:
        if pifast_plan is not None:
            raise ValueError(
                "PI-FAST uses its registered data/FAST audit, not clean60 admission")
        if registered_config is None:
            raise ValueError("clean60 training requires its registered configuration")
        problems = clean60.training_config_problems(registered_config)
        if problems:
            raise ValueError("registered training configuration failed: " + "; ".join(problems))
        binding_path = train_view / "protocol_binding.json"
        split = json.loads((train_view / "splits.json").read_text(encoding="utf-8"))
        binding = (json.loads(binding_path.read_text(encoding="utf-8"))
                   if binding_path.is_file() else None)
        problems = clean60.view_contract_problems(
            meta, protocol_binding=binding, require_clean=True)
        admission_path = (args.data_admission or train_view / "e4t_admission.json")
        admission = (json.loads(admission_path.read_text(encoding="utf-8"))
                     if admission_path.is_file() else {})
        problems.extend(clean60.admission_contract_problems(
            admission, meta=meta, split=split,
            train_stats_sha256=_sha_file(train_view / "train_stats.json")))
        for name in ("scene_audit.json", "data_audit.json", "stats_recompute.json"):
            if not (train_view / name).is_file():
                problems.append(f"missing {name}")
        if not problems:
            norm_stats_sha256 = _sha_file(train_view / "assets/so101/norm_stats.json")
            problems.extend(clean60.scene_audit_problems(
                json.loads((train_view / "scene_audit.json").read_text()),
                view_sha256=meta.get("view_sha256", "")))
            problems.extend(clean60.data_audit_problems(
                json.loads((train_view / "data_audit.json").read_text()),
                view_sha256=meta.get("view_sha256", "")))
            problems.extend(clean60.stats_recompute_problems(
                json.loads((train_view / "stats_recompute.json").read_text()),
                view_sha256=meta.get("view_sha256", ""),
                norm_stats_sha256=norm_stats_sha256))
        if problems:
            raise ValueError("clean data contract failed: " + "; ".join(problems))
        data = registered_config["data"]
        if (args.expected_total_episodes, args.expected_train_episodes,
                args.expected_validation_episodes) != (
                    data["view_episodes"], data["train"], data["validation"]):
            raise ValueError("CLI episode counts differ from the registered clean data profile")
        if (args.mode == "initial" and not args.checkpoint_probe
                and args.steps != registered_config["optimisation"]["planned_updates"]):
            raise ValueError("formal clean60 initial training must use 25000 updates")
        if args.batch_size != registered_config["optimisation"]["primary_batch_size"]:
            raise ValueError("formal clean60 training must use global batch 8")
    if (args.registered_base_sha256
            and clean60.sha256_tree(base_checkpoint) != args.registered_base_sha256):
        raise ValueError("base checkpoint does not match the probe-bound hash")
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
    elif phase in {"E4-T", "scene1"}:
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
    norm_audit_path = train_view / "action_norm_audit.json"
    norm_audit = None
    norm_audit_sha256 = None
    if args.require_action_norm_audit:
        if not norm_audit_path.is_file():
            raise ValueError("training requires the action normalization audit")
        norm_audit = json.loads(norm_audit_path.read_text(encoding="utf-8"))
        norm_audit_sha256 = _sha_file(norm_audit_path)
        problems = clean60.norm_audit_problems(
            norm_audit,
            view_sha256=meta.get("view_sha256", ""),
            norm_stats_sha256=_sha_file(train_view / "assets/so101/norm_stats.json"),
            base_checkpoint_sha256=args.registered_base_sha256 or "",
            training_config_sha256=args.registered_config_sha256 or "",
            expected_train=52, expected_validation=12,
        )
        if problems:
            raise ValueError("action normalization audit failed: " + "; ".join(problems))
    os.environ["SO101_TRAIN_VIEW"] = str(train_view)
    if args.train_episodes.strip():
        requested = sorted(int(value) for value in args.train_episodes.split(",") if value.strip())
        if phase in {"E4-T", "scene1"} and requested != train_ids_meta:
            raise ValueError("E4-T must train the complete frozen train split")
        os.environ["SO101_TRAIN_EPISODES"] = args.train_episodes.strip()
    else:
        os.environ.pop("SO101_TRAIN_EPISODES", None)

    validation_plans = {}
    validation_plan_hashes = {}
    validation_keys = {}
    if args.require_validation_loss:
        validation_dataset = openpi_so101.SO101EpisodeDataset(
            train_view,
            split="validation",
            action_horizon=args.action_horizon,
            episode_ids=val_ids_meta,
        )
        full_sample_count = len(validation_dataset)
        expected_samples = int(full_settings["samples"])
        if full_sample_count != expected_samples:
            raise ValueError(
                f"validation sample count {full_sample_count} differs "
                f"from registered {expected_samples}")
        probe_indices, probe_selection = validation.balanced_probe_indices(
            validation_dataset.sample_metadata,
            int(probe_settings["samples"]),
            args.validation_seed,
        )
        plan_inputs = {
            "full": (list(range(full_sample_count)), None),
            "probe": (probe_indices, probe_selection),
        }
        for kind, (sample_indices, selection) in plan_inputs.items():
            settings = full_settings if kind == "full" else probe_settings
            planned = validation.fixed_sample_plan(
                sample_indices, args.validation_batch_size, args.validation_seed)
            keys = validation.validation_batch_keys(
                args.validation_seed, planned["batch_count"])
            plan = validation.build_plan(
                sample_indices=sample_indices,
                global_batch_size=args.validation_batch_size,
                seed=args.validation_seed,
                metric=settings["metric"],
                validation_every=(
                    args.validation_every if kind == "probe" else None),
                action_horizon=args.action_horizon,
                internal_action_dim=32,
                view_sha256=meta.get("view_sha256", ""),
                norm_stats_sha256=_sha_file(
                    train_view / "assets/so101/norm_stats.json"),
                batch_keys=keys,
                selection=selection,
                episode_ids=val_ids_meta,
            )
            plan["record_prefix"] = settings["record_prefix"]
            plan["requires_checkpoint"] = bool(settings["requires_checkpoint"])
            _, digest = validation.register_plan(
                run_root,
                plan,
                name=f"validation_{kind}_plan.json",
                fresh=args.mode in {"initial", "probe", "smoke"},
            )
            validation_plans[kind] = plan
            validation_keys[kind] = keys
            validation_plan_hashes[kind] = digest

    sampling_summary = None
    if (phase in {"E4-T", "scene1"}
            and (args.mode not in {"smoke", "probe"} or args.checkpoint_probe)):
        if args.mode == "resume" and args.resume_step is None:
            raise ValueError("checkpoint resume requires the restored TrainState.step")
        policy = json.loads(args.sampling_plan.read_text(encoding="utf-8"))
        raw = openpi_so101.SO101EpisodeDataset(
            train_view, split="train", action_horizon=args.action_horizon,
            episode_ids=train_ids_meta)
        total = raw.sampling_summary()
        if total["split"] != "train" or len(total["groups"]) < args.expected_groups:
            raise ValueError("E4-T training sample audit is incomplete")
        weights, audit = raw.weighted_sample_probabilities(policy)
        count = int(args.steps) * int(args.batch_size)
        indices = raw.precomputed_sample_indices(weights, count, seed=args.seed)
        id_path = run_root / "training_sample_ids.jsonl.gz"
        import gzip
        with gzip.open(id_path, "wt", encoding="utf-8") as handle:
            for value in indices:
                handle.write(f"{value}\n")
        offset = (int(args.resume_step) * int(args.batch_size)
                  if args.mode == "resume" else 0)
        if offset < 0 or offset >= len(indices):
            raise ValueError("resume sample offset is outside the precomputed sequence")
        active_indices_path = id_path
        if offset:
            active_indices_path = run_root / "resume_training_sample_ids.jsonl.gz"
            with gzip.open(active_indices_path, "wt", encoding="utf-8") as handle:
                for value in indices[offset:]:
                    handle.write(f"{value}\n")
        active_indices = indices[offset:]
        unique = len(set(active_indices))
        group_counts: dict[str, int] = {}
        for value in active_indices:
            group = raw.sample_metadata[value]["subgroup"]
            group_counts[group] = group_counts.get(group, 0) + 1
        sampling_summary = {
            "policy_sha256": _sha_file(args.sampling_plan),
            "eligible_windows": total,
            "configured_audit": audit,
            "requested_samples": len(active_indices),
            "full_requested_samples": count,
            "unique_samples_covered": unique,
            "empirical_coverage": unique / total["total_eligible_windows"],
            "actual_group_counts": group_counts,
            "actual_group_fractions": {
                key: value / len(active_indices)
                for key, value in sorted(group_counts.items())},
            "sample_ids_sha256": _sha_file(active_indices_path),
            "full_sample_ids_sha256": _sha_file(id_path),
            "sample_ids_file": active_indices_path.name,
            "full_sample_ids_file": id_path.name,
            "resume_batch_offset": offset,
            "restored_train_state_step": args.resume_step,
            "validation_used_for_training": False,
        }
        os.environ["SO101_TRAINING_SAMPLE_IDS"] = str(active_indices_path)

    if pifast_plan is not None:
        interface = pifast_plan["interface"]
        optimization = pifast_plan["optimization"]
        formal_pifast_run = (
            args.mode in {"initial", "resume"} and not args.checkpoint_probe)
        from scene1_pifast_ki_contract import plan_problems as plan_contract_problems

        plan_problems = plan_contract_problems(
            pifast_plan, formal=formal_pifast_run)
        expected_values = {
            "interface.action_horizon": (
                interface["action_horizon"], args.action_horizon),
            "interface.n_action_steps": (
                interface["n_action_steps"], args.action_horizon),
            "interface.internal_action_dim": (
                interface["internal_action_dim"], 32),
            "optimization.warmup_updates": (
                optimization["warmup_updates"], args.lr_warmup_steps),
            "optimization.peak_lr": (optimization["peak_lr"], args.peak_lr),
            "optimization.cosine_decay_updates": (
                optimization["cosine_decay_updates"], args.decay_steps),
            "optimization.final_lr": (optimization["final_lr"], args.decay_lr),
            "optimization.ema_enabled": (False, args.ema_decay is not None),
            "runtime.fsdp_devices": (
                pifast_plan["runtime"]["fsdp_devices"], args.fsdp_devices),
            "model_scope.language.rank": (
                pifast_plan["model_scope"]["language"]["rank"], 16),
            "model_scope.action_expert.rank": (
                pifast_plan["model_scope"]["action_expert"]["rank"], 64),
            "model_scope.vision_encoder": (
                pifast_plan["model_scope"]["vision_encoder"],
                "frozen" if args.freeze_vision_encoder else "full_trainable"),
        }
        if formal_pifast_run:
            expected_values.update({
                "optimization.optimizer_updates": (
                    optimization["optimizer_updates"], args.steps),
                "optimization.global_batch_size": (
                    optimization["global_batch_size"], args.batch_size),
            })
        for name, (expected, actual) in expected_values.items():
            comparable = actual
            if name == "optimization.ema_enabled":
                comparable = bool(actual)
            if expected != comparable:
                plan_problems.append(
                    f"{name}: plan={expected!r}, actual={comparable!r}")
        if formal_pifast_run:
            if args.steps != 40000:
                plan_problems.append("formal optimizer update count is not 40000")
            if args.batch_size != optimization["global_batch_size"]:
                plan_problems.append("formal batch differs from selected plan")
            if tuple(checkpoint_steps) != tuple(
                    optimization_updates
                    for optimization_updates in range(2000, 40001, 2000)):
                plan_problems.append("formal checkpoint schedule is not 2000..40000")
            if pifast_plan.get("resource", {}).get("minimum_memory_margin_mib", -1) < 100:
                plan_problems.append("formal memory margin is below 100 MiB")
            if not pifast_plan.get("resource", {}).get("wall_limit_s"):
                plan_problems.append("formal wall-clock limit is not registered")
        if args.registered_base_sha256 != (
                pifast_plan["initialization"]["pi05_base"]["tree_sha256"]):
            plan_problems.append("base checkpoint tree hash is not plan-bound")
        if plan_problems:
            raise ValueError(
                "PI-FAST plan does not match concrete launch: "
                + "; ".join(plan_problems))
        import scene1_pifast_ki_runtime as pifast_runtime

        config = pifast_runtime.make_train_config(
            mode=args.mode,
            run_root=run_root,
            train_view=train_view,
            base_checkpoint=base_checkpoint,
            exp_name=args.exp_name,
            steps=int(args.steps),
            batch_size=int(args.batch_size),
            save_interval=(
                1 if args.require_validation_loss and (
                    args.mode in {"initial", "resume"}
                    or args.validation_benchmark)
                else int(args.save_interval)
            ),
            warmup_steps=int(args.lr_warmup_steps),
            peak_lr=float(args.peak_lr),
            decay_steps=int(args.decay_steps),
            decay_lr=float(args.decay_lr),
            seed=int(args.seed),
            fsdp_devices=int(args.fsdp_devices),
            num_workers=int(args.data_loader_workers),
            fast_ce_weight=float(args.fast_ce_weight),
            fast_tokenizer_dir=args.fast_tokenizer_dir,
            paligemma_tokenizer_path=args.paligemma_tokenizer,
            dataset_label=args.dataset_label or "scene1_single_random_v1",
        )
    else:
        config = openpi_so101.make_train_config(
            mode=args.mode, run_root=run_root, train_view=train_view,
            base_checkpoint=base_checkpoint, exp_name=args.exp_name,
            state_mode=args.state_mode, steps_override=args.steps,
            save_interval=(
                1 if args.require_validation_loss and (
                    args.mode in {"initial", "resume"} or args.validation_benchmark)
                else args.save_interval
            ),
            batch_size=args.batch_size,
            lr_warmup_steps=args.lr_warmup_steps, peak_lr=args.peak_lr,
            decay_steps=args.decay_steps, decay_lr=args.decay_lr,
            dataset_label=args.dataset_label or meta.get("dataset_label", "so101"),
            phase=phase, num_workers=args.data_loader_workers,
            fsdp_devices=args.fsdp_devices, **model_scope,
            action_horizon=args.action_horizon, seed=args.seed,
            gradient_clip_norm=args.gradient_clip_norm, optimizer_name=args.optimizer,
            ema_decay=args.ema_decay, dtype=args.dtype,
        )
    if args.require_validation_loss and (
        args.validation_batch_size % int(config.fsdp_devices) != 0
    ):
        raise ValueError("validation batch size must divide across FSDP devices")
    snapshot = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "mode": args.mode,
        "run_id": args.run_id,
        "registered_config_sha256": args.registered_config_sha256,
        "runtime": _runtime_snapshot(config),
        "run_root": str(run_root),
        "train_view": str(train_view),
        "base_checkpoint": str(base_checkpoint),
        "exp_name": args.exp_name,
        "config": _config_snapshot(config),
        "state_mode": args.state_mode,
        "model_scope": model_scope,
        "train_episodes": args.train_episodes,
        "steps": config.num_train_steps,
        "save_interval": config.save_interval,
        "checkpoint_steps": list(checkpoint_steps),
        "validation_loss": validation_settings,
        "validation_probe_plan_sha256": validation_plan_hashes.get("probe"),
        "validation_full_plan_sha256": validation_plan_hashes.get("full"),
        "validation_sample_counts": {
            kind: plan["sample_count"] for kind, plan in validation_plans.items()
        },
        "batch_size": config.batch_size,
        "seed": config.seed,
        "action_horizon": config.model.action_horizon,
        "optimizer": args.optimizer,
        "gradient_clip_norm": args.gradient_clip_norm,
        "ema_decay": config.ema_decay,
        "dtype": config.model.dtype,
        "lr_schedule": {"warmup_steps": args.lr_warmup_steps, "peak_lr": args.peak_lr,
                        "decay_steps": args.decay_steps, "decay_lr": args.decay_lr},
        "sampling_plan": str(args.sampling_plan) if args.sampling_plan else None,
        "sampling_summary": sampling_summary,
        "data_admission": str(args.data_admission) if args.data_admission else None,
        "train_view_meta_sha256": _sha_file(train_view / "meta.json"),
        "norm_stats_sha256": _sha_file(train_view / "assets/so101/norm_stats.json"),
        "norm_audit_sha256": norm_audit_sha256,
        "base_checkpoint_sha256": (norm_audit.get("base_checkpoint_sha256")
                                   if norm_audit else None),
    }
    config_path = run_root / f"config_{args.mode}.json"
    config_path.write_text(json.dumps(snapshot, indent=2, default=str) + "\n", encoding="utf-8")
    if args.mode == "probe" and args.checkpoint_probe:
        initial_sidecar = run_root / "config_initial.json"
        if initial_sidecar.exists():
            raise ValueError("checkpoint probe initial sidecar already exists")
        initial_sidecar.write_text(
            json.dumps(snapshot, indent=2, default=str) + "\n", encoding="utf-8")
    train_stats = json.loads((train_view / "train_stats.json").read_text(encoding="utf-8"))
    meta = json.loads((train_view / "meta.json").read_text(encoding="utf-8"))
    admission = (json.loads(args.data_admission.read_text(encoding="utf-8"))
                 if args.data_admission else {})
    run_config = {
        "schema_version": 1,
        "phase": phase,
        "state_mode": args.state_mode,
        "model_scope": model_scope,
        "train_view": str(train_view),
        "train_episodes": args.train_episodes or f"all:{len(train_ids_meta)}",
        "train_episode_ids": train_ids_meta,
        "validation_episode_ids": val_ids_meta,
        "steps": config.num_train_steps,
        "save_interval": config.save_interval,
        "checkpoint_steps": list(checkpoint_steps),
        "validation_loss": validation_settings,
        "validation_probe_plan_sha256": validation_plan_hashes.get("probe"),
        "validation_full_plan_sha256": validation_plan_hashes.get("full"),
        "validation_sample_counts": {
            kind: plan["sample_count"] for kind, plan in validation_plans.items()
        },
        "batch_size": config.batch_size,
        "seed": config.seed,
        "action_horizon": config.model.action_horizon,
        "optimizer": args.optimizer,
        "gradient_clip_norm": args.gradient_clip_norm,
        "ema_decay": config.ema_decay,
        "dtype": config.model.dtype,
        "run_id": args.run_id,
        "registered_config_sha256": args.registered_config_sha256,
        "lr_schedule": snapshot["lr_schedule"],
        "sampling_plan": snapshot["sampling_plan"],
        "sampling_summary": sampling_summary,
        "exp_name": args.exp_name,
        "base_checkpoint": str(base_checkpoint),
        "config_file": config_path.name,
        "config_sha256": _sha_file(config_path),
        "norm_stats_canonical_sha256": train_stats.get("norm_stats_canonical_sha256")
        or train_stats.get("norm_stats_sha256"),
        "norm_audit_sha256": norm_audit_sha256,
        "base_checkpoint_sha256": (norm_audit.get("base_checkpoint_sha256")
                                   if norm_audit else None),
        "view_sha256": meta.get("view_sha256"),
        "train_stats_sha256": _sha_file(train_view / "train_stats.json"),
        "data_admission_sha256": (_sha_file(args.data_admission) if args.data_admission else None),
        "admission_dataset_sha256": admission.get("dataset_sha256"),
        "admission_split_sha256": admission.get("split_sha256"),
    }
    (run_root / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    train_module, train_path = _load_train_module()
    saw_checkpoint_steps: list[int] = []
    if args.require_validation_loss:
        import jax
        import numpy as np
        from openpi.training import sharding

        mesh = sharding.make_mesh(config.fsdp_devices)
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        validation_config = dataclasses.replace(
            config, batch_size=int(args.validation_batch_size))
        pvalidate = jax.jit(validation.validation_batch_losses)
        validation_loader_holders = {"probe": None, "full": None}

        def validation_loader(kind: str):
            loader = validation_loader_holders[kind]
            if loader is None:
                plan = validation_plans[kind]
                loader = openpi_so101.create_so101_data_loader(
                    validation_config,
                    sharding=data_sharding,
                    shuffle=False,
                    num_batches=plan["batch_count"],
                    split="validation",
                    episode_ids=val_ids_meta,
                    sampler_indices=plan["_sampler_indices"],
                    expected_samples=len(validation_dataset),
                )
                validation_loader_holders[kind] = loader
            return loader

        def evaluate_validation(state, kind: str, checkpoint_info: dict) -> dict:
            if kind not in {"probe", "full"}:
                raise ValueError(f"unknown validation metric {kind}")
            plan = validation_plans[kind]
            keys = validation_keys[kind]
            step = int(jax.device_get(state.step))
            path = run_root / f"{plan['record_prefix']}_{step}.json"
            if path.is_file():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if (
                    int(saved.get("train_state_step", -1)) != step
                    or saved.get("validation_plan_sha256") != validation_plan_hashes[kind]
                ):
                    raise ValueError(f"validation record is inconsistent: {path.name}")
                return saved

            before_step = int(jax.device_get(state.step))
            before_state_ids = {
                "params": tuple(
                    (jax.tree_util.keystr(path), id(value))
                    for path, value in jax.tree_util.tree_leaves_with_path(state.params)
                ),
                "optimizer": tuple(
                    (jax.tree_util.keystr(path), id(value))
                    for path, value in jax.tree_util.tree_leaves_with_path(state.opt_state)
                ),
                "ema": (
                    tuple(
                        (jax.tree_util.keystr(path), id(value))
                        for path, value in jax.tree_util.tree_leaves_with_path(
                            state.ema_params)
                    ) if state.ema_params is not None else None
                ),
            }
            last_heartbeat = [0.0]
            started = time.monotonic()

            def evaluate_batch(batch, batch_index):
                with sharding.set_mesh(mesh):
                    values = pvalidate(
                        state, batch[0], batch[1], keys[batch_index])
                return np.asarray(jax.device_get(values), dtype=np.float32).reshape(-1)

            def progress(completed, total):
                now = time.monotonic()
                if completed == total or now - last_heartbeat[0] >= 30.0:
                    last_heartbeat[0] = now
                    print(json.dumps({
                        "event": "validation_progress",
                        "metric": plan["metric"],
                        "train_state_step": step,
                        "batches_completed": int(completed),
                        "batches_total": int(total),
                    }), flush=True)

            result = validation.collect_validation_loss(
                validation_loader(kind),
                plan["valid_samples_per_batch"],
                evaluate_batch,
                progress=progress,
            )
            if result["samples_evaluated"] != plan["sample_count"]:
                raise RuntimeError(
                    f"validation evaluated {result['samples_evaluated']} of "
                    f"{plan['sample_count']} samples")
            after_step = int(jax.device_get(state.step))
            if before_step != step or after_step != step:
                raise RuntimeError("validation changed TrainState.step")
            after_state_ids = {
                "params": tuple(
                    (jax.tree_util.keystr(path), id(value))
                    for path, value in jax.tree_util.tree_leaves_with_path(state.params)
                ),
                "optimizer": tuple(
                    (jax.tree_util.keystr(path), id(value))
                    for path, value in jax.tree_util.tree_leaves_with_path(state.opt_state)
                ),
                "ema": (
                    tuple(
                        (jax.tree_util.keystr(path), id(value))
                        for path, value in jax.tree_util.tree_leaves_with_path(
                            state.ema_params)
                    ) if state.ema_params is not None else None
                ),
            }
            if before_state_ids != after_state_ids:
                raise RuntimeError("validation replaced a parameter or optimizer leaf")
            payload = {
                "schema_version": 1,
                "event": "validation_loss",
                "metric": plan["metric"],
                "record_prefix": plan["record_prefix"],
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "wall_time_s": time.monotonic() - started,
                "split": "validation",
                "train_state_step": step,
                "checkpoint_saved": bool(checkpoint_info.get("checkpoint_saved")),
                "checkpoint_info": checkpoint_info,
                "samples_expected": plan["sample_count"],
                **result,
                "validation_plan_sha256": validation_plan_hashes[kind],
                "view_sha256": meta.get("view_sha256"),
                "norm_stats_sha256": _sha_file(
                    train_view / "assets/so101/norm_stats.json"),
                "norm_stats_source": "train_only",
                "weights_source": "TrainState.params",
                "validation_seed": args.validation_seed,
                "batch_key_sha256": plan["batch_key_sha256"],
                "noise_sha256": plan["noise_sha256"],
                "flow_time_sha256": plan["flow_time_sha256"],
                "train_false": True,
                "loss": "flow_matching_mse",
                "loss_definition": validation.LOSS_DEFINITION,
                "gradient_updates": 0,
                "state_tree_unchanged": True,
            }
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(path)
            print(json.dumps(payload, sort_keys=True), flush=True)
            return payload

        original_save_state = train_module._checkpoints.save_state
        original_restore_state = train_module._checkpoints.restore_state

        def scheduled_save_state(manager, state, data_loader, step):
            actual_step = int(jax.device_get(state.step))
            checkpoint_info = {
                "checkpoint_saved": False,
                "train_state_step": actual_step,
                "checkpoint_metadata_step": None,
                "retained_steps": list(saw_checkpoint_steps),
                "retain_all_saved_steps": True,
                "full_train_state": True,
            }
            benchmark_save = (
                args.validation_benchmark and actual_step == int(args.steps))
            probe_final_save = (
                args.checkpoint_probe and args.mode in {"probe", "resume"}
                and actual_step == int(args.steps))
            if actual_step in checkpoint_steps or benchmark_save or probe_final_save:
                original_save_state(manager, state, data_loader, actual_step)
                manager.wait_until_finished()
                metadata_step = int(manager.latest_step())
                if metadata_step != actual_step:
                    raise RuntimeError(
                        "checkpoint manager step differs from TrainState.step: "
                        f"{metadata_step} != {actual_step}")
                saw_checkpoint_steps[:] = [int(value) for value in manager.all_steps()]
                expected_retained = [
                    value for value in checkpoint_steps if value <= actual_step]
                missing_retained = sorted(
                    set(expected_retained) - set(saw_checkpoint_steps))
                if missing_retained:
                    raise RuntimeError(
                        "checkpoint retention lost registered steps: "
                        + ",".join(str(value) for value in missing_retained))
                checkpoint_info.update({
                    "checkpoint_saved": True,
                    "checkpoint_metadata_step": metadata_step,
                    "retained_steps": list(saw_checkpoint_steps),
                })
                (run_root / f"checkpoint_steps_{actual_step}.json").write_text(
                    json.dumps(checkpoint_info, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            production_due = (
                args.mode in {"initial", "resume"}
                and actual_step % int(args.validation_every) == 0
            )
            benchmark_due = (
                args.validation_benchmark and actual_step == int(args.steps))
            if checkpoint_info.get("checkpoint_saved") and (
                actual_step in checkpoint_steps or benchmark_due):
                evaluate_validation(state, "full", checkpoint_info)
            if production_due or benchmark_due:
                evaluate_validation(state, "probe", checkpoint_info)

        def restore_state_with_validation(manager, state, data_loader, step=None):
            restored = original_restore_state(manager, state, data_loader, step)
            actual_step = int(jax.device_get(restored.step))
            archived = validation.archive_future_validation_records(
                run_root,
                actual_step,
                archive_name=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            )
            if archived:
                print(json.dumps({
                    "event": "archived_future_validation_records",
                    "train_state_step": actual_step,
                    "count": len(archived),
                }), flush=True)
            saw_checkpoint_steps[:] = [int(value) for value in manager.all_steps()]
            if actual_step in checkpoint_steps:
                manager.wait_until_finished()
                checkpoint_info = {
                    "checkpoint_saved": True,
                    "train_state_step": actual_step,
                    "checkpoint_metadata_step": int(manager.latest_step()),
                    "retained_steps": list(saw_checkpoint_steps),
                    "retain_all_saved_steps": True,
                    "full_train_state": True,
                }
                evaluate_validation(restored, "full", checkpoint_info)
                if actual_step % int(args.validation_every) == 0:
                    evaluate_validation(restored, "probe", checkpoint_info)
            return restored

        train_module._checkpoints.save_state = scheduled_save_state
        train_module._checkpoints.restore_state = restore_state_with_validation

    original_loader = train_module._data_loader.create_data_loader
    original_stdout = sys.stdout
    train_module._data_loader.create_data_loader = openpi_so101.create_so101_data_loader
    print(json.dumps({"phase": phase, "mode": args.mode, "train_entry": str(train_path),
                      "checkpoint_dir": str(config.checkpoint_dir),
                      "steps": config.num_train_steps, "batch_size": config.batch_size,
                      "lr_schedule": snapshot["lr_schedule"],
                      "lora": [config.model.paligemma_variant,
                               config.model.action_expert_variant]}), flush=True)
    try:
        sys.stdout = _FiniteOutputGuard(
            original_stdout,
            require_initial_update=bool(
                args.require_clean_data or args.require_validation_loss),
            step_offset=1 if args.require_validation_loss else 0,
        )
        train_module.main(config)
    finally:
        train_module._data_loader.create_data_loader = original_loader
        if args.require_validation_loss:
            train_module._checkpoints.save_state = original_save_state
            train_module._checkpoints.restore_state = original_restore_state
        sys.stdout = original_stdout
    (run_root / f"done_{args.mode}.json").write_text(
        json.dumps({"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                    "mode": args.mode, "checkpoint_dir": str(config.checkpoint_dir)},
                   indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
