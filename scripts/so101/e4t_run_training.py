#!/usr/bin/env python3
"""Resource-owned E4-T OpenPI launcher.

This starts one non-auto-removed container with an exact name and full registry
entry.  GPU admission is based on measured free memory for that physical device,
never utilization. The selected device UUID is rechecked inside the container.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import clean60_contracts as clean60
from resource_policy import memory_plan, sampled_margin_problems

REPO = Path(__file__).resolve().parents[2]
PROJECT = "pi05-reproduction"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(argv: list[str], *, timeout: int = 120, check: bool = True) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def sha256_path(path: Path) -> str:
    """Hash a file or a directory tree without depending on external manifests."""
    path = Path(path)
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError("base checkpoint contains no readable files")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_hash = hashlib.sha256()
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                file_hash.update(block)
        digest.update(file_hash.digest())
    return digest.hexdigest()


def gpu_state(index: int) -> dict:
    raw = run(["nvidia-smi", "-i", str(index),
               "--query-gpu=uuid,memory.total,memory.used,memory.free",
               "--format=csv,noheader,nounits"])
    gpu_uuid, total, used, free = [part.strip() for part in raw.split(",", 4)]
    return {"index": index, "uuid": gpu_uuid, "memory_total_mib": int(total),
            "memory_used_mib": int(used), "memory_free_mib": int(free)}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_artifact_paths(run_dir: Path, mode: str, run_id: str) -> dict[str, Path]:
    """Give each container its own logs while preserving the training-step log contract."""
    stdout = (run_dir / "train_initial.log" if mode == "initial" else
              run_dir / f"train_{mode}_{run_id}.log")
    return {
        "stdout_log": stdout,
        "gpu_log": run_dir / f"gpu_{mode}_{run_id}.txt",
        "run_payload": run_dir / f"run_{mode}_{run_id}.json",
    }


def add_freeze_vision_argument(parser: argparse.ArgumentParser) -> None:
    """Share the real CLI definition with CPU contract tests."""
    parser.add_argument("--freeze-vision-encoder", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="freeze the SigLIP vision encoder (default: train it)")


def add_model_scope_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--paligemma-variant", default="gemma_2b_lora")
    parser.add_argument("--action-expert-variant", default="gemma_300m")
    add_freeze_vision_argument(parser)


def build_training_command(args, container_run: str, container_view: str) -> list[str]:
    """Build the exact in-container training command and its scope flags."""
    command = ["-u", "/work/scripts/so101/train_so101.py", "--mode", args.mode,
               "--phase", "E4-T", "--run-root", container_run, "--train-view", container_view,
               "--base-checkpoint", "/base", "--exp-name", args.exp_name,
               "--state-mode", args.state_mode, "--batch-size", str(args.batch_size),
               "--steps", str(args.steps), "--save-interval", str(args.save_interval),
               "--data-loader-workers", str(args.data_loader_workers),
               "--lr-warmup-steps", str(args.lr_warmup_steps), "--peak-lr", str(args.peak_lr),
               "--decay-steps", str(args.decay_steps), "--decay-lr", str(args.decay_lr),
               "--seed", str(args.seed), "--action-horizon", str(args.action_horizon),
               "--optimizer", args.optimizer,
               "--gradient-clip-norm", str(args.gradient_clip_norm),
               "--dtype", args.dtype,
               "--fsdp-devices", str(args.fsdp_devices),
               "--paligemma-variant", args.paligemma_variant,
               "--action-expert-variant", args.action_expert_variant,
               "--freeze-vision-encoder" if args.freeze_vision_encoder
               else "--no-freeze-vision-encoder",
               "--expected-groups", str(args.expected_groups)]
    if args.training_config_sha256:
        command += ["--registered-config-sha256", args.training_config_sha256]
    if getattr(args, "training_config", None):
        command += ["--registered-training-config", "/work/" + str(
            Path(args.training_config).resolve().relative_to(REPO))]
    if getattr(args, "require_clean_data", False):
        command += ["--require-clean-data"]
    if getattr(args, "require_action_norm_audit", False):
        command += ["--require-action-norm-audit"]
    if getattr(args, "registered_base_sha256", None):
        command += ["--registered-base-sha256", args.registered_base_sha256]
    if args.ema_decay is not None:
        command += ["--ema-decay", str(args.ema_decay)]
    if args.run_id:
        command += ["--run-id", args.run_id]
    if args.mode == "resume":
        command += ["--resume-step", str(args.resume_step)]
    if args.sampling_plan:
        command += ["--sampling-plan", "/work/" + str(args.sampling_plan.resolve().relative_to(REPO))]
    if args.data_admission:
        command += ["--data-admission", "/work/" + str(args.data_admission.resolve().relative_to(REPO))]
    command += ["--dataset-label", args.dataset_label,
                "--expected-total-episodes", str(args.expected_total_episodes),
                "--expected-train-episodes", str(args.expected_train_episodes),
                "--expected-validation-episodes", str(args.expected_validation_episodes)]
    return command


def registry_call(registry: Path, argv: list[str], timeout: int = 86400) -> dict:
    result = subprocess.run([sys.executable, str(REPO / "scripts/so101/e4t_resources.py"),
                             "--registry", str(registry), *argv], capture_output=True, text=True,
                            timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return json.loads(result.stdout)


def _reap_process(process, label: str) -> list[str]:
    """Boundedly stop and reap one subprocess owned by this launcher."""
    if process is None:
        return []
    errors: list[str] = []
    try:
        process.wait(timeout=5)
        return errors
    except subprocess.TimeoutExpired:
        pass
    except Exception as error:
        errors.append(f"{label} wait failed: {error}")

    for action in ("terminate", "kill"):
        try:
            getattr(process, action)()
        except Exception as error:
            errors.append(f"{label} {action} failed: {error}")
        try:
            process.wait(timeout=5)
            return errors
        except subprocess.TimeoutExpired:
            errors.append(f"{label} still running after {action}")
        except Exception as error:
            errors.append(f"{label} wait after {action} failed: {error}")
    return errors


def _cleanup_owned_run(registry: Path, entry_id: str, monitor, stream) -> list[str]:
    """Reap owned helper processes, then stop only this registered container."""
    errors = _reap_process(stream, "docker log stream")
    errors.extend(_reap_process(monitor, "GPU monitor"))
    try:
        registry_call(registry, ["stop", "--entry-id", entry_id,
                                 "--confirm", "STOP-REGISTERED-EXACT-ENTRIES"])
    except Exception as error:
        errors.append(f"controlled container cleanup failed: {error}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "probe", "initial", "resume", "eval"),
                        required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-sha256", default=None)
    parser.add_argument("--image", default="openpi-server:latest")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--state-mode", choices=("off", "on"), default="on")
    parser.add_argument("--gpu-indices", required=True,
                        help="comma-separated physical device ids for the one JAX process "
                             "(FSDP shards across fsdp_devices, the rest are data parallel)")
    parser.add_argument("--fsdp-devices", type=int, default=1)
    add_model_scope_arguments(parser)
    parser.add_argument("--expected-groups", type=int, default=2)
    parser.add_argument("--required-memory-mib", type=int, required=True)
    parser.add_argument("--memory-margin-mib", type=int, default=1500)
    parser.add_argument("--eval-checkpoint", default=None,
                        help="absolute path of a saved checkpoint directory, mounted read-only")
    parser.add_argument("--eval-run-config", type=Path, default=None)
    parser.add_argument("--eval-windows", type=Path, default=None)
    parser.add_argument("--eval-split", default=None, choices=("train", "validation"))
    parser.add_argument("--eval-label", default=None)
    parser.add_argument("--eval-train-state-step", type=int, default=None)
    parser.add_argument("--eval-assets-root", type=Path, default=None)
    parser.add_argument("--eval-max-records", type=int, default=0,
                        help="cap the windows scored per checkpoint; 0 keeps the frozen window set")
    parser.add_argument("--eval-anchor-events", action="store_true",
                        help="run the supplementary per-transition timing pass into its own report")
    parser.add_argument("--eval-dense-stride", type=int, default=0,
                        help="superset sweep for the best-checkpoint dense curve")
    parser.add_argument("--noise-copies", type=int, default=2)
    parser.add_argument("--jax-memory-fraction", type=float, default=None)
    parser.add_argument("--jax-preallocate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--peak-lr", type=float, default=5e-5)
    parser.add_argument("--decay-steps", type=int, required=True)
    parser.add_argument("--decay-lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--optimizer", choices=("AdamW",), default="AdamW")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--data-loader-workers", type=int, default=0)
    parser.add_argument("--sampling-plan", type=Path, default=None)
    parser.add_argument("--data-admission", type=Path, default=None)
    parser.add_argument("--training-config-sha256", default=None)
    parser.add_argument("--training-config", type=Path, default=None)
    parser.add_argument("--require-clean-data", action="store_true")
    parser.add_argument("--require-action-norm-audit", action="store_true")
    parser.add_argument("--ready-manifest", type=Path, default=None)
    parser.add_argument("--require-ready-manifest", action="store_true")
    parser.add_argument("--registered-base-sha256", default=None)
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--expected-total-episodes", type=int, default=None)
    parser.add_argument("--expected-train-episodes", type=int, default=None)
    parser.add_argument("--expected-validation-episodes", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-step", type=int, default=None,
                        help="authoritative TrainState.step restored by a resume run")
    parser.add_argument("--timeout-seconds", type=int, default=8 * 3600)
    args = parser.parse_args()

    if args.mode == "probe":
        args.steps = 1
        args.save_interval = 1
    if args.mode in {"initial", "resume"}:
        missing = [name for name, value in (("--dataset-label", args.dataset_label),
                                            ("--expected-total-episodes", args.expected_total_episodes),
                                            ("--expected-train-episodes", args.expected_train_episodes),
                                            ("--expected-validation-episodes", args.expected_validation_episodes))
                   if value is None]
        if missing:
            raise RuntimeError(f"{args.mode} training must bind its data version explicitly: "
                               f"missing {', '.join(missing)}")
    run_dir = args.run_dir.expanduser().resolve()
    if args.mode == "initial" and (run_dir / "config_initial.json").exists():
        raise RuntimeError("refusing to overwrite an existing initial training run")
    if args.mode == "resume":
        if args.resume_step is None or args.resume_step < 1:
            raise RuntimeError("resume requires the positive authoritative --resume-step")
        if not (run_dir / "config_initial.json").is_file():
            raise RuntimeError("resume requires the existing run's config_initial.json")
    train_view = args.train_view.expanduser().resolve()
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    tokenizer = args.tokenizer.expanduser().resolve()
    if not run_dir.is_relative_to(REPO) or not train_view.is_relative_to(REPO) or not base_checkpoint.is_absolute():
        raise RuntimeError("training paths must be repo-relative mounts")
    run_id = args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    checkpoint_sha256 = sha256_path(base_checkpoint)
    tokenizer_sha256 = sha256_path(tokenizer)
    if args.expected_tokenizer_sha256 and tokenizer_sha256 != args.expected_tokenizer_sha256:
        raise RuntimeError("tokenizer hash differs from the registered runtime binding")
    if args.registered_base_sha256 and checkpoint_sha256 != args.registered_base_sha256:
        raise RuntimeError("base checkpoint hash differs from the probe-bound hash")
    training_config = (args.training_config.expanduser().resolve()
                       if args.training_config else None)
    if args.training_config_sha256:
        if training_config is None or not training_config.is_file():
            raise RuntimeError("training configuration path is required for hash verification")
        if hashlib.sha256(training_config.read_bytes()).hexdigest() != args.training_config_sha256:
            raise RuntimeError("training configuration hash mismatch")
    if args.require_clean_data:
        if training_config is None or not training_config.is_file():
            raise RuntimeError("clean data training requires the registered training configuration")
        registered = json.loads(training_config.read_text(encoding="utf-8"))
        config_problems = clean60.training_config_problems(registered)
        if config_problems:
            raise RuntimeError("registered training configuration failed: "
                               + "; ".join(config_problems))
        if args.mode in {"initial", "resume"} and (
                args.steps != registered["optimisation"]["planned_updates"]
                or args.batch_size != registered["optimisation"]["primary_batch_size"]):
            raise RuntimeError("formal clean60 training must keep global batch 8 and 25000 updates")
    norm_audit = None
    norm_audit_path = train_view / "action_norm_audit.json"
    norm_audit_sha256 = None
    if args.require_clean_data:
        meta = json.loads((train_view / "meta.json").read_text(encoding="utf-8"))
        split = json.loads((train_view / "splits.json").read_text(encoding="utf-8"))
        binding_path = train_view / "protocol_binding.json"
        binding = (json.loads(binding_path.read_text(encoding="utf-8"))
                   if binding_path.is_file() else None)
        problems = clean60.view_contract_problems(
            meta, protocol_binding=binding, require_clean=True)
        admission_path = (args.data_admission.expanduser().resolve()
                          if args.data_admission else train_view / "e4t_admission.json")
        if not admission_path.is_file():
            problems.append("clean data admission manifest is missing")
            admission = {}
        else:
            admission = json.loads(admission_path.read_text(encoding="utf-8"))
            problems.extend(clean60.admission_contract_problems(
                admission, meta=meta, split=split,
                train_stats_sha256=hashlib.sha256(
                    (train_view / "train_stats.json").read_bytes()).hexdigest()))
        if problems:
            raise RuntimeError("clean data contract failed: " + "; ".join(problems))
    if args.require_action_norm_audit:
        if not norm_audit_path.is_file():
            raise RuntimeError("training requires action_norm_audit.json")
        norm_audit = json.loads(norm_audit_path.read_text(encoding="utf-8"))
        norm_audit_sha256 = hashlib.sha256(norm_audit_path.read_bytes()).hexdigest()
        meta = json.loads((train_view / "meta.json").read_text(encoding="utf-8"))
        problems = clean60.norm_audit_problems(
            norm_audit,
            view_sha256=meta.get("view_sha256", ""),
            norm_stats_sha256=hashlib.sha256(
                (train_view / "assets/so101/norm_stats.json").read_bytes()).hexdigest(),
            base_checkpoint_sha256=checkpoint_sha256,
            training_config_sha256=args.training_config_sha256 or "",
        )
        if problems:
            raise RuntimeError("normalization audit contract failed: " + "; ".join(problems))
    if args.require_ready_manifest:
        if not args.ready_manifest or not args.ready_manifest.is_file():
            raise RuntimeError("formal clean60 training requires a ready manifest")
        ready = json.loads(args.ready_manifest.read_text(encoding="utf-8"))
        meta = json.loads((train_view / "meta.json").read_text(encoding="utf-8"))
        problems = clean60.ready_manifest_problems(
            ready,
            view_sha256=meta.get("view_sha256", ""),
            norm_stats_sha256=hashlib.sha256(
                (train_view / "assets/so101/norm_stats.json").read_bytes()).hexdigest(),
            norm_audit_sha256=norm_audit_sha256 or "",
            training_config_sha256=args.training_config_sha256 or "",
            base_checkpoint_sha256=checkpoint_sha256,
            scene_audit_sha256=hashlib.sha256(
                (train_view / "scene_audit.json").read_bytes()).hexdigest(),
            data_audit_sha256=hashlib.sha256(
                (train_view / "data_audit.json").read_bytes()).hexdigest(),
            stats_recompute_sha256=hashlib.sha256(
                (train_view / "stats_recompute.json").read_bytes()).hexdigest(),
        )
        if problems:
            raise RuntimeError("ready manifest failed: " + "; ".join(problems))
    exact_name = f"so101-e4t-{args.mode}-{run_id}"
    cache = run_dir / "openpi_cache/big_vision"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "paligemma_tokenizer.model"
    if not destination.is_file() or destination.stat().st_size == 0:
        subprocess.run(["cp", "-n", str(tokenizer), str(destination)], check=False)
    if not destination.is_file():
        raise RuntimeError("tokenizer cache is not populated")

    indices = [int(value) for value in args.gpu_indices.split(",") if value.strip() != ""]
    if not indices or len(set(indices)) != len(indices):
        raise RuntimeError(f"--gpu-indices must be distinct device ids, got {args.gpu_indices!r}")
    if not 1 <= args.fsdp_devices <= len(indices) or len(indices) % args.fsdp_devices:
        raise RuntimeError(f"fsdp_devices={args.fsdp_devices} must divide {len(indices)} devices")
    devices = [gpu_state(index) for index in indices]
    for device in devices:
        if device["memory_free_mib"] < args.required_memory_mib + args.memory_margin_mib:
            raise RuntimeError(
                f"GPU {device['index']} has {device['memory_free_mib']} MiB free; "
                f"requires {args.required_memory_mib + args.memory_margin_mib} MiB including margin")
    memory = memory_plan(
        devices, demand_mib=args.required_memory_mib,
        margin_mib=args.memory_margin_mib,
        explicit_fraction=args.jax_memory_fraction)
    fraction = memory["jax_memory_fraction"]

    registry = run_dir / "resource_registry.json"
    if registry.is_file():
        existing = json.loads(registry.read_text(encoding="utf-8"))
        if any(item["exact_name"] == exact_name for item in existing.get("entries", [])):
            raise RuntimeError("exact run name already registered")

    container_run = "/work/" + str(run_dir.relative_to(REPO))
    container_view = "/work/" + str(train_view.relative_to(REPO))
    if args.mode == "eval":
        for required in ("eval_checkpoint", "eval_run_config", "eval_windows",
                         "eval_split", "eval_label", "eval_train_state_step"):
            if getattr(args, required) in (None, ""):
                raise RuntimeError(f"--{required.replace('_', '-')} is required in eval mode")
        checkpoint = Path(args.eval_checkpoint).expanduser().resolve()
        if not checkpoint.is_dir():
            raise RuntimeError(f"eval checkpoint does not exist: {checkpoint}")
        command = ["-u", "/work/scripts/so101/evaluate_e4t_openloop.py", "--checkpoint", "/ckpt",
                   "--run-config", "/work/" + str(Path(args.eval_run_config).resolve().relative_to(REPO)),
                   "--view", container_view,
                   "--windows", "/work/" + str(Path(args.eval_windows).resolve().relative_to(REPO)),
                   "--output", f"{container_run}/evaluation/{args.eval_label}",
                   "--split", args.eval_split, "--label", args.eval_label,
                   "--train-state-step", str(args.eval_train_state_step),
                   "--noise-copies", str(args.noise_copies)]
        if args.eval_assets_root:
            command += ["--assets-root", "/work/" + str(
                Path(args.eval_assets_root).resolve().relative_to(REPO))]
        if args.eval_max_records:
            command += ["--max-records", str(args.eval_max_records)]
        if args.eval_anchor_events:
            command += ["--anchor-events"]
        if args.eval_dense_stride:
            command += ["--dense-stride", str(args.eval_dense_stride)]
    else:
        command = build_training_command(args, container_run, container_view)

    env_args = [
        "-e", f"HOME={container_run}/home",
        "-e", f"OPENPI_DATA_HOME={container_run}/openpi_cache",
        "-e", "PYTHONPATH=/work/scripts/so101",
        "-e", "JAX_PLATFORMS=cuda",
        "-e", "XLA_FORCE_COMPILATION_CACHE=1",
        "-e", "JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache",
        "-e", f"XLA_PYTHON_CLIENT_PREALLOCATE={str(args.jax_preallocate).lower()}",
        "-e", f"XLA_PYTHON_CLIENT_MEM_FRACTION={fraction:.6f}",
        "-e", f"SO101_TRAIN_VIEW={container_view}",
    ]
    if args.mode != "probe":
        env_args += ["-e", f"JAX_COMPILATION_CACHE_DIR={container_run}/jax_cache"]
    launch_args = ["--entrypoint", "python3", "--shm-size=4g", "-v", f"{REPO}:/work",
                   "-v", f"{base_checkpoint}:/base:ro", "-w", "/work"] + env_args
    if args.mode == "eval":
        launch_args = launch_args[:5] + ["-v", f"{checkpoint}:/ckpt:ro"] + launch_args[5:]
    allocated = registry_call(registry, [
        "allocate", "--run-id", run_id, "--exact-name", exact_name, "--image", args.image,
        "--command-json", json.dumps(command), "--docker-arg-json", json.dumps(launch_args),
        "--gpu-index", ",".join(str(value) for value in indices)])
    artifacts = run_artifact_paths(run_dir, args.mode, run_id)
    gpu_log = artifacts["gpu_log"]
    stdout_log = artifacts["stdout_log"]
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    monitor = stream = stdout_handle = None
    started = time.time()
    started_at = now()
    status = 1
    try:
        inside_uuid = run(["docker", "exec", allocated["container_id"], "nvidia-smi",
                           "--query-gpu=uuid", "--format=csv,noheader"])
        inside_uuids = sorted({line.strip() for line in inside_uuid.splitlines() if line.strip()})
        registered_uuids = sorted(device["uuid"] for device in devices)
        if inside_uuids != registered_uuids:
            raise RuntimeError(f"container GPU mismatch: registered {registered_uuids}, "
                               f"visible {inside_uuids}")
        monitor = subprocess.Popen(
            ["bash", "-c",
             f'while docker container inspect {shlex.quote(allocated["container_id"])} >/dev/null 2>&1; do '
             f'nvidia-smi -i {",".join(str(value) for value in indices)} '
             f'--query-gpu=index,uuid,memory.used --format=csv,noheader,nounits '
             f'>> {shlex.quote(str(gpu_log))} 2>/dev/null; sleep 1; done'])
        stdout_handle = stdout_log.open("w", encoding="utf-8")
        stream = subprocess.Popen(["docker", "logs", "-f", allocated["container_id"]],
                                  stdout=stdout_handle, stderr=subprocess.STDOUT)
        result = registry_call(registry, ["wait", "--entry-id", allocated["registry_entry_id"],
                                          "--timeout", str(args.timeout_seconds)], timeout=args.timeout_seconds + 60)
        status = int(result["exit_code"])
    finally:
        cleanup_errors = _cleanup_owned_run(
            registry, allocated["registry_entry_id"], monitor, stream)
        if stdout_handle is not None:
            try:
                stdout_handle.close()
            except Exception as error:
                cleanup_errors.append(f"log file close failed: {error}")
        for cleanup_error in cleanup_errors:
            print(cleanup_error, file=sys.stderr)

    peaks: dict[str, int] = {}
    if gpu_log.is_file():
        for line in gpu_log.read_text(encoding="utf-8").splitlines():
            parts = [value.strip() for value in line.split(",")]
            if len(parts) == 3 and parts[0].isdigit() and parts[2].isdigit():
                peaks[parts[0]] = max(peaks.get(parts[0], 0), int(parts[2]))
    payload = {
        "mode": args.mode, "run_id": run_id, "exact_name": exact_name,
        "registry_entry_id": allocated["registry_entry_id"],
        "container_id": allocated["container_id"], "exit_code": status,
        "owner": allocated["owner"],
        "started_at": started_at, "finished_at": now(),
        "wall_time_s": time.time() - started,
        "gpu": {
            "requested_indices": indices,
            "fsdp_devices": args.fsdp_devices,
            "data_parallel_replicas": len(indices) // args.fsdp_devices,
            "devices": [{
                "index": device["index"], "uuid": device["uuid"],
                "memory_total_mib": device["memory_total_mib"],
                "memory_used_mib_at_admission": device["memory_used_mib"],
                "memory_free_mib_at_admission": device["memory_free_mib"],
                "peak_mib_sampled": peaks.get(str(device["index"]), 0),
            } for device in devices],
            "utilisation_is_not_an_admission_gate": True,
            "jax_memory_fraction": round(fraction, 6),
            "memory_margin_mib": args.memory_margin_mib,
            "peak_mib_sampled_max": max(peaks.values(), default=0),
            "memory_plan": memory,
        },
        "batch_size": args.batch_size, "steps": args.steps, "seed": args.seed,
        "action_horizon": args.action_horizon, "optimizer": args.optimizer,
        "gradient_clip_norm": args.gradient_clip_norm, "ema_decay": args.ema_decay,
        "dtype": args.dtype, "jax_preallocate": args.jax_preallocate,
        "training_config_sha256": args.training_config_sha256,
        "base_checkpoint_sha256": checkpoint_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "image": args.image,
        "norm_audit_sha256": norm_audit_sha256,
        "training_config_sha256": args.training_config_sha256,
        "ready_manifest_sha256": (
            hashlib.sha256(args.ready_manifest.read_bytes()).hexdigest()
            if args.ready_manifest and args.ready_manifest.is_file() else None),
        "model_scope": {
            "paligemma_variant": args.paligemma_variant,
            "action_expert_variant": args.action_expert_variant,
            "freeze_vision_encoder": args.freeze_vision_encoder,
        },
        "command": command, "log": stdout_log.name, "gpu_log": gpu_log.name,
        "stop_method": "registered exact container wait plus controlled stop",
    }
    margin_problems = sampled_margin_problems(
        payload["gpu"]["devices"], margin_mib=args.memory_margin_mib)
    payload["gpu"]["sampled_margin_problems"] = margin_problems
    if margin_problems and status == 0:
        status = 99
        payload["exit_code"] = status
        payload["failure"] = margin_problems
    write_json(artifacts["run_payload"], payload)
    # Retain the most-recent legacy alias for scripts that only need the latest launch.
    write_json(run_dir / f"run_{args.mode}.json", payload)
    print(json.dumps(payload, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
