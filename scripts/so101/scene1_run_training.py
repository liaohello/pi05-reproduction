#!/usr/bin/env python3
"""Resource-owned scene1 training launcher.

One registered openpi-server container per probe/formal run.  GPU admission is
based on measured free memory, the JAX pool is capped from the tightest card,
the container-visible UUID set is rechecked, and peak memory is sampled until
the run exits.  Other tenants are never stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("PI05_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(REPO / "scripts" / "so101"))
from resource_policy import memory_plan, sampled_margin_problems  # noqa: E402


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gpu_state(index):
    raw = subprocess.run(
        ["nvidia-smi", "-i", str(index),
         "--query-gpu=uuid,memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60, check=True).stdout
    uuid, total, used, free = [part.strip() for part in raw.strip().split(",", 3)]
    return {"index": index, "uuid": uuid, "memory_total_mib": int(total),
            "memory_used_mib": int(used), "memory_free_mib": int(free)}


def registry(registry_path, argv, timeout=86400):
    result = subprocess.run([sys.executable, str(REPO / "scripts/so101/e4t_resources.py"),
                             "--registry", str(registry_path), *argv],
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return json.loads(result.stdout)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_training_config_problems(config, args, train_view: Path) -> list[str]:
    """Check the registered scene1 config against the concrete launch arguments."""
    problems = []
    optimisation = config.get("optimisation", {})
    devices = config.get("devices", {})
    interface = config.get("interface_contract", {})
    scope = config.get("model_scope", {})
    checkpointing = config.get("checkpointing", {})
    data = config.get("data_binding", {})
    runtime = config.get("runtime_binding", {})
    expected = {
        "optimisation.global_batch_size": (
            optimisation.get("global_batch_size"), args.batch_size),
        "optimisation.warmup_updates": (
            optimisation.get("warmup_updates"), args.lr_warmup_steps),
        "optimisation.peak_lr": (optimisation.get("peak_lr"), args.peak_lr),
        "optimisation.cosine_decay_updates": (
            optimisation.get("cosine_decay_updates"), args.decay_steps),
        "optimisation.final_lr": (optimisation.get("final_lr"), args.decay_lr),
        "optimisation.ema_decay": (
            optimisation.get("ema_decay"), args.ema_decay),
        "optimisation.gradient_clip_norm": (
            optimisation.get("gradient_clip_norm"), args.gradient_clip_norm),
        "devices.fsdp_devices": (devices.get("fsdp_devices"), args.fsdp_devices),
        "devices.minimum_memory_margin_mib": (
            devices.get("minimum_memory_margin_mib"), args.memory_margin_mib),
        "interface_contract.action_horizon": (
            interface.get("action_horizon"), args.action_horizon),
        "interface_contract.state_mode": (
            interface.get("state_mode"), args.state_mode),
        "runtime_binding.base_checkpoint_sha256": (
            runtime.get("base_checkpoint_sha256"), args.registered_base_sha256),
    }
    if not (args.checkpoint_probe or args.validation_benchmark):
        expected["optimisation.updates"] = (optimisation.get("updates"), args.steps)
    if "checkpoint_steps" not in checkpointing:
        expected["checkpointing.save_every_updates"] = (
            checkpointing.get("save_every_updates"), args.save_interval)
    for name, (registered, actual) in expected.items():
        if registered != actual:
            problems.append(f"{name}: registered={registered!r}, actual={actual!r}")
    if scope.get("vision_encoder") != (
        "frozen" if args.freeze_vision_encoder else "full"):
        problems.append("vision scope differs from the registered config")
    if scope.get("action_expert") != "full":
        problems.append("action expert must use the registered full scope")
    if scope.get("language_backbone") != "frozen":
        problems.append("language backbone must remain frozen")
    if (not args.checkpoint_probe and not args.validation_benchmark
            and checkpointing.get("checkpoint_steps", [])[-1:] != [args.steps]):
        problems.append("final checkpoint step differs from the training target")
    if config.get("validation_loss") and not args.require_validation_loss:
        problems.append("registered validation loss requires --require-validation-loss")
    if args.require_validation_loss:
        validation = config.get("validation_loss") or {}
        probe = validation.get("probe") or {}
        full = validation.get("full") or {}
        if probe.get("every_updates") != args.validation_every:
            problems.append("validation interval differs from the registered config")
        if validation.get("seed") != args.validation_seed:
            problems.append("validation seed differs from the registered config")
        if validation.get("batch_size") != args.validation_batch_size:
            problems.append("validation batch size differs from the registered config")
        if probe.get("samples") != 256:
            problems.append("probe validation must use 256 samples")
        if probe.get("metric") != "val_loss_probe" or probe.get("record_prefix") != "val_loss_probe":
            problems.append("probe validation metric identity differs")
        if probe.get("requires_checkpoint") is not False:
            problems.append("probe validation must not require a checkpoint")
        if full.get("samples") != 8300:
            problems.append("full validation must use 8300 samples")
        if full.get("metric") != "val_loss_full" or full.get("record_prefix") != "val_loss_full":
            problems.append("full validation metric identity differs")
        if full.get("requires_checkpoint") is not True:
            problems.append("full validation must require a completed checkpoint")
        if full.get("checkpoint_steps") != checkpointing.get("checkpoint_steps"):
            problems.append("full validation checkpoint schedule differs")
        if checkpointing.get("retain_all_saved_steps") is not True:
            problems.append("all registered checkpoints must be retained")
    meta_path = train_view / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if data.get("view_sha256") != meta.get("view_sha256"):
            problems.append("registered view hash differs from training view")
        norm_path = train_view / "assets/so101/norm_stats.json"
        if norm_path.is_file() and data.get("norm_stats_sha256") != sha256_file(norm_path):
            problems.append("registered norm hash differs from training view")
        if data.get("train_episodes") != sum(
                row.get("split") == "train" for row in meta.get("episodes", [])):
            problems.append("registered train episode count differs")
        if data.get("validation_episodes") != sum(
                row.get("split") == "validation" for row in meta.get("episodes", [])):
            problems.append("registered validation episode count differs")
    else:
        problems.append("training view meta.json is missing")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probe", "smoke", "initial", "resume"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--image", default="openpi-server@sha256:405af52051682d5e13169d1de1523fd0adc116eceae4fc314cadfacd45b1ad52")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--label", default="train")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--gpu-indices", required=True)
    parser.add_argument("--fsdp-devices", type=int, default=4)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--data-loader-workers", type=int, default=8)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--peak-lr", type=float, default=5e-5)
    parser.add_argument("--decay-steps", type=int, required=True)
    parser.add_argument("--decay-lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--state-mode", choices=("off", "on"), default="on")
    parser.add_argument("--freeze-vision-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--training-config-sha256", required=True)
    parser.add_argument("--source-root", type=Path, default=None,
                        help="independent Scene1 source directory containing the trainer modules")
    parser.add_argument("--registered-base-sha256", required=True)
    parser.add_argument("--dataset-label", default="scene1_single_random_v1")
    parser.add_argument("--expected-total-episodes", type=int, default=64)
    parser.add_argument("--expected-train-episodes", type=int, default=52)
    parser.add_argument("--expected-validation-episodes", type=int, default=12)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--validation-batch-size", type=int, default=16)
    parser.add_argument("--require-validation-loss", action="store_true")
    parser.add_argument("--validation-benchmark", action="store_true")
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--strict-training-config", action="store_true")
    parser.add_argument("--checkpoint-probe", action="store_true")
    parser.add_argument("--require-action-norm-audit", action="store_true")
    parser.add_argument("--required-memory-mib", type=int, default=0)
    parser.add_argument("--memory-margin-mib", type=int, default=100)
    parser.add_argument("--jax-memory-fraction", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=30 * 3600)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    train_view = args.train_view.expanduser().resolve()
    base = args.base_checkpoint.expanduser().resolve()
    tokenizer = args.tokenizer.expanduser().resolve()
    training_config = args.training_config.expanduser().resolve()
    source_root = (args.source_root.expanduser().resolve()
                   if args.source_root is not None else REPO / "scripts" / "so101")
    if not (train_view / "meta.json").is_file():
        raise SystemExit(f"training view missing meta.json: {train_view}")
    if not (source_root / "scene1_train_so101.py").is_file():
        raise SystemExit("source snapshot is missing scene1_train_so101.py")
    if not (source_root / "scene1_validation.py").is_file():
        raise SystemExit("source snapshot is missing scene1_validation.py")
    if sha256_file(training_config) != args.training_config_sha256:
        raise SystemExit("training config hash mismatch")
    if args.strict_training_config:
        registered_config = json.loads(training_config.read_text(encoding="utf-8"))
        problems = strict_training_config_problems(registered_config, args, train_view)
        if problems:
            raise SystemExit("strict training config failed: " + "; ".join(problems))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "home").mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-scene1-{args.label}"
    exact_name = f"so101-scene1-{args.mode}-{run_id}"
    indices = [int(value) for value in args.gpu_indices.split(",") if value.strip()]
    if not indices or len(set(indices)) != len(indices):
        raise SystemExit("--gpu-indices must be distinct")
    if not 1 <= args.fsdp_devices <= len(indices) or len(indices) % args.fsdp_devices:
        raise SystemExit("fsdp_devices must divide the selected device count")
    devices = [gpu_state(index) for index in indices]
    memory = memory_plan(devices, demand_mib=args.required_memory_mib,
                         margin_mib=args.memory_margin_mib,
                         explicit_fraction=args.jax_memory_fraction)
    cache = run_dir / "openpi_cache/big_vision"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "paligemma_tokenizer.model"
    if not destination.is_file():
        shutil.copy2(tokenizer, destination)
    container_run = "/work/" + str(run_dir.relative_to(REPO))
    container_view = "/work/" + str(train_view.relative_to(REPO))
    container_source = "/work/" + str(source_root.relative_to(REPO))
    command = ["-u", f"{container_source}/scene1_train_so101.py", "--mode", args.mode,
               "--phase", "scene1", "--run-root", container_run, "--train-view", container_view,
               "--base-checkpoint", "/base", "--exp-name", args.exp_name,
               "--state-mode", args.state_mode, "--batch-size", str(args.batch_size),
               "--steps", str(args.steps), "--save-interval", str(args.save_interval),
               "--data-loader-workers", str(args.data_loader_workers),
               "--lr-warmup-steps", str(args.lr_warmup_steps), "--peak-lr", str(args.peak_lr),
               "--decay-steps", str(args.decay_steps), "--decay-lr", str(args.decay_lr),
               "--seed", str(args.seed), "--action-horizon", str(args.action_horizon),
               "--optimizer", args.optimizer, "--gradient-clip-norm", str(args.gradient_clip_norm),
               "--dtype", args.dtype, "--fsdp-devices", str(args.fsdp_devices),
               "--paligemma-variant", "gemma_2b_lora", "--action-expert-variant", "gemma_300m",
               "--freeze-vision-encoder" if args.freeze_vision_encoder else "--no-freeze-vision-encoder",
               "--expected-groups", "1", "--dataset-label", args.dataset_label,
               "--expected-total-episodes", str(args.expected_total_episodes),
               "--expected-train-episodes", str(args.expected_train_episodes),
               "--expected-validation-episodes", str(args.expected_validation_episodes),
               "--sampling-plan", f"{container_view}/sampling_plan.json",
               "--data-admission", f"{container_view}/e4t_admission.json",
               "--registered-training-config", "/work/" + str(training_config.relative_to(REPO)),
               "--registered-config-sha256", args.training_config_sha256,
               "--registered-base-sha256", args.registered_base_sha256,
               "--run-id", run_id]
    if args.ema_decay is not None:
        command += ["--ema-decay", str(args.ema_decay)]
    if args.mode == "resume":
        if args.resume_step is None:
            raise SystemExit("--resume-step is required in resume mode")
        command += ["--resume-step", str(args.resume_step)]
    if args.require_action_norm_audit:
        command += ["--require-action-norm-audit"]
    if args.require_validation_loss:
        command += [
            "--require-validation-loss",
            "--validation-every", str(args.validation_every),
            "--validation-seed", str(args.validation_seed),
            "--validation-batch-size", str(args.validation_batch_size),
        ]
    if args.validation_benchmark:
        command += ["--validation-benchmark"]
    if args.checkpoint_probe:
        command += ["--checkpoint-probe"]
    env_args = ["-e", f"HOME={container_run}/home",
                "-e", f"OPENPI_DATA_HOME={container_run}/openpi_cache",
                "-e", f"PYTHONPATH={container_source}", "-e", "JAX_PLATFORMS=cuda",
                "-e", "XLA_FORCE_COMPILATION_CACHE=1",
                "-e", f"JAX_COMPILATION_CACHE_DIR={container_run}/jax_cache",
                "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                "-e", f"XLA_PYTHON_CLIENT_MEM_FRACTION={memory['jax_memory_fraction']:.6f}",
                "-e", f"SO101_TRAIN_VIEW={container_view}"]
    launch = ["--entrypoint", "python3", "--shm-size=4g", "-v", f"{REPO}:/work",
              "-v", f"{base}:/base:ro", "-w", "/work"] + env_args
    registry_path = run_dir / "resource_registry.json"
    allocated = registry(registry_path, [
        "allocate", "--run-id", run_id, "--exact-name", exact_name, "--image", args.image,
        "--command-json", json.dumps(command), "--docker-arg-json", json.dumps(launch),
        "--gpu-index", ",".join(str(value) for value in indices)])
    gpu_log = run_dir / f"gpu_{run_id}.txt"
    stdout_log = run_dir / f"train_{args.mode}_{run_id}.log"
    monitor = stream = handle = None
    status = -1
    started = time.time()
    try:
        inside = subprocess.run(["docker", "exec", allocated["container_id"], "nvidia-smi",
                                 "--query-gpu=uuid", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=120, check=True).stdout
        visible = sorted({line.strip() for line in inside.splitlines() if line.strip()})
        expected = sorted(device["uuid"] for device in devices)
        if visible != expected:
            raise RuntimeError(f"container GPU mismatch: registered {expected}, visible {visible}")
        monitor = subprocess.Popen(
            ["bash", "-c",
             f'while docker container inspect {shlex.quote(allocated["container_id"])} >/dev/null 2>&1; do '
             f'nvidia-smi -i {",".join(str(value) for value in indices)} '
             f'--query-gpu=index,uuid,memory.used --format=csv,noheader,nounits '
             f'>> {shlex.quote(str(gpu_log))} 2>/dev/null; sleep 1; done'])
        handle = stdout_log.open("w", encoding="utf-8")
        stream = subprocess.Popen(["docker", "logs", "-f", allocated["container_id"]],
                                  stdout=handle, stderr=subprocess.STDOUT)
        result = registry(registry_path, ["wait", "--entry-id", allocated["registry_entry_id"],
                                          "--timeout", str(args.timeout_seconds)],
                          timeout=args.timeout_seconds + 60)
        status = int(result["exit_code"])
    finally:
        if monitor is not None:
            monitor.terminate()
        if stream is not None:
            stream.terminate()
        if handle is not None:
            handle.close()
        registry(registry_path, ["stop", "--entry-id", allocated["registry_entry_id"],
                                 "--confirm", "STOP-REGISTERED-EXACT-ENTRIES"], timeout=180)
    peaks = {}
    if gpu_log.is_file():
        for line in gpu_log.read_text(encoding="utf-8").splitlines():
            parts = [value.strip() for value in line.split(",")]
            if len(parts) == 3 and parts[0].isdigit() and parts[2].isdigit():
                peaks[parts[0]] = max(peaks.get(parts[0], 0), int(parts[2]))
    summary = {
        "schema_version": 1, "mode": args.mode, "run_id": run_id, "exact_name": exact_name,
        "owner": allocated["owner"], "container_id": allocated["container_id"],
        "registry_entry_id": allocated["registry_entry_id"], "image": args.image,
        "exit_code": status, "started_at": now(), "wall_time_s": time.time() - started,
        "gpu": [{"index": device["index"], "uuid": device["uuid"],
                 "memory_total_mib": device["memory_total_mib"],
                 "memory_used_mib_at_admission": device["memory_used_mib"],
                 "memory_free_mib_at_admission": device["memory_free_mib"],
                 "peak_mib_sampled": peaks.get(str(device["index"]), 0)} for device in devices],
        "memory_plan": memory, "memory_margin_mib": args.memory_margin_mib,
        "fsdp_devices": args.fsdp_devices, "batch_size": args.batch_size, "steps": args.steps,
        "data_loader_workers": args.data_loader_workers, "command": command,
        "log": stdout_log.name, "gpu_log": gpu_log.name,
        "sampled_margin_problems": sampled_margin_problems(
            [{"index": device["index"], "memory_total_mib": device["memory_total_mib"],
              "peak_mib_sampled": peaks.get(str(device["index"]), 0)} for device in devices],
            margin_mib=args.memory_margin_mib),
    }
    (run_dir / f"train_{args.mode}_{run_id}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
