#!/usr/bin/env python3
"""Resource-owned E4-T OpenPI launcher.

This starts one non-auto-removed container with an exact name and full registry
entry.  GPU admission is based on measured free memory for that physical device,
never utilization. The selected device UUID is rechecked inside the container.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROJECT = "pi05-reproduction"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(argv: list[str], *, timeout: int = 120, check: bool = True) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


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


def registry_call(registry: Path, argv: list[str], timeout: int = 86400) -> dict:
    result = subprocess.run([sys.executable, str(REPO / "scripts/so101/e4t_resources.py"),
                             "--registry", str(registry), *argv], capture_output=True, text=True,
                            timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "probe", "initial", "eval"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--image", default="openpi-server:latest")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--state-mode", choices=("off", "on"), default="on")
    parser.add_argument("--gpu-index", type=int, required=True)
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
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--peak-lr", type=float, default=5e-5)
    parser.add_argument("--decay-steps", type=int, required=True)
    parser.add_argument("--decay-lr", type=float, default=5e-6)
    parser.add_argument("--data-loader-workers", type=int, default=0)
    parser.add_argument("--sampling-plan", type=Path, default=None)
    parser.add_argument("--data-admission", type=Path, default=None)
    parser.add_argument("--dataset-label", default="so101_e4t_50")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=8 * 3600)
    args = parser.parse_args()

    if args.mode == "probe":
        args.steps = 1
        args.save_interval = 1
    run_dir = args.run_dir.expanduser().resolve()
    train_view = args.train_view.expanduser().resolve()
    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    tokenizer = args.tokenizer.expanduser().resolve()
    if not run_dir.is_relative_to(REPO) or not train_view.is_relative_to(REPO) or not base_checkpoint.is_absolute():
        raise RuntimeError("training paths must be repo-relative mounts")
    run_id = args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    exact_name = f"so101-e4t-{args.mode}-{run_id}"
    cache = run_dir / "openpi_cache/big_vision"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "paligemma_tokenizer.model"
    if not destination.is_file() or destination.stat().st_size == 0:
        subprocess.run(["cp", "-n", str(tokenizer), str(destination)], check=False)
    if not destination.is_file():
        raise RuntimeError("tokenizer cache is not populated")

    before = gpu_state(args.gpu_index)
    if before["memory_free_mib"] < args.required_memory_mib + args.memory_margin_mib:
        raise RuntimeError(
            f"GPU {args.gpu_index} has {before['memory_free_mib']} MiB free; "
            f"requires {args.required_memory_mib + args.memory_margin_mib} MiB including margin")
    utilisation = before["memory_used_mib"] / before["memory_total_mib"]
    fraction = args.jax_memory_fraction
    if fraction is None:
        fraction = max(0.10, min(0.80, (before["memory_free_mib"] - args.memory_margin_mib)
                                 / before["memory_total_mib"]))
    if args.jax_memory_fraction is not None:
        estimated = args.jax_memory_fraction * before["memory_total_mib"]
        if estimated > before["memory_free_mib"] - args.memory_margin_mib:
            raise RuntimeError("explicit JAX fraction exceeds safe free memory plus margin")

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
        command = ["-u", "/work/scripts/so101/train_so101.py", "--mode", args.mode,
               "--phase", "E4-T", "--run-root", container_run, "--train-view", container_view,
               "--base-checkpoint", "/base", "--exp-name", args.exp_name,
               "--state-mode", args.state_mode, "--batch-size", str(args.batch_size),
               "--steps", str(args.steps), "--save-interval", str(args.save_interval),
               "--data-loader-workers", str(args.data_loader_workers),
               "--lr-warmup-steps", str(args.lr_warmup_steps), "--peak-lr", str(args.peak_lr),
               "--decay-steps", str(args.decay_steps), "--decay-lr", str(args.decay_lr)]
    if args.mode != "eval":
        if args.sampling_plan:
            command += ["--sampling-plan", "/work/" + str(args.sampling_plan.resolve().relative_to(REPO))]
        if args.data_admission:
            command += ["--data-admission", "/work/" + str(args.data_admission.resolve().relative_to(REPO))]
        command += ["--dataset-label", args.dataset_label,
                    "--expected-total-episodes", "50",
                    "--expected-train-episodes", "40",
                    "--expected-validation-episodes", "10"]

    env_args = [
        "-e", f"HOME={container_run}/home",
        "-e", f"OPENPI_DATA_HOME={container_run}/openpi_cache",
        "-e", "PYTHONPATH=/work/scripts/so101",
        "-e", "JAX_PLATFORMS=cuda",
        "-e", "XLA_FORCE_COMPILATION_CACHE=1",
        "-e", "JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache",
        "-e", f"XLA_PYTHON_CLIENT_PREALLOCATE={str(args.mode in ('initial', 'smoke')).lower()}",
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
        "--gpu-index", str(args.gpu_index)])
    try:
        inside_uuid = run(["docker", "exec", allocated["container_id"], "nvidia-smi",
                           "--query-gpu=uuid", "--format=csv,noheader"])
        inside_uuids = [line.strip() for line in inside_uuid.splitlines() if line.strip()]
        if inside_uuids != [before["uuid"]]:
            raise RuntimeError(f"container GPU mismatch: registered {before['uuid']}, visible {inside_uuids}")
    except Exception as error:
        registry_call(registry, ["stop", "--entry-id", allocated["registry_entry_id"],
                                 "--confirm", "STOP-REGISTERED-EXACT-ENTRIES"])
        raise

    artifacts = run_artifact_paths(run_dir, args.mode, run_id)
    gpu_log = artifacts["gpu_log"]
    monitor = subprocess.Popen(
        ["bash", "-c",
         f'while docker container inspect {shlex.quote(allocated["container_id"])} >/dev/null 2>&1; do '
         f'nvidia-smi -i {args.gpu_index} --query-gpu=memory.used --format=csv,noheader,nounits '
         f'>> {shlex.quote(str(gpu_log))} 2>/dev/null; sleep 1; done'])
    stdout_log = artifacts["stdout_log"]
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    started_at = now()
    stream = subprocess.Popen(["docker", "logs", "-f", allocated["container_id"]],
                              stdout=stdout_log.open("w", encoding="utf-8"),
                              stderr=subprocess.STDOUT)
    status = 1
    try:
        result = registry_call(registry, ["wait", "--entry-id", allocated["registry_entry_id"],
                                          "--timeout", str(args.timeout_seconds)], timeout=args.timeout_seconds + 60)
        status = int(result["exit_code"])
    finally:
        stream.wait(timeout=30)
        monitor.terminate()
        try:
            registry_call(registry, ["stop", "--entry-id", allocated["registry_entry_id"],
                                     "--confirm", "STOP-REGISTERED-EXACT-ENTRIES"])
        except Exception as cleanup_error:
            print(f"controlled cleanup failed: {cleanup_error}", file=sys.stderr)

    peak = 0
    if gpu_log.is_file():
        values = [int(line) for line in gpu_log.read_text(encoding="utf-8").splitlines() if line.strip().isdigit()]
        peak = max(values, default=0)
    payload = {
        "mode": args.mode, "run_id": run_id, "exact_name": exact_name,
        "registry_entry_id": allocated["registry_entry_id"],
        "container_id": allocated["container_id"], "exit_code": status,
        "started_at": started_at, "finished_at": now(),
        "wall_time_s": time.time() - started,
        "gpu": {"requested_index": args.gpu_index, "selected_uuid": before["uuid"],
                "memory_free_mib_at_admission": before["memory_free_mib"],
                "memory_total_mib": before["memory_total_mib"],
                "memory_used_mib_at_admission": before["memory_used_mib"],
                "utilisation_percent_at_admission": round(utilisation * 100, 1),
                "utilisation_is_not_an_admission_gate": True,
                "jax_memory_fraction": round(fraction, 6),
                "memory_margin_mib": args.memory_margin_mib, "peak_mib_sampled": peak},
        "batch_size": args.batch_size, "steps": args.steps,
        "command": command, "log": stdout_log.name, "gpu_log": gpu_log.name,
        "stop_method": "registered exact container wait plus controlled stop",
    }
    write_json(artifacts["run_payload"], payload)
    # Retain the most-recent legacy alias for scripts that only need the latest launch.
    write_json(run_dir / f"run_{args.mode}.json", payload)
    print(json.dumps(payload, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
