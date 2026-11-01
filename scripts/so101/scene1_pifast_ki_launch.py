#!/usr/bin/env python3
"""Launch one registered Scene1 PI-FAST probe or formal training container."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/deploy"))
sys.path.insert(0, str(REPO / "scripts/so101"))
import common  # noqa: E402
from resource_policy import memory_plan  # noqa: E402


def project_paths(cfg: dict, path: Path) -> tuple[Path, Path, Path]:
    """Return local, remote-host and container forms for project paths."""
    path = path.expanduser().resolve()
    if path.is_relative_to(REPO):
        relative = path.relative_to(REPO)
    else:
        root = Path(cfg["remote"]["root"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("path is outside both the local and remote project")
        relative = path.relative_to(root)
    return (
        REPO / relative,
        Path(cfg["remote"]["root"]) / relative,
        Path("/work") / relative,
    )


def gpu_states(cfg: dict, indices: list[int]) -> list[dict]:
    query = ",".join(str(value) for value in indices)
    result = common.run(common.ssh_argv(cfg) + [
        f"nvidia-smi -i {query} --query-gpu=index,uuid,memory.total,"
        "memory.used,memory.free --format=csv,noheader,nounits"
    ], timeout=30)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-1000:])
    devices = []
    for line in result.stdout.strip().splitlines():
        index, uuid, total, used, free = [
            value.strip() for value in line.split(",", 4)]
        devices.append({
            "index": int(index),
            "uuid": uuid,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
        })
    if [item["index"] for item in devices] != indices:
        raise RuntimeError("nvidia-smi did not return the requested device order")
    return devices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("initial", "probe"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--data-audit-sha256", required=True)
    parser.add_argument("--data-admission", type=Path, required=True)
    parser.add_argument("--sampling-plan", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--fast-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--paligemma-tokenizer", type=Path, required=True)
    parser.add_argument("--fast-ce-weight", type=float, required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--gpu-indices", required=True)
    parser.add_argument("--fsdp-devices", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--demand-mib", type=int, required=True)
    parser.add_argument("--memory-margin-mib", type=int, default=100)
    parser.add_argument("--jax-memory-fraction", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=108000)
    parser.add_argument(
        "--image",
        default=(
            "openpi-server@sha256:"
            "405af52051682d5e13169d1de1523fd0adc116eceae4fc314cadfacd45b1ad52"
        ),
    )
    parser.add_argument("--checkpoint-probe", action="store_true")
    parser.add_argument(
        "--deterministic-ops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable XLA GPU deterministic operations (higher memory)",
    )
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    cfg = common.load_config()
    root = Path(cfg["remote"]["root"]).resolve()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_relative_to(REPO):
        raise ValueError("run directory must be inside the remote project")
    run_relative = run_dir.relative_to(REPO)
    remote_run_dir = root / run_relative
    if args.memory_margin_mib < 100:
        raise ValueError("memory margin must be at least 100 MiB")
    if args.mode == "initial" and (run_dir / "checkpoints").exists():
        raise ValueError("formal initial run directory already has checkpoints")
    indices = [
        int(value) for value in args.gpu_indices.split(",") if value.strip()
    ]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("GPU indices must be non-empty and unique")
    if not 1 <= args.fsdp_devices <= len(indices):
        raise ValueError("fsdp_devices must fit the selected device set")
    if len(indices) % args.fsdp_devices:
        raise ValueError("fsdp_devices must divide the visible device count")

    devices = gpu_states(cfg, indices)
    plan = memory_plan(
        devices,
        demand_mib=args.demand_mib,
        margin_mib=args.memory_margin_mib,
        explicit_fraction=args.jax_memory_fraction,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    created = common.run(
        common.ssh_argv(cfg) + [f"mkdir -p {shlex.quote(str(remote_run_dir))}"],
        timeout=30,
    )
    if created.returncode:
        raise RuntimeError((created.stderr or created.stdout)[-1000:])
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    exact_name = f"scene1-pifast-{args.mode}-{args.run_id}-{stamp}"
    _, _, container_source_path = project_paths(cfg, args.source_dir)
    _, _, container_overlay_path = project_paths(cfg, args.overlay_dir)
    _, _, container_run_path = project_paths(cfg, run_dir)
    _, _, container_view_path = project_paths(cfg, args.train_view)
    _, _, container_plan_path = project_paths(cfg, args.plan)
    _, _, container_audit_path = project_paths(cfg, args.data_audit)
    _, _, container_admission_path = project_paths(cfg, args.data_admission)
    _, _, container_sampling_path = project_paths(cfg, args.sampling_plan)
    _, remote_fast_tokenizer, _ = project_paths(cfg, args.fast_tokenizer_dir)
    _, remote_paligemma_tokenizer, _ = project_paths(
        cfg, args.paligemma_tokenizer)
    container_source = str(container_source_path)
    container_overlay = str(container_overlay_path)
    container_run = str(container_run_path)
    container_view = str(container_view_path)
    container_plan = str(container_plan_path)
    container_audit = str(container_audit_path)
    container_admission = str(container_admission_path)
    container_sampling = str(container_sampling_path)

    command = [
        "-u", f"{container_source}/scene1_train_so101.py",
        "--mode", "probe" if args.mode == "probe" else "initial",
        "--phase", "scene1",
        "--run-id", args.run_id,
        "--run-root", container_run,
        "--train-view", container_view,
        "--base-checkpoint", "/base",
        "--exp-name", args.exp_name,
        "--state-mode", "on",
        "--batch-size", str(args.batch_size),
        "--steps", str(args.steps),
        "--save-interval", "2000",
        "--data-loader-workers", "8",
        "--lr-warmup-steps", "1000",
        "--peak-lr", "2.5e-5",
        "--decay-steps", "40000",
        "--decay-lr", "2.5e-6",
        "--seed", "42",
        "--action-horizon", "16",
        "--optimizer", "AdamW",
        "--gradient-clip-norm", "1.0",
        "--dtype", "bfloat16",
        "--fsdp-devices", str(args.fsdp_devices),
        "--paligemma-variant", "gemma_2b_lora",
        "--action-expert-variant", "gemma_300m_lora_r64",
        "--no-freeze-vision-encoder",
        "--expected-groups", "1",
        "--dataset-label", "scene1_single_random_v1",
        "--expected-total-episodes", "64",
        "--expected-train-episodes", "52",
        "--expected-validation-episodes", "12",
        "--sampling-plan", container_sampling,
        "--data-admission", container_admission,
        "--require-validation-loss",
        "--validation-every", "50",
        "--validation-seed", "42",
        "--validation-batch-size", "16",
        "--pifast-plan", container_plan,
        "--pifast-plan-sha256", args.plan_sha256,
        "--pifast-data-audit", container_audit,
        "--pifast-data-audit-sha256", args.data_audit_sha256,
        "--fast-tokenizer-dir", "/pifast/fast",
        "--paligemma-tokenizer", "/pifast/paligemma.model",
        "--fast-ce-weight", str(args.fast_ce_weight),
        "--registered-base-sha256",
        "9909d717ec51c078a3d5fd3464cb7612ae7b6a0a713121190e9455420b2350cd",
    ]
    if args.checkpoint_probe:
        command.append("--checkpoint-probe")
    docker = [
        "docker", "run", "-d",
        "--name", exact_name,
        "--label", "pi05.e4t.project=pi05-reproduction",
        "--label", f"pi05.e4t.role=scene1-pifast-{args.mode}",
        "--label", f"pi05.e4t.run-id={args.run_id}",
        "--network", "none",
        "--shm-size", "16g",
        "--memory", "64g",
        "--gpus", f'"device={args.gpu_indices}"',
        "-e", "JAX_PLATFORMS=cuda",
        "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "-e",
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={plan['jax_memory_fraction']:.6f}",
        "-e", f"PYTHONPATH={container_overlay}:{container_source}:/opt/openpi/src",
        "-v", f"{root}:/work",
        "-v", f"{args.base_checkpoint.expanduser()}:/base:ro",
        "-v", f"{remote_fast_tokenizer}:/pifast/fast:ro",
        "-v", f"{remote_paligemma_tokenizer}:/pifast/paligemma.model:ro",
        "-v", f"{run_dir / 'jax_cache'}:/root/.cache/jax",
        "-w", container_source,
        "--entrypoint", "python3",
        args.image,
        *command,
    ]
    if args.deterministic_ops:
        anchor = docker.index("-w")
        docker[anchor:anchor] = [
            "-e", "XLA_FLAGS=--xla_gpu_deterministic_ops=true"]
    (run_dir / "jax_cache").mkdir(parents=True, exist_ok=True)
    launched = common.run(
        common.ssh_argv(cfg) + [shlex.join(docker)], timeout=120)
    if launched.returncode:
        raise RuntimeError((launched.stderr or launched.stdout)[-2000:])
    container_id = launched.stdout.strip()
    inspect = common.run(
        common.ssh_argv(cfg) + [
            shlex.join(["docker", "inspect", exact_name])
        ], timeout=60)
    (run_dir / "resource_registry.json").write_text(
        json.dumps({
            "schema_version": 1,
            "run_id": args.run_id,
            "exact_name": exact_name,
            "container_id": container_id,
            "owner": os.environ.get("USER", ""),
            "image": args.image,
            "command": command,
            "docker_command": docker,
            "gpu_admission": {
                "devices": devices,
                "memory_plan": plan,
            },
            "inspect": json.loads(inspect.stdout)[0]
            if inspect.returncode == 0 else None,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gpu_log = remote_run_dir / "gpu_peak_samples.csv"
    sampler = (
        f"while [ \"$(docker inspect -f '{{{{.State.Running}}}}' "
        f"{exact_name} 2>/dev/null)\" = true ]; do "
        f"nvidia-smi -i {args.gpu_indices} "
        "--query-gpu=index,uuid,memory.used,memory.free "
        f"--format=csv,noheader,nounits >> {shlex.quote(str(gpu_log))}; "
        "sleep 1; done"
    )
    common.run(
        common.ssh_argv(cfg) + [
            f"rm -f {shlex.quote(str(gpu_log))}; "
            f"nohup bash -lc {shlex.quote(sampler)} >/dev/null 2>&1 &"
        ], timeout=30)
    public = {
        "status": "RUNNING",
        "run_id": args.run_id,
        "container_id": container_id,
        "container_name": exact_name,
        "gpu_indices": indices,
        "gpu_uuids": [device["uuid"] for device in devices],
        "memory_free_mib_at_admission": [
            device["memory_free_mib"] for device in devices],
        "required_memory_mib": args.demand_mib + args.memory_margin_mib,
        "jax_memory_fraction": plan["jax_memory_fraction"],
        "batch_size": args.batch_size,
        "fsdp_devices": args.fsdp_devices,
        "xla_deterministic_ops": args.deterministic_ops,
        "steps": args.steps,
    }
    print(json.dumps(public, indent=2, sort_keys=True))
    if args.wait:
        waited = common.run(
            common.ssh_argv(cfg) + [f"docker wait {exact_name}"],
            timeout=args.timeout_seconds,
        )
        print(json.dumps({
            "container_exit": waited.stdout.strip(),
            "wait_returncode": waited.returncode,
        }, sort_keys=True))
        return 0 if waited.stdout.strip() == "0" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
