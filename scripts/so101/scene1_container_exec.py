#!/usr/bin/env python3
"""Run one registered command inside the pinned openpi-server image.

Used by the clean60 round for read-only runtime inspection (parameter tree,
normalisation source, sharding semantics) and for the GPU capacity probes.
Without ``--gpu-index`` the container gets no device at all.  With it, every
selected card is admitted on measured free memory, the JAX pool is capped from
the tightest card, the container-visible UUID set is rechecked and per-card
peaks are sampled while the probe runs.  Every container goes through the
exact-name registry so its id, name, run-id and owner are recorded.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "so101"))
from resource_policy import memory_plan  # noqa: E402


def gpu_state(index: int) -> dict:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(index),
         "--query-gpu=uuid,memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    uuid, total, used, free = [part.strip() for part in result.stdout.strip().split(",", 3)]
    return {"index": index, "uuid": uuid, "memory_total_mib": int(total),
            "memory_used_mib": int(used), "memory_free_mib": int(free)}


def select_devices(indices: list[int], required_mib: int, margin_mib: int,
                   fraction: float | None) -> tuple[list[dict], dict]:
    """Admit on measured free memory only, then cap the JAX pool at the tightest card.

    Mirrors the training launcher so the probe ceiling and the real run ceiling are
    the same number; utilisation is deliberately not a gate.
    """
    devices = [gpu_state(index) for index in indices]
    plan = memory_plan(
        devices, demand_mib=required_mib, margin_mib=margin_mib,
        explicit_fraction=fraction)
    return devices, plan


def sample_peaks(container_id: str, indices: list[int], log_path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        ["bash", "-c",
         f'while docker container inspect {shlex.quote(container_id)} >/dev/null 2>&1; do '
         f'nvidia-smi -i {",".join(str(value) for value in indices)} '
         f'--query-gpu=index,uuid,memory.used --format=csv,noheader,nounits '
         f'>> {shlex.quote(str(log_path))} 2>/dev/null; sleep 1; done'])


def read_peaks(log_path: Path) -> dict[str, int]:
    peaks: dict[str, int] = {}
    if not log_path.is_file():
        return peaks
    for line in log_path.read_text(encoding="utf-8").splitlines():
        parts = [value.strip() for value in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit() and parts[2].isdigit():
            peaks[parts[0]] = max(peaks.get(parts[0], 0), int(parts[2]))
    return peaks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", required=True, help="short probe name used in the run id")
    parser.add_argument("--python-code", default=None)
    parser.add_argument("--container-python-file", default=None,
                        help="path already visible inside the /work mount, e.g. /work/scripts/so101/...py")
    parser.add_argument("--python-arg", action="append", default=[],
                        help="argv passed to the container python file (after its own name)")
    parser.add_argument("--mount", action="append", default=[],
                        help="host:path container mounts, given as /abs/host:/abs/container")
    parser.add_argument("--env", action="append", default=[],
                        help="extra KEY=VALUE container environment entries")
    parser.add_argument("--image", default="openpi-server:latest")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--gpu-index", default=None,
                        help="comma-separated physical device ids; omits --gpus when unset")
    parser.add_argument("--required-memory-mib", type=int, default=0)
    parser.add_argument("--memory-margin-mib", type=int, default=512)
    parser.add_argument("--jax-memory-fraction", type=float, default=None)
    parser.add_argument("--shm-size", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"scene1-{args.label}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    registry = run_dir / "resource_registry.json"
    launch = ["--entrypoint", "python3", "-w", "/work", "-v", f"{REPO}:/work"]
    if args.offline:
        launch.insert(0, "--network")
        launch.insert(1, "none")
    for value in args.mount:
        launch += ["-v", value]
    for value in args.env:
        launch += ["-e", value]
    if args.shm_size:
        launch += [f"--shm-size={args.shm_size}"]
    if bool(args.python_code) == bool(args.container_python_file):
        parser.error("provide exactly one of --python-code / --container-python-file")
    script = str(args.container_python_file) if args.container_python_file else None
    if script:
        tail = json.dumps([str(value) for value in args.python_arg])
        command = ["-c", f"import sys, runpy; sys.argv = [{script!r}] + {tail}; "
                         f"runpy.run_path({script!r}, run_name='__main__')"]
    else:
        if args.python_arg:
            parser.error("--python-arg needs --container-python-file")
        command = ["-c", args.python_code]

    indices = ([int(value) for value in args.gpu_index.split(",") if value.strip()]
               if args.gpu_index else [])
    if len(set(indices)) != len(indices):
        parser.error("--gpu-index must list distinct device ids")
    devices: list[dict] = []
    memory = None
    if indices:
        devices, memory = select_devices(indices, args.required_memory_mib,
                                         args.memory_margin_mib, args.jax_memory_fraction)
        launch += ["-e", "JAX_PLATFORMS=cuda",
                   "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                   "-e", f"XLA_PYTHON_CLIENT_MEM_FRACTION={memory['jax_memory_fraction']:.6f}"]
    allocate = [
        sys.executable, str(REPO / "scripts/so101/e4t_resources.py"), "--registry", str(registry),
        "allocate", "--run-id", run_id, "--exact-name", f"so101-{run_id}", "--image", args.image,
        "--command-json", json.dumps(command), "--docker-arg-json", json.dumps(launch),
    ]
    if indices:
        allocate += ["--gpu-index", args.gpu_index]
    result = subprocess.run(allocate, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(result.stderr[-2000:], file=sys.stderr)
        return 1
    entry = json.loads(result.stdout)
    entry_id = entry["registry_entry_id"]
    monitor = None
    gpu_log = run_dir / f"gpu_{run_id}.txt"
    exit_code = -1
    try:
        if indices:
            inside = subprocess.run(["docker", "exec", entry["container_id"], "nvidia-smi",
                                     "--query-gpu=uuid", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=120)
            visible = sorted({line.strip() for line in inside.stdout.splitlines() if line.strip()})
            registered = sorted(device["uuid"] for device in devices)
            if visible != registered:
                raise RuntimeError(f"container GPU mismatch: registered {registered}, visible {visible}")
            monitor = sample_peaks(entry["container_id"], indices, gpu_log)
        wait = subprocess.run(
            [sys.executable, str(REPO / "scripts/so101/e4t_resources.py"), "--registry", str(registry),
             "wait", "--entry-id", entry_id, "--timeout", str(args.timeout)],
            capture_output=True, text=True, timeout=args.timeout + 60)
        exit_code = json.loads(wait.stdout)["exit_code"] if wait.returncode == 0 else -1
        logs = subprocess.run(["docker", "logs", entry["container_id"]],
                              capture_output=True, text=True, timeout=120)
        Path(run_dir / f"probe_{run_id}.log").write_text(
            logs.stdout + ("\n--- stderr ---\n" + logs.stderr if logs.stderr else "") + "\n",
            encoding="utf-8")
        print(logs.stdout)
        if logs.stderr.strip():
            print(logs.stderr, file=sys.stderr)
    finally:
        if monitor is not None:
            monitor.terminate()
        subprocess.run([sys.executable, str(REPO / "scripts/so101/e4t_resources.py"),
                        "--registry", str(registry), "stop", "--entry-id", entry_id,
                        "--confirm", "STOP-REGISTERED-EXACT-ENTRIES"],
                       capture_output=True, text=True, timeout=180)

    peaks = read_peaks(gpu_log) if indices else {}
    summary = {"run_id": run_id, "exact_name": entry["exact_name"],
               "image": args.image,
               "container_id": entry["container_id"], "registry_entry_id": entry_id,
               "exit_code": exit_code, "log": f"probe_{run_id}.log",
               "label": args.label, "timeout_s": args.timeout}
    if indices:
        summary["gpu"] = {
            "devices": [{"index": device["index"], "uuid": device["uuid"],
                         "memory_total_mib": device["memory_total_mib"],
                         "memory_used_mib_at_admission": device["memory_used_mib"],
                         "memory_free_mib_at_admission": device["memory_free_mib"],
                         "peak_mib_sampled": peaks.get(str(device["index"]), 0)}
                        for device in devices],
            "peak_mib_sampled_max": max(peaks.values(), default=0),
            "jax_memory_fraction": round(memory["jax_memory_fraction"], 6),
            "required_memory_mib": args.required_memory_mib,
            "memory_margin_mib": args.memory_margin_mib,
            "memory_plan": memory,
            "utilisation_is_not_an_admission_gate": True,
            "gpu_log": gpu_log.name,
        }
    (run_dir / f"probe_{run_id}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
