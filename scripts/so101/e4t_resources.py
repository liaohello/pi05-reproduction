#!/usr/bin/env python3
"""Exact-owner container registry for E4-T.

A container may be started, inspected or removed only when it matches one
registry entry on all of: full container ID, exact name, project, run id,
owner, image and launch command. There is no broad label/name prefix cleanup
path and no purge operation. The registry is a private run artifact.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT = "pi05-reproduction"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run(argv: list[str], timeout: int = 120) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def _inspect(name: str) -> dict:
    template = "{{.Id}}|{{.State.Status}}|{{.Config.Image}}|{{json .Config.Labels}}|{{json .Path}}|{{json .Args}}"
    raw = _run(["docker", "container", "inspect", "--format", template, name])
    container_id, state, image, labels_raw, path_raw, args_raw = raw.split("|", 5)
    return {
        "container_id": container_id,
        "state": state,
        "image": image,
        "labels": json.loads(labels_raw),
        "path": json.loads(path_raw),
        "args": json.loads(args_raw),
    }


def _load_registry(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": 1, "project": PROJECT, "entries": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("project") != PROJECT:
        raise RuntimeError("registry belongs to another project")
    value.setdefault("entries", [])
    return value


@contextlib.contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / f"{path.name}.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write_registry(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _entry_matches(entry: dict, inspected: dict) -> list[str]:
    problems = []
    if entry["container_id"] != inspected["container_id"]:
        problems.append("full container id differs")
    if entry["exact_name"] != inspected["labels"].get("pi05.e4t.exact_name"):
        problems.append("registered exact-name label differs")
    for key in ("project", "run_id", "owner"):
        expected = {"project": PROJECT, "run_id": entry["run_id"], "owner": entry["owner"]}[key]
        if inspected["labels"].get(f"pi05.e4t.{key}") != expected:
            problems.append(f"{key} label differs")
    if entry["image"] != inspected["image"]:
        problems.append("image differs")
    if entry["command"] != list(inspected["args"]) or not str(inspected["path"]).endswith("python3"):
        problems.append("launch command or pinned python entrypoint differs")
    return problems


def _gpu_info(gpu_index: int | None) -> dict:
    if gpu_index is None:
        return {"gpu_uuid": None, "memory_free_mib": None}
    raw = _run(["nvidia-smi", "-i", str(gpu_index),
                "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"])
    gpu_uuid, free = [part.strip() for part in raw.split(",", 1)]
    return {"gpu_uuid": gpu_uuid, "memory_free_mib": int(free)}


def allocate(args) -> int:
    registry_path = Path(args.registry).expanduser().resolve()
    if Path(args.exact_name).exists() or "/" in args.exact_name or args.exact_name.startswith("-"):
        raise RuntimeError("invalid exact container name")
    try:
        existing = _inspect(args.exact_name)
        raise RuntimeError(f"container name already exists: {existing['state']}")
    except RuntimeError as error:
        message = str(error).lower()
        if "no such object" not in message and "no such container" not in message and "does not exist" not in message:
            raise
    owner = f"pid-{os.getpid()}-uuid-{uuid.uuid4()}"
    command = json.loads(args.command_json)
    gpu = _gpu_info(None if args.gpu_index is None else int(args.gpu_index))
    argv = ["docker", "run", "-d", "--name", args.exact_name,
            "--label", f"pi05.e4t.project={PROJECT}",
            "--label", f"pi05.e4t.run_id={args.run_id}",
            "--label", f"pi05.e4t.owner={owner}",
            "--label", f"pi05.e4t.exact_name={args.exact_name}"]
    if args.gpu_index is not None:
        argv += ["--gpus", f"device={args.gpu_index}"]
    for item in json.loads(args.docker_arg_json):
        argv.append(item)
    argv += [args.image, *command]
    started = time.time()
    container_id = _run(argv)
    inspected = _inspect(args.exact_name)
    if not container_id.startswith(inspected["container_id"][:12]):
        raise RuntimeError("Docker returned a container id that cannot be re-inspected by exact name")
    entry = {
        "registry_entry_id": str(uuid.uuid4()), "created": _now(),
        "project": PROJECT, "run_id": args.run_id, "owner": owner,
        "exact_name": args.exact_name, "container_id": inspected["container_id"],
        "image": inspected["image"], "command": command,
        "launch_args": json.loads(args.docker_arg_json), "gpu_index": args.gpu_index, **gpu,
        "state_at_registration": inspected["state"], "status": "running",
        "matched_all_registration_fields": True,
    }
    if _entry_matches(entry, inspected):
        _run(["docker", "container", "stop", inspected["container_id"]])
        _run(["docker", "container", "rm", "-f", inspected["container_id"]])
        raise RuntimeError("new container does not match its own registration: " + "; ".join(_entry_matches(entry, inspected)))
    with _locked(registry_path):
        value = _load_registry(registry_path)
        if any(item["exact_name"] == args.exact_name and item.get("status") in {"running", "exited"}
               for item in value["entries"]):
            raise RuntimeError("registry already contains this exact name")
        value["entries"].append(entry)
        _write_registry(registry_path, value)
    entry["elapsed_to_registration_s"] = time.time() - started
    print(json.dumps(entry, indent=2))
    return 0


def wait(args) -> int:
    registry_path = Path(args.registry).expanduser().resolve()
    value = _load_registry(registry_path)
    entry = next((item for item in value["entries"] if item["registry_entry_id"] == args.entry_id), None)
    if entry is None:
        raise RuntimeError("registry entry not found; refusing to wait on an arbitrary container")
    inspected = _inspect(entry["container_id"])
    problems = _entry_matches(entry, inspected)
    if problems:
        raise RuntimeError("registry mismatch before wait: " + "; ".join(problems))
    exit_code = int(_run(["docker", "container", "wait", entry["container_id"]], timeout=args.timeout))
    entry["status"] = "exited"
    entry["exit_code"] = exit_code
    entry["finished_at"] = _now()
    with _locked(registry_path):
        value = _load_registry(registry_path)
        for item in value["entries"]:
            if item["registry_entry_id"] == entry["registry_entry_id"]:
                item.update({"status": "exited", "exit_code": exit_code, "finished_at": entry["finished_at"]})
        _write_registry(registry_path, value)
    print(json.dumps({"entry_id": entry["registry_entry_id"], "exit_code": exit_code,
                      "exact_name": entry["exact_name"]}, indent=2))
    return 0


def verify(args) -> int:
    registry_path = Path(args.registry).expanduser().resolve()
    with _locked(registry_path):
        value = _load_registry(registry_path)
        entries = [item for item in value["entries"]
                   if args.entry_id is None or item["registry_entry_id"] == args.entry_id]
        if not entries:
            raise RuntimeError("no matching registry entry; refusing to act")
        report = []
        for entry in entries:
            try:
                inspected = _inspect(entry["container_id"])
            except RuntimeError as error:
                report.append({"entry": entry["registry_entry_id"], "exists": False, "error": str(error)})
                continue
            problems = _entry_matches(entry, inspected)
            report.append({"entry": entry["registry_entry_id"], "exists": True,
                           "exact_name": entry["exact_name"], "state": inspected["state"],
                           "matched_all": not problems, "problems": problems})
        print(json.dumps({"verified_at": _now(), "containers": report}, indent=2))
        return 0 if all(row.get("matched_all", False) or not row.get("exists") for row in report) else 1


def stop_registered(args) -> int:
    registry_path = Path(args.registry).expanduser().resolve()
    with _locked(registry_path):
        value = _load_registry(registry_path)
        selected = []
        for entry in value["entries"]:
            if args.entry_id and entry["registry_entry_id"] not in args.entry_id:
                continue
            if not args.entry_id:
                if args.run_id and entry["run_id"] != args.run_id:
                    continue
                if not args.all_this_run:
                    raise RuntimeError("refusing to stop: supply --entry-id, or --all-this-run with --confirm")
                selected.append(entry)
            else:
                selected.append(entry)
        if args.all_this_run and args.confirm != "STOP-REGISTERED-EXACT-ENTRIES":
            raise RuntimeError("all-this-run requires the exact confirmation token")
        if not selected:
            raise RuntimeError("no registry-selected exact entry; refusing")
        operations = []
        for entry in selected:
            inspected = _inspect(entry["container_id"])
            problems = _entry_matches(entry, inspected)
            if problems:
                raise RuntimeError(f"registry mismatch for {entry['exact_name']}: " + "; ".join(problems))
            _run(["docker", "container", "stop", entry["container_id"]], timeout=args.timeout)
            _run(["docker", "container", "rm", entry["container_id"]], timeout=args.timeout)
            entry["status"] = "released"
            entry["released_at"] = _now()
            operations.append({"container_id": entry["container_id"], "exact_name": entry["exact_name"],
                               "status": "released"})
        _write_registry(registry_path, value)
        print(json.dumps({"released": operations, "broad_filter_used": False,
                          "purge_used": False}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    alloc = sub.add_parser("allocate")
    alloc.add_argument("--run-id", required=True)
    alloc.add_argument("--exact-name", required=True)
    alloc.add_argument("--image", required=True)
    alloc.add_argument("--command-json", required=True)
    alloc.add_argument("--docker-arg-json", default="[]")
    alloc.add_argument("--gpu-index")
    alloc.set_defaults(func=allocate)
    ver = sub.add_parser("verify")
    ver.add_argument("--entry-id")
    ver.set_defaults(func=verify)
    stop = sub.add_parser("stop")
    stop.add_argument("--entry-id", action="append")
    stop.add_argument("--run-id")
    stop.add_argument("--all-this-run", action="store_true")
    stop.add_argument("--confirm", default="")
    stop.add_argument("--timeout", type=int, default=120)
    stop.set_defaults(func=stop_registered)
    waitp = sub.add_parser("wait")
    waitp.add_argument("--entry-id", required=True)
    waitp.add_argument("--timeout", type=int, default=86400)
    waitp.set_defaults(func=wait)
    sub.add_parser("list").set_defaults(func=lambda args: print(json.dumps(_load_registry(Path(args.registry)), indent=2)))
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as error:
        print(f"E4-T RESOURCE REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
