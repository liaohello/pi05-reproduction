"""Shared config resolution and SSH helpers for the deploy scripts.

Public repository rule: no real host, port, user, key path or private absolute
path may appear in source. Everything deployment-specific is resolved at runtime
from environment variables or from `scripts/deploy/config.local.json`, which is
git-ignored.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = ROOT / "scripts" / "deploy"
STATE_DIR = ROOT / "experiments" / ".state"


class ConfigError(RuntimeError):
    pass


def _resolve(value, field: str):
    if isinstance(value, dict) and "env" in value:
        raw = os.environ.get(value["env"], value.get("default"))
        if raw is None:
            raise ConfigError(
                f"missing environment variable {value['env']} (for {field}). "
                "Export the variables listed in scripts/deploy/config.example.json, "
                "or create the git-ignored scripts/deploy/config.local.json."
            )
        if value.get("type") == "int":
            return int(raw)
        return raw
    return value


def _load_raw() -> dict:
    for name in ("config.local.json", "config.json", "config.example.json"):
        path = DEPLOY_DIR / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise ConfigError("no deploy config found: scripts/deploy/config.example.json")


def load_config() -> dict:
    raw = _load_raw()
    ssh = raw["ssh"]
    remote = raw["remote"]
    assets = {k: _resolve(v, f"assets.{k}") for k, v in raw["assets"].items()}
    return {
        "project": raw.get("project", "pi05-reproduction"),
        "contract_version": raw.get("contract_version", 2),
        "ssh": {
            "host": _resolve(ssh["host"], "ssh.host"),
            "port": _resolve(ssh["port"], "ssh.port"),
            "user": _resolve(ssh["user"], "ssh.user"),
            "key": _resolve(ssh["key"], "ssh.key"),
            "connect_timeout": ssh.get("connect_timeout", 20),
        },
        "remote": {
            "label": remote.get("label", "remote-gpu-server"),
            "root": _resolve(remote["root"], "remote.root"),
            "runs_dir": _resolve(remote["runs_dir"], "remote.runs_dir"),
            "artifact_dir": _resolve(remote["artifact_dir"], "remote.artifact_dir"),
            "log_dir": _resolve(remote["log_dir"], "remote.log_dir"),
        },
        "assets": assets,
        "images": raw.get("images", {}),
        "sync": raw.get("sync", {"excludes": []}),
        "local": raw.get("local", {"min_free_gb": 2}),
    }


def load_config_or_exit() -> dict:
    """Load config, or print one actionable line and exit 2 (public-repo friendly)."""
    try:
        return load_config()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        raise SystemExit(2)


def key_path(cfg: dict) -> Path:
    return Path(os.path.expanduser(cfg["ssh"]["key"]))


def target(cfg: dict) -> str:
    """Real ssh destination. Never written into public artifacts."""
    return f"{cfg['ssh']['user']}@{cfg['ssh']['host']}"


def label(cfg: dict) -> str:
    """Public-safe name of the remote machine."""
    return cfg["remote"]["label"]


def ssh_argv(cfg: dict) -> list[str]:
    return [
        "ssh",
        "-i", str(key_path(cfg)),
        "-p", str(cfg["ssh"]["port"]),
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", f"ConnectTimeout={cfg['ssh'].get('connect_timeout', 20)}",
        "-o", "StrictHostKeyChecking=accept-new",
        target(cfg),
    ]


def rsync_ssh(cfg: dict) -> str:
    return (
        f"ssh -i {shlex.quote(str(key_path(cfg)))} -p {cfg['ssh']['port']}"
        " -o BatchMode=yes -o IdentitiesOnly=yes"
        f" -o ConnectTimeout={cfg['ssh'].get('connect_timeout', 20)}"
        " -o StrictHostKeyChecking=accept-new"
    )


def run(cmd: list[str], *, input_text: str | None = None, timeout: int = 120,
        check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=check,
    )


def remote_probe(cfg: dict, timeout: int = 60) -> tuple[dict, str]:
    """Run the read-only probe on the remote host and return (kv, error)."""
    probe = (DEPLOY_DIR / "remote_probe.sh").read_text(encoding="utf-8")
    args = " ".join(shlex.quote(str(v)) for v in [
        cfg["remote"]["root"],
        cfg["assets"].get("checkpoint", ""),
        cfg["assets"].get("norm_stats", ""),
        cfg["assets"].get("lerobot_data", ""),
    ])
    result = run(ssh_argv(cfg) + [f"bash -s -- {args}"], input_text=probe, timeout=timeout)
    if result.returncode != 0:
        return {}, (result.stderr or result.stdout)[-1500:].strip()
    data: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            data[key.strip()] = value.strip()
    return data, ""


def short_gpu_summary(cfg: dict, probe: dict) -> str:
    gpus = probe.get("gpu_list", "").strip()
    if not gpus:
        return "no GPU information"
    parts = []
    for line in gpus.split(";"):
        line = line.strip()
        if not line:
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) >= 4:
            parts.append(f"#{cols[0]} {cols[1]} {cols[2]}MiB(used {cols[3]}MiB)")
    return "; ".join(parts)


def sanitize_path(cfg: dict, value: str) -> str:
    """Replace private absolute prefixes with stable placeholders."""
    if not value:
        return value
    pairs = [
        (cfg["remote"]["runs_dir"], "<remote-workdir>/runs"),
        (cfg["remote"]["root"], "<remote-workdir>"),
        (str(key_path(cfg)), "<ssh-key>"),
    ]
    for real, placeholder in pairs:
        if real and real in value:
            value = value.replace(real, placeholder)
    home = str(Path.home())
    if value.startswith(home):
        value = "~" + value[len(home):]
    return value


def state_path(name: str) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / name
