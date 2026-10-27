#!/usr/bin/env python3
"""把本地源码同步到服务器实验目录，并做内容级一致性核验。

用法：
    python3 deploy/sync.py --check      # 只比对，不改服务器（内容级 -c 校验）
    python3 deploy/sync.py --dry-run    # 预览将发生的变化
    python3 deploy/sync.py              # 实际同步（含远程目录骨架）

同步范围：本仓库全部源码与资源，排除 .git/.venv/缓存/runs。
远程权重、数据、镜像留在资源池，通过 deploy/config.json 的 assets 引用，不复制。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROOT = common.ROOT
LAST_SYNC = common.state_path("last-sync.json")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def git_rev() -> str:
    result = common.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])
    if result.returncode != 0:
        return "no-git"
    dirty = common.run(["git", "-C", str(ROOT), "status", "--porcelain"]).stdout.strip()
    return result.stdout.strip() + ("-dirty" if dirty else "")


def precheck(cfg: dict) -> str | None:
    key = common.key_path(cfg)
    if not key.is_file():
        return (f"missing private key at {key}\n"
                "  copy your key there (chmod 600) or point PI05_SSH_KEY_PATH at it")
    result = common.run(common.ssh_argv(cfg) + ["true"], timeout=45)
    if result.returncode != 0:
        return f"SSH 连接失败：{(result.stderr or result.stdout)[-500:].strip()}"
    return None


def ensure_remote_dirs(cfg: dict) -> None:
    root = cfg["remote"]["root"]
    dirs = [root, cfg["remote"]["runs_dir"], cfg["remote"]["artifact_dir"], cfg["remote"]["log_dir"]]
    quoted = " ".join(f'"{d}"' for d in dirs)
    common.run(common.ssh_argv(cfg) + [f"mkdir -p {quoted} && echo OK"], timeout=60)


def build_rsync(cfg: dict, *, check: bool, dry_run: bool) -> list[str]:
    # The remote workdir is shared.  Deployment is add/update only: never pass --delete
    # to a real rsync run.  The check path still asks rsync for a *dry-run* deletion
    # report so remote-only files are visible, but nothing is removed.
    argv = ["rsync", "-a", "--itemize-changes", "--stats"]
    if check or dry_run:
        argv.insert(2, "--delete")
    cmd = argv + [
        "-e", common.rsync_ssh(cfg),
    ]
    if check:
        cmd += ["-c", "--dry-run"]
    elif dry_run:
        cmd += ["--dry-run"]
    for pattern in cfg["sync"]["excludes"]:
        cmd.append(f"--exclude={pattern}")
    cmd += [f"{ROOT}/", f"{common.target(cfg)}:{cfg['remote']['root']}/"]
    return cmd


def parse_stats(output: str) -> dict:
    info = {"changed": 0, "deleted": 0, "transferred": 0, "bytes": 0}
    for line in output.splitlines():
        if line.startswith(">f") or line.startswith("c") or line.startswith("<f"):
            info["changed"] += 1
        elif line.startswith("*deleting"):
            info["deleted"] += 1
        elif line.startswith("Number of regular files transferred:"):
            info["transferred"] = int(line.split(":")[1].strip().replace(",", ""))
        elif line.startswith("Total transferred file size:"):
            raw = line.split(":")[1].strip().split()[0].replace(",", "")
            try:
                info["bytes"] = int(raw)
            except ValueError:
                info["bytes"] = 0
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="内容级比对，不做修改")
    parser.add_argument("--dry-run", action="store_true", help="预览变化")
    args = parser.parse_args()

    cfg = common.load_config_or_exit()
    problem = precheck(cfg)
    if problem:
        print(f"[BLOCKED] {problem}")
        return 2

    if not args.check:
        ensure_remote_dirs(cfg)

    cmd = build_rsync(cfg, check=args.check, dry_run=args.dry_run)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = result.stdout + result.stderr
    stats = parse_stats(output)

    mode = "check" if args.check else ("dry-run" if args.dry_run else "sync")
    print(f"mode={mode} local -> {common.label(cfg)}:{cfg['remote']['root']}")
    if args.check:
        if result.returncode == 0 and stats["changed"] == 0 and stats["deleted"] == 0:
            print("VERIFY PASSED：服务器工作目录与本地内容逐字节一致")
        else:
            print(f"VERIFY FAILED：{stats['changed']} 处内容差异，{stats['deleted']} 处删除待处理")
            for line in output.splitlines()[:40]:
                if line.startswith((">f", "<f", "*deleting", "c")):
                    print("  " + line)
    else:
        print(f"差异条目={stats['changed']} 删除={stats['deleted']} "
              f"传输文件={stats['transferred']} 传输字节={stats['bytes']}")
        if result.returncode != 0:
            print(output[-2500:])
            return 1
        print("同步完成。下一步：python3 deploy/sync.py --check 做内容级复核")

    record = {
        "ts": now(),
        "mode": mode,
        "git": git_rev(),
        "target": common.label(cfg),
        "returncode": result.returncode,
        "stats": stats,
        "ok": result.returncode == 0 and (stats["changed"] == 0 if args.check else True),
    }
    LAST_SYNC.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
