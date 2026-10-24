#!/usr/bin/env python3
"""把一次运行写入追加式台账 track/ledger.jsonl。

每次冒烟 / 评估 / 训练 / 启动服务都应记录一次，字段固定，便于事后回溯：
代码版本（git）、镜像 ID、权重与归一化统计哈希、命令、指标、产物。

示例：
    python3 track/record.py --phase D3 --kind smoke --suite libero_spatial \\
        --tasks 1 --episodes 1 --successes 1 \\
        --command "docker run ... main.py --task-suite-name libero_spatial" \\
        --artifact <remote-workdir>/runs/<run-id>/task_00_episode_000_success.mp4

    python3 track/record.py --phase D4 --kind eval --suite libero_10 \\
        --tasks 10 --episodes 500 --successes 481 --metric sr=0.962 --metric spl=0.91
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))

import common  # noqa: E402

ROOT = common.ROOT
LEDGER = ROOT / "experiments" / "ledger.jsonl"
PHASES = ROOT / "experiments" / "phases.json"
KINDS = {"smoke", "eval", "train", "serve", "verify"}
STATUSES = {"success", "failed", "aborted"}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def git_rev() -> str:
    result = common.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])
    if result.returncode != 0:
        return "no-git"
    dirty = common.run(["git", "-C", str(ROOT), "status", "--porcelain"]).stdout.strip()
    return result.stdout.strip() + ("-dirty" if dirty else "")


def existing_runs() -> list[dict]:
    if not LEDGER.is_file():
        return []
    runs = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            runs.append(json.loads(line))
    return runs


def parse_metrics(items: list[str]) -> dict:
    metrics: dict[str, float] = {}
    for item in items:
        key, _, value = item.partition("=")
        try:
            metrics[key.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"--metric 需要 key=number 形式，收到：{item}")
    return metrics


def main() -> int:
    cfg = common.load_config()
    phases = {p["id"] for p in json.loads(PHASES.read_text(encoding="utf-8"))["phases"]}

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=sorted(phases))
    parser.add_argument("--kind", required=True, choices=sorted(KINDS))
    parser.add_argument("--suite", default="", help="libero_spatial / libero_object / libero_goal / libero_10")
    parser.add_argument("--tasks", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--successes", type=int, default=0)
    parser.add_argument("--metric", action="append", default=[], help="key=value，可重复")
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint", default=Path(cfg["assets"]["checkpoint"]).name,
                        help="checkpoint id, not a path")
    parser.add_argument("--command", default="", help="完整启动命令，便于复跑")
    parser.add_argument("--artifact", action="append", default=[], help="产物路径，可重复")
    parser.add_argument("--status", default="success", choices=sorted(STATUSES))
    parser.add_argument("--note", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.kind in {"smoke", "eval"} and args.episodes <= 0:
        raise SystemExit("smoke / eval 记录必须给出 --episodes")
    if args.successes > args.episodes:
        raise SystemExit("--successes 不能大于 --episodes")

    metrics = parse_metrics(args.metric)
    if args.episodes:
        metrics.setdefault("sr", round(args.successes / args.episodes, 6))

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%dT%H%M%S")
    record = {
        "run_id": f"{stamp}-{args.phase}-{args.kind}" + (f"-{args.suite}" if args.suite else ""),
        "ts": now(),
        "phase": args.phase,
        "kind": args.kind,
        "suite": args.suite,
        "tasks": args.tasks,
        "episodes": args.episodes,
        "successes": args.successes,
        "metrics": metrics,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "images": cfg["images"],
        "norm_stats_sha256": cfg["assets"]["norm_stats_sha256"],
        "git": git_rev(),
        "command": args.command,
        "artifacts": [common.sanitize_path(cfg, a) for a in args.artifact],
        "status": args.status,
        "notes": args.note,
    }

    if args.dry_run:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(existing_runs())
    print(f"已记录 {record['run_id']}  (台账共 {total} 条)")
    print(f"  {record['kind']}/{record['suite'] or '-'} "
          f"episodes={record['episodes']} successes={record['successes']} "
          f"sr={metrics.get('sr', 'n/a')} status={record['status']}")
    print("  下一步：python3 track/status.py 查看阶段门禁")
    return 0


if __name__ == "__main__":
    sys.exit(main())
