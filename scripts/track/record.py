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
import re
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
SCOPES = {"libero", "so101"}
SO101_PHASES = {"E0", "E1", "E2", "E3"}


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
    parser.add_argument("--scope", default="libero", choices=sorted(SCOPES),
                        help="experiment branch; LIBERO gates only consume scope=libero")
    parser.add_argument("--phase", required=True,
                        help="D0-D5 for scope=libero, E0-E3 for scope=so101")
    parser.add_argument("--substage", default="",
                        help="optional substage, e.g. E3.1-G1; kept in the run id and record")
    parser.add_argument("--kind", required=True, choices=sorted(KINDS))
    parser.add_argument("--suite", default="", help="libero_spatial / libero_object / libero_goal / libero_10")
    parser.add_argument("--tasks", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--successes", type=int, default=0)
    parser.add_argument("--metric", action="append", default=[], help="key=value，可重复")
    parser.add_argument("--config", default=None,
                        help="config id (LIBERO defaults to pi05_libero; SO101 records must name one)")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint id, not a path")
    parser.add_argument("--config-sha256", default="", help="content fingerprint of the run config")
    parser.add_argument("--checkpoint-sha256", default="", help="content fingerprint of the checkpoint")
    parser.add_argument("--dataset-sha256", default="", help="content fingerprint of the dataset")
    parser.add_argument("--norm-stats-sha256", default="",
                        help="norm-stats fingerprint; LIBERO defaults to the pinned value")
    parser.add_argument("--command", default="", help="完整启动命令，便于复跑")
    parser.add_argument("--artifact", action="append", default=[], help="产物路径，可重复")
    parser.add_argument("--status", default="success", choices=sorted(STATUSES))
    parser.add_argument("--note", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.scope == "libero":
        if args.phase not in phases:
            raise SystemExit(f"scope=libero 只接受 phases.json 中的阶段：{sorted(phases)}")
    elif args.phase not in SO101_PHASES:
        raise SystemExit(f"scope=so101 只接受 {sorted(SO101_PHASES)}，收到：{args.phase}")

    if args.scope == "so101" and args.kind in {"smoke", "eval", "train"}:
        if not args.config or not args.checkpoint:
            raise SystemExit("SO101 run 记录必须显式给出 --config 与 --checkpoint，"
                             "不能沿用 LIBERO 默认值")

    if args.kind in {"smoke", "eval"} and args.episodes <= 0:
        raise SystemExit("smoke / eval 记录必须给出 --episodes")
    if args.successes > args.episodes:
        raise SystemExit("--successes 不能大于 --episodes")

    metrics = parse_metrics(args.metric)
    if args.episodes:
        metrics.setdefault("sr", round(args.successes / args.episodes, 6))

    config_id = args.config or "pi05_libero"
    checkpoint_id = args.checkpoint or Path(cfg["assets"]["checkpoint"]).name
    norm_stats = args.norm_stats_sha256 or (
        cfg["assets"]["norm_stats_sha256"] if args.scope == "libero" else "")

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%dT%H%M%S")
    substage = re.sub(r"[^A-Za-z0-9._-]+", "-", args.substage.strip()).strip("-")
    record = {
        "run_id": f"{stamp}-{args.scope}-{args.phase}-{args.kind}"
                  + (f"-{substage}" if substage else "")
                  + (f"-{args.suite}" if args.suite else ""),
        "ts": now(),
        "scope": args.scope,
        "phase": args.phase,
        "substage": substage,
        "kind": args.kind,
        "suite": args.suite,
        "tasks": args.tasks,
        "episodes": args.episodes,
        "successes": args.successes,
        "metrics": metrics,
        "config": config_id,
        "checkpoint": checkpoint_id,
        "images": cfg["images"] if args.scope == "libero" else {},
        "norm_stats_sha256": norm_stats,
        "fingerprints": {"config_sha256": args.config_sha256,
                         "checkpoint_sha256": args.checkpoint_sha256,
                         "dataset_sha256": args.dataset_sha256,
                         "norm_stats_sha256": norm_stats},
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
