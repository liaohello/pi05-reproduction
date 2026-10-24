#!/usr/bin/env python3
"""在服务器上跑一次 LIBERO episode（起服务→仿真→成功判定），并把结果写进台账。

用法：
    python3 deploy/smoke.py --suite libero_spatial --tasks 0 --episodes 1
    python3 deploy/smoke.py --suite libero_10 --tasks 0 --episodes 1 --gpu 1 --no-record

默认 GPU 由远程脚本挑选空闲显存最大的一张；显存不足 18 GiB 会拒绝启动。
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROOT = common.ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--tasks", default="0", help="逗号分隔的 task id，或 all")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--replan", type=int, default=10)
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--timeout", type=int, default=2700)
    parser.add_argument("--no-record", action="store_true", help="只跑，不写台账")
    args = parser.parse_args()

    cfg = common.load_config()
    if not common.key_path(cfg).is_file():
        print(f"[BLOCKED] 缺少私钥 {common.key_path(cfg)}")
        return 2

    env = {
        "PI05_REMOTE_ROOT": cfg["remote"]["root"],
        "PI05_CKPT": cfg["assets"]["checkpoint"],
        "PI05_TOKENIZER": cfg["assets"]["tokenizer"],
        "PI05_SERVER_IMAGE": next(iter(cfg["images"])),
        "PI05_CLIENT_IMAGE": list(cfg["images"])[1],
        "PI05_SMOKE_SUITE": args.suite,
        "PI05_SMOKE_TASKS": args.tasks,
        "PI05_SMOKE_EPISODES": str(args.episodes),
        "PI05_SMOKE_REPLAN": str(args.replan),
        "PI05_GPU": args.gpu,
    }
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    script = (ROOT / "scripts" / "deploy" / "remote_smoke.sh").read_text(encoding="utf-8")

    print(f"远程执行冒烟：suite={args.suite} tasks={args.tasks} episodes={args.episodes} "
          f"replan={args.replan} gpu={args.gpu}")
    result = common.run(common.ssh_argv(cfg) + [f"env {prefix} bash -s"],
                        input_text=script, timeout=args.timeout)
    transcript = result.stdout + result.stderr
    print(transcript[-4000:])

    if result.returncode != 0:
        print(f"[FAILED] 远程退出码 {result.returncode}")
        return 1

    info = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            if key in {"RESULT_JSON", "VIDEO_DIR", "SUMMARY"}:
                info[key.strip()] = value.strip()

    result_json = info.get("RESULT_JSON", "")
    if not result_json:
        print("[FAILED] 未取得 RESULT_JSON")
        return 1

    fetched = common.run(common.ssh_argv(cfg) + [f"cat {shlex.quote(result_json)}"], timeout=120)
    if fetched.returncode != 0:
        print(f"[FAILED] 读取 evaluation.json 失败：{fetched.stderr[-500:]}")
        return 1
    evaluation = json.loads(fetched.stdout)
    episodes = int(evaluation.get("completed_episodes", 0))
    successes = int(evaluation.get("successes", 0))
    sr = evaluation.get("success_rate")

    print(f"episode 完成 {episodes}，成功 {successes}，SR={sr}")

    if args.no_record:
        return 0 if episodes else 1

    command = (
        f"PI05_SMOKE_SUITE={args.suite} PI05_SMOKE_TASKS={args.tasks} "
        f"PI05_SMOKE_EPISODES={args.episodes} PI05_SMOKE_REPLAN={args.replan} "
        f"PI05_GPU={args.gpu} bash scripts/deploy/remote_smoke.sh"
    )
    record = [
        sys.executable, str(ROOT / "scripts" / "track" / "record.py"),
        "--phase", "D3" if episodes < 50 else "D4",
        "--kind", "smoke",
        "--suite", args.suite,
        "--tasks", str(len(str(args.tasks).split(",")) if args.tasks != "all" else "10"),
        "--episodes", str(episodes),
        "--successes", str(successes),
        "--command", command,
        "--artifact", result_json,
        "--artifact", info.get("VIDEO_DIR", ""),
        "--note", f"seed={evaluation.get('seed')} replan_steps={evaluation.get('replan_steps')} "
                  f"settle={evaluation.get('settle_steps')} max_steps={evaluation.get('max_steps')}",
    ]
    subprocess.run(record, check=False)
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
