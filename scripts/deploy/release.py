#!/usr/bin/env python3
"""释放服务器上本项目占用的资源：容器、网络，并报告 GPU 是否已归还。

只处理带 `pi05.repro=1` 标签或 `pi05-` 前缀的容器，以及同标签的网络；
不碰机器上其他人的容器。运行产物（runs/）默认保留，作为 D3 证据。

用法：
    python3 deploy/release.py
    python3 deploy/release.py --purge-runs   # 连 runs/ 目录一起删
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

PAYLOAD = r"""
set -u
echo "[1] 本项目容器"
ids="$(docker ps -aq --filter label=pi05.repro=1 2>/dev/null | tr '\n' ' ')"
if [ -n "${ids// /}" ]; then
  docker rm -f $ids >/dev/null && echo "  已删除: $ids"
else
  echo "  无"
fi
echo "[2] 名称兜底（pi05-*）"
left="$(docker ps -aq --filter 'name=^pi05-' 2>/dev/null | tr '\n' ' ')"
if [ -n "${left// /}" ]; then
  docker rm -f $left >/dev/null && echo "  已删除: $left"
else
  echo "  无"
fi
echo "[3] 本项目网络"
nets="$(docker network ls --filter label=pi05.repro=1 --format '{{.Name}}' 2>/dev/null | tr '\n' ' ')"
if [ -n "${nets// /}" ]; then
  for n in $nets; do docker network rm "$n" >/dev/null && echo "  已删除: $n"; done
else
  echo "  无"
fi
echo "[4] 残留推理进程"
if pgrep -af 'serve_policy\.py' >/dev/null 2>&1; then
  pgrep -af 'serve_policy\.py'
else
  echo "  无"
fi
echo "[5] GPU 现状"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "[6] 其他容器（不属于本项目，未改动）"
docker ps --format '{{.Names}} | {{.Status}}' | sed 's/^/  /'
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--purge-runs", action="store_true", help="同时删除远程 runs/ 运行产物")
    args = parser.parse_args()

    cfg = common.load_config()
    if not common.key_path(cfg).is_file():
        print(f"[BLOCKED] 缺少私钥 {common.key_path(cfg)}")
        return 2

    script = PAYLOAD
    if args.purge_runs:
        script += f'\nrm -rf {cfg["remote"]["runs_dir"]} && echo "[7] 已删除 runs/"\n'

    result = common.run(common.ssh_argv(cfg) + ["bash -s"], input_text=script, timeout=180)
    print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
