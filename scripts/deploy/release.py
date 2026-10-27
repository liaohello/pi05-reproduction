#!/usr/bin/env python3
"""Release only containers explicitly registered by ``scripts/so101/e4t_resources.py``.

The former label/name-prefix cleanup is deliberately disabled. A missing
registry means no container may be stopped, and run artifacts are never purged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

PAYLOAD = r"""
set -u
if [ ! -f "$REGISTRY" ]; then
  echo "[BLOCKED] no exact-owner registry: refusing destructive container cleanup"
  echo "[GPU] current state (read-only)"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /'
  exit 2
fi
python3 scripts/so101/e4t_resources.py --registry "$REGISTRY" verify || exit 2
python3 scripts/so101/e4t_resources.py --registry "$REGISTRY" stop \
  --run-id "$RUN_ID" --all-this-run --confirm STOP-REGISTERED-EXACT-ENTRIES
echo "[GPU] current state"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /'
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True,
                        help="private registry path, relative to the remote workdir")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--purge-runs", action="store_true",
                        help="accepted only to preserve the old CLI; always refused")
    args = parser.parse_args()

    cfg = common.load_config_or_exit()
    if args.purge_runs:
        print("[BLOCKED] --purge-runs is disabled; run artifacts are retained", file=sys.stderr)
        return 2
    if not common.key_path(cfg).is_file():
        print(f"[BLOCKED] 缺少私钥 {common.key_path(cfg)}")
        return 2

    script = f'REGISTRY="{args.registry}"\nRUN_ID="{args.run_id}"\n' + PAYLOAD
    result = common.run(common.ssh_argv(cfg) + ["bash -s"], input_text=script, timeout=180)
    print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
