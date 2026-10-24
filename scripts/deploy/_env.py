#!/usr/bin/env python3
"""Print resolved deployment variables for manual shell use.

    eval "$(python3 scripts/deploy/_env.py)"

Values come from the same resolver as the other scripts, so nothing here is
hard-coded and no secret is printed unless it is already in your environment.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

cfg = common.load_config_or_exit()
values = {
    "PI05_REMOTE_HOST": cfg["ssh"]["host"],
    "PI05_REMOTE_PORT": str(cfg["ssh"]["port"]),
    "PI05_REMOTE_USER": cfg["ssh"]["user"],
    "PI05_SSH_KEY_PATH": str(common.key_path(cfg)),
    "PI05_REMOTE_ROOT": cfg["remote"]["root"],
    "PI05_RUNS_DIR": cfg["remote"]["runs_dir"],
    "PI05_CKPT": cfg["assets"].get("checkpoint", ""),
    "PI05_NORM_STATS": cfg["assets"].get("norm_stats", ""),
    "PI05_TOKENIZER": cfg["assets"].get("tokenizer", ""),
    "PI05_LEROBOT_DATA": cfg["assets"].get("lerobot_data", ""),
}
for name, value in values.items():
    print(f"export {name}={shlex.quote(str(value))}")
