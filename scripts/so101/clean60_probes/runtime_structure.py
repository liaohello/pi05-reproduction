"""Read-only probe of the pinned OpenPI runtime: model module names and sharding.

Runs inside openpi-server with no GPU.  Prints the attribute tree the pi0.5 model
builds (which decides the freeze-filter regexes) and the trainer lines that turn
TrainConfig.batch_size / fsdp_devices into device sharding.
"""
from __future__ import annotations

import inspect
import pathlib
import re

ROOT = pathlib.Path("/opt/openpi")


def show(path: pathlib.Path, pattern: str, *, limit: int = 200) -> None:
    if not path.is_file():
        print(f"MISSING {path}")
        return
    print(f"=== {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if re.search(pattern, line) and index <= limit:
            print(f"{index}: {line}")


import openpi.models.pi0_config as pi0_config  # noqa: E402
import openpi.models.pi0 as pi0  # noqa: E402
import openpi.training.config as training_config  # noqa: E402

print("OPENPI PACKAGE", pathlib.Path(pi0.__file__).parent)
print("PI0CONFIG SOURCE")
print(inspect.getsource(pi0_config.Pi0Config))
for name in ("src/openpi/models/pi0.py", "src/openpi/models/gemma.py",
             "src/openpi/shared/nnx_utils.py", "src/openpi/models/lora.py"):
    show(ROOT / name, r"class |nnx\.|self\.[a-zA-Z_]+ *=|lora|LoRA|PaliGemma|img|llm|expert|PathRegex")
train_entry = next(iter([p for p in ROOT.rglob("train.py")]), None)
if train_entry:
    show(train_entry, r"shard|fsdp|batch_size|axis|DataSharding|replicat|num_train_steps|progress", limit=400)
print("FREEZE HELPERS")
print([name for name in dir(training_config) if "Freeze" in name or "filter" in name.lower()])
