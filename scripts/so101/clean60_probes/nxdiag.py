"""Dump the pinned runtime parameter paths and the LoRA config actually built."""
import pathlib
import sys

sys.path.insert(0, "/work/scripts/so101")
import flax.nnx as nnx

import openpi_so101

RUN = pathlib.Path("/work/runs/so101/clean60_runtime")
mc = openpi_so101.make_model_config("on", paligemma_variant="gemma_2b_lora",
                                    action_expert_variant="gemma_300m")
model = mc.create(0)
graphdef, state = nnx.split(model)
rows = []
for key, value in sorted(state.raw_mapping.items()):
    name = str(key)
    node = value
    shape = getattr(getattr(node, "value", node), "shape", None)
    rows.append(f"{name} | {type(node).__name__} | {shape}")
RUN.joinpath("parameter_paths.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"tensors_top_level={len(rows)}")
print("\n".join(rows[:14]))
print("llm_1_paths", sum(1 for r in rows if "llm_1" in r))
print("img_paths", sum(1 for r in rows if "img" in r))
print("lora_paths", sum(1 for r in rows if "lora" in r))
