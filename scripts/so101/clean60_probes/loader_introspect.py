"""Print the pinned image's weight-loader and checkpoint-restore call signatures."""
import inspect
import openpi.training.weight_loaders as wl
import openpi.training.checkpoints as ck

print("=== CheckpointWeightLoader ===")
print(inspect.getsource(wl.CheckpointWeightLoader))
print("=== weight_loaders module API ===")
print([n for n in dir(wl) if not n.startswith("_")])
print("=== checkpoints.restore_state ===")
try:
    print(inspect.signature(ck.restore_state))
except Exception as error:
    print("sig failed", error)
src = inspect.getsource(ck)
for marker in ("def restore_state", "def load_params", "PyTreeCheckpointer", "def _restore"):
    idx = src.find(marker)
    print(f"--- {marker} @{idx}")
    if idx >= 0:
        print(src[idx:idx + 700])
