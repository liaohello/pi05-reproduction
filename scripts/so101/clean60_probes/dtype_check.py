"""Why does the frozen language tower differ from the released checkpoint?

Casts the stored base tensor through the checkpoint's own dtype and compares again:
a frozen tensor must be bit-identical once both sides share the storage dtype.
"""
import pathlib
import numpy as np
import openpi.models.model as _model

BASE = pathlib.Path("/base")
CKPT = pathlib.Path("/ckpt")
SAMPLES = {
    "language_backbone": "PaliGemma/llm/embedder/input_embedding",
    "action_expert": "PaliGemma/llm/final_norm_1/Dense_0/bias",
    "vision_encoder": "PaliGemma/img/Transformer/encoder_norm/bias",
    "projection": "action_in_proj/kernel",
}


def walk(node, prefix, out):
    if isinstance(node, dict):
        for key, child in node.items():
            walk(child, f"{prefix}/{key}" if prefix else str(key), out)
    elif hasattr(node, "value"):
        walk(node.value, prefix, out)
    elif hasattr(node, "dtype"):
        out[prefix] = node


def get(tree, suffix):
    for name, array in tree.items():
        if name.endswith(suffix):
            return name, np.asarray(array)
    return None, None


base, trained = {}, {}
walk(_model.restore_params(str(BASE), restore_type=np.ndarray), "", base)
walk(_model.restore_params(str(CKPT), restore_type=np.ndarray), "", trained)
print("base_tensors", len(base), "trained_tensors", len(trained))
print("base_dtypes", sorted({str(a.dtype) for a in base.values()}))
print("trained_dtypes", sorted({str(a.dtype) for a in trained.values()}))
for group, suffix in SAMPLES.items():
    bn, ba = get(base, suffix)
    tn, ta = get(trained, suffix)
    if ba is None or ta is None:
        print(f"{group:20s} MISSING base={bn} trained={tn}")
        continue
    raw = float(np.max(np.abs(ta.astype(np.float64) - ba.astype(np.float64))))
    cast = ba.astype(ta.dtype).astype(np.float32)
    after = float(np.max(np.abs(ta.astype(np.float32) - cast)))
    bit = bool(np.array_equal(ta.astype(ta.dtype), ba.astype(ta.dtype)))
    print(f"{group:20s} base={ba.dtype}{ba.shape} trained={ta.dtype}{ta.shape} "
          f"raw_delta={raw:.3e} same_dtype_delta={after:.3e} bit_equal_after_cast={bit}")
