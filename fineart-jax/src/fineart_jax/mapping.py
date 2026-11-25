"""Declared, reversible source-tensor rules.

Kept separate from the loader so the same declaration can be used for forward
mapping, inverse mapping and strict shape/dtype validation.
"""
from __future__ import annotations

import numpy as np

V = "model.paligemma_with_expert.paligemma.model.vision_tower."
L = "model.paligemma_with_expert.paligemma.model.language_model."
P = "model.paligemma_with_expert.paligemma.model.multi_modal_projector."
E = "model.paligemma_with_expert.gemma_expert.model."


def per_layer(name: str, count: int, fn):
    # Always hand the transform one layer slice, never the whole stacked tensor.
    def make(index):
        def slice_fn(array):
            layer = array[index]
            return fn(layer) if fn else layer
        return slice_fn
    return [(name.format(i=i), make(i)) for i in range(count)]


def rules():
    r = {}

    def add(key, targets):
        assert key not in r, key
        r[key] = targets

    # Vision tower
    add("img.embedding.kernel", [(V + "embeddings.patch_embedding.weight", lambda a: a.transpose(3, 2, 0, 1))])
    add("img.embedding.bias", [(V + "embeddings.patch_embedding.bias", None)])
    add("img.pos_embedding", [(V + "embeddings.position_embedding.weight", lambda a: a[0])])
    add("img.head.kernel", [(P + "linear.weight", lambda a: a.T)])
    add("img.head.bias", [(P + "linear.bias", None)])
    add("img.Transformer.encoder_norm.scale", [(V + "post_layernorm.weight", None)])
    add("img.Transformer.encoder_norm.bias", [(V + "post_layernorm.bias", None)])
    for j, suffix in ((0, "layer_norm1"), (1, "layer_norm2")):
        for part, field in (("scale", "weight"), ("bias", "bias")):
            add(f"img.Transformer.encoderblock.LayerNorm_{j}.{part}",
                per_layer(V + f"encoder.layers.{{i}}.{suffix}.{field}", 27, None))
    for field, target in (("bias", "fc1.bias"), ("kernel", "fc1.weight")):
        fn = (lambda a: a.T) if field == "kernel" else None
        add(f"img.Transformer.encoderblock.MlpBlock_0.Dense_0.{field}",
            per_layer(V + f"encoder.layers.{{i}}.mlp.{target}", 27, fn))
    for field, target in (("bias", "fc2.bias"), ("kernel", "fc2.weight")):
        fn = (lambda a: a.T) if field == "kernel" else None
        add(f"img.Transformer.encoderblock.MlpBlock_0.Dense_1.{field}",
            per_layer(V + f"encoder.layers.{{i}}.mlp.{target}", 27, fn))
    for part, target in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj"), ("out", "out_proj")):
        kernel_fn = (
            (lambda a: a.reshape(-1, a.shape[-1]).T)
            if part == "out"
            else (lambda a: a.reshape(a.shape[0], -1).T)
        )
        add(f"img.Transformer.encoderblock.MultiHeadDotProductAttention_0.{part}.kernel",
            per_layer(V + f"encoder.layers.{{i}}.self_attn.{target}.weight", 27, kernel_fn))
        add(f"img.Transformer.encoderblock.MultiHeadDotProductAttention_0.{part}.bias",
            per_layer(V + f"encoder.layers.{{i}}.self_attn.{target}.bias", 27,
                      lambda a: a.reshape(-1)))

    # VLM language tower
    add("llm.embedder.input_embedding", [
        (L + "embed_tokens.weight", None),
        ("model.paligemma_with_expert.paligemma.lm_head.weight", None),
    ])
    add("llm.final_norm.scale", [(L + "norm.weight", None)])
    add("llm.layers.pre_attention_norm.scale",
        per_layer(L + "layers.{i}.input_layernorm.weight", 18, None))
    add("llm.layers.pre_ffw_norm.scale",
        per_layer(L + "layers.{i}.post_attention_layernorm.weight", 18, None))
    add("llm.layers.attn.q_einsum.w",
        per_layer(L + "layers.{i}.self_attn.q_proj.weight", 18,
                  lambda a: a.transpose(0, 2, 1).reshape(a.shape[0] * a.shape[2], a.shape[1])))
    add("llm.layers.attn.kv_einsum.w", [
        (L + f"layers.{{i}}.self_attn.{head}_proj.weight".format(i=i),
         lambda a, k=k, i=i: a[i][k].transpose(0, 2, 1).reshape(a[i][k].shape[0] * a[i][k].shape[2], a[i][k].shape[1]))
        for k, head in enumerate(("k", "v"))
        for i in range(18)
    ])
    add("llm.layers.attn.attn_vec_einsum.w",
        per_layer(L + "layers.{i}.self_attn.o_proj.weight", 18,
                  lambda a: a.reshape(a.shape[0] * a.shape[1], a.shape[2]).T))
    add("llm.layers.mlp.gating_einsum", [
        ((L + f"layers.{{i}}.mlp.{part}_proj.weight").format(i=i), lambda a, k=k, i=i: a[i][k].T)
        for k, part in enumerate(("gate", "up")) for i in range(18)
    ])
    add("llm.layers.mlp.linear",
        per_layer(L + "layers.{i}.mlp.down_proj.weight", 18, lambda a: a.T))

    # Action expert
    for src, target in (
        ("llm.layers.pre_attention_norm_1.Dense_0.bias", "input_layernorm.dense.bias"),
        ("llm.layers.pre_ffw_norm_1.Dense_0.bias", "post_attention_layernorm.dense.bias"),
        ("llm.final_norm_1.Dense_0.bias", "norm.dense.bias"),
    ):
        if "layers." in src:
            add(src, per_layer(E + f"layers.{{i}}.{target}", 18, None))
        else:
            add(src, [(E + target, None)])
    for src, target in (
        ("llm.layers.pre_attention_norm_1.Dense_0.kernel", "input_layernorm.dense.weight"),
        ("llm.layers.pre_ffw_norm_1.Dense_0.kernel", "post_attention_layernorm.dense.weight"),
        ("llm.final_norm_1.Dense_0.kernel", "norm.dense.weight"),
    ):
        if "layers." in src:
            add(src, per_layer(E + f"layers.{{i}}.{target}", 18, lambda a: a.T))
        else:
            add(src, [(E + target, lambda a: a.T)])
    add("llm.layers.attn.q_einsum_1.w",
        per_layer(E + "layers.{i}.self_attn.q_proj.weight", 18,
                  lambda a: a.transpose(0, 2, 1).reshape(a.shape[0] * a.shape[2], a.shape[1])))
    add("llm.layers.attn.kv_einsum_1.w", [
        (E + f"layers.{{i}}.self_attn.{head}_proj.weight".format(i=i),
         lambda a, k=k, i=i: a[i][k].transpose(0, 2, 1).reshape(a[i][k].shape[0] * a[i][k].shape[2], a[i][k].shape[1]))
        for k, head in enumerate(("k", "v")) for i in range(18)
    ])
    add("llm.layers.attn.attn_vec_einsum_1.w",
        per_layer(E + "layers.{i}.self_attn.o_proj.weight", 18,
                  lambda a: a.reshape(a.shape[0] * a.shape[1], a.shape[2]).T))
    add("llm.layers.mlp_1.gating_einsum", [
        ((E + f"layers.{{i}}.mlp.{part}_proj.weight").format(i=i), lambda a, k=k, i=i: a[i][k].T)
        for k, part in enumerate(("gate", "up")) for i in range(18)
    ])
    add("llm.layers.mlp_1.linear",
        per_layer(E + "layers.{i}.mlp.down_proj.weight", 18, lambda a: a.T))

    # Conditioning
    for source, target in (
        ("action_in_proj.kernel", "model.action_in_proj.weight"),
        ("action_out_proj.kernel", "model.action_out_proj.weight"),
        ("time_mlp_in.kernel", "model.time_mlp_in.weight"),
        ("time_mlp_out.kernel", "model.time_mlp_out.weight"),
    ):
        add(source, [(target, lambda a: a.T)])
    for source, target in (
        ("action_in_proj.bias", "model.action_in_proj.bias"),
        ("action_out_proj.bias", "model.action_out_proj.bias"),
        ("time_mlp_in.bias", "model.time_mlp_in.bias"),
        ("time_mlp_out.bias", "model.time_mlp_out.bias"),
    ):
        add(source, [(target, None)])
    return r




SOURCE_RULES = rules()


def source_key(name: str) -> str:
    """Strip the Orbax tree prefixes to the key used by SOURCE_RULES."""
    prefix = "params.PaliGemma."
    if name.startswith(prefix):
        return name[len(prefix):]
    if name.startswith("params."):
        return name[len("params."):]
    raise ValueError("not a checkpoint parameter name: " + name)


def _stack(values):
    return np.stack([np.ascontiguousarray(value) for value in values], axis=0)


def inverse_source(key: str, values: list[np.ndarray], source_shape: tuple[int, ...]) -> np.ndarray:
    """Reverse SOURCE_RULES[key] using the original shape as the axis witness."""
    shape = tuple(source_shape)
    if key == "img.embedding.kernel":
        # forward is source.transpose(3,2,0,1); the inverse reads it back as
        # [out,c,h,w] -> [h,w,c,out].
        return values[0].transpose(2, 3, 1, 0)
    if key == "img.embedding.bias" or key.startswith("img.Transformer.encoder_norm."):
        return values[0]
    if key == "img.pos_embedding":
        return values[0][None, ...]
    if key == "img.head.kernel":
        return values[0].T
    if key == "img.head.bias":
        return values[0]

    if "MultiHeadDotProductAttention_0." in key:
        part, field = key.split(".")[-2:]
        if field == "bias":
            return _stack([value.reshape(shape[1:]) for value in values])
        if part == "out":
            return _stack([value.T.reshape(shape[1:]) for value in values])
        return _stack([value.T.reshape(shape[1:]) for value in values])

    if "MlpBlock_0.Dense_" in key:
        field = key.split(".")[-1]
        return _stack([(value.T if field == "kernel" else value) for value in values])

    if "LayerNorm_" in key or key in (
        "llm.layers.pre_attention_norm.scale", "llm.layers.pre_ffw_norm.scale"
    ):
        return _stack(values)

    if key == "llm.embedder.input_embedding":
        # The second target is a declared clone, never an independent source.
        return values[0]
    if key == "llm.final_norm.scale" or key == "llm.final_norm_1.Dense_0.bias":
        return values[0]
    if key == "llm.final_norm_1.Dense_0.kernel":
        return values[0].T

    if key.endswith("Dense_0.bias"):
        return _stack(values)
    if key.endswith("Dense_0.kernel"):
        return _stack([value.T for value in values])

    if "kv_einsum" in key:
        layers = len(values) // 2
        per_layer = []
        for index in range(layers):
            # The kv rule enumerates k for every layer first, then v: the target
            # order is [k_layer0..k_layerN, v_layer0..v_layerN].
            # forward: [K,D,H] -> transpose(0,2,1) -> [K,H,D] -> [K*H, D]
            k = values[index].reshape(shape[2], shape[4], shape[3]).transpose(0, 2, 1)
            v = values[layers + index].reshape(shape[2], shape[4], shape[3]).transpose(0, 2, 1)
            per_layer.append(np.stack([k, v], axis=0))
        return np.stack(per_layer, axis=0)

    if "gating_einsum" in key:
        layers = len(values) // 2
        per_layer = []
        for index in range(layers):
            # Same enumeration as the kv rule: gate for every layer first, then up.
            gate = values[index].T
            up = values[layers + index].T
            per_layer.append(np.stack([gate, up], axis=0))
        return np.stack(per_layer, axis=0)

    if key.endswith("q_einsum.w") or key.endswith("q_einsum_1.w"):
        return _stack([
            value.reshape(shape[1], shape[3], shape[2]).transpose(0, 2, 1) for value in values
        ])
    if "attn_vec_einsum" in key:
        return _stack([value.T.reshape(shape[1:]) for value in values])
    if key.endswith("mlp.linear") or key.endswith("mlp_1.linear"):
        return _stack([value.T for value in values])

    if key.endswith(".kernel"):
        if key.startswith("action") or key.startswith("time_mlp"):
            # Conditioning MLPs are single matrices: [out, in] -> [in, out].
            return values[0].T
    if key.endswith(".bias"):
        return values[0]

    raise ValueError("no inverse rule for " + key)
