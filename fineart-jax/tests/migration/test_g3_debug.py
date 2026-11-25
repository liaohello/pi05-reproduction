"""Small G3 operator checks, kept separate from the seven formal gate entries."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fineart_jax.nn.attention import rope
from fineart_jax.nn.joint import (
    expert_layer_with_cache,
    flow_matching_inputs,
    flow_mse_loss,
    joint_flow_forward,
    joint_layer,
    make_prefix_cache,
    prefix_cache_matches,
    prefix_layer_with_cache,
)
from fineart_jax.nn.norms import ada_rms
from tests.migration import support as S

ORACLE = S.PROJECT / "tests" / "migration" / "oracle" / "export_g3_layer.py"
HEADS, KV_HEADS, HEAD_DIM, CHUNK = 8, 2, 16, 4


def _npz_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture() -> tuple[dict, dict, Path]:
    roots = S.roots()
    evidence = S.evidence_dir("g3_debug")
    oracle_root = evidence / "oracle"
    manifests = []
    for run in ("run1", "run2"):
        target = oracle_root / run
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **S.reference_env(roots)}
        env.update({
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        })
        result = subprocess.run(
            [roots["REF_PYTHON"], str(ORACLE), "--out", str(target)],
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
            check=False,
        )
        assert result.returncode == 0, f"reference fixture export failed: {result.stderr[-3000:]}"
        manifests.append(json.loads((target / "g3_layer_fixture.json").read_text()))
    first, second = manifests
    first_hashes = {name: row["sha256"] for name, row in first["arrays"].items()}
    second_hashes = {name: row["sha256"] for name, row in second["arrays"].items()}
    assert first_hashes == second_hashes, "reference fixture arrays changed between exports"
    assert first["reference_source_sha256"] == second["reference_source_sha256"]
    assert first["nondegenerate_adarms"]["input_gate_max_abs"] > 0.1
    assert first["nondegenerate_adarms"]["input_gate_nonzero_count"] == HEAD_DIM * HEADS
    arrays_npz = np.load(oracle_root / "run1" / "g3_layer_fixture.npz")
    arrays = {name: arrays_npz[name] for name in arrays_npz.files}
    return arrays, first, evidence


def _params(arrays: dict[str, np.ndarray]) -> dict:
    result = {
        "prefix": {}, "expert": {}, "suffix_embedding": {}, "action_out_proj": {},
        "final_norms": {"prefix": {}, "expert": {}},
    }
    for name, value in arrays.items():
        if name.startswith("p__"):
            _, stream, group, *tail = name.split("__")
            node = result[stream].setdefault(group, {})
            if len(tail) == 1:
                node[tail[0]] = jnp.asarray(value)
            elif len(tail) == 2:
                node.setdefault(tail[0], {})[tail[1]] = jnp.asarray(value)
            else:
                raise AssertionError(f"invalid fixture parameter key: {name}")
        elif name.startswith("se__"):
            _, group, field = name.split("__")
            result["suffix_embedding"].setdefault(group, {})[field] = jnp.asarray(value)
        elif name.startswith("fn__"):
            _, stream, field = name.split("__")
            result["final_norms"][stream][field] = jnp.asarray(value)
        elif name.startswith("out__"):
            _, field = name.split("__")
            result["action_out_proj"][field] = jnp.asarray(value)
    return result


def _parameter_key(reference_name: str) -> str:
    stream, remainder = reference_name.split("__", 1)
    parts = remainder.split(".")
    if parts[0] == "input_layernorm":
        group, tail = "input_norm", parts[1:]
        if tail == ["dense", "weight"]:
            tail = ["dense_weight"]
        elif tail == ["dense", "bias"]:
            tail = ["dense_bias"]
    elif parts[0] == "post_attention_layernorm":
        group, tail = "post_norm", parts[1:]
        if tail == ["dense", "weight"]:
            tail = ["dense_weight"]
        elif tail == ["dense", "bias"]:
            tail = ["dense_bias"]
    elif parts[0] == "self_attn":
        group, tail = "attention", parts[1:]
        if tail[-1] == "weight":
            tail = tail[:-1]
    elif parts[0] == "mlp":
        group, tail = "mlp", parts[1:]
        if tail[-1] == "weight":
            tail = tail[:-1]
    else:
        raise AssertionError(reference_name)
    return "/".join([stream, group, *tail])


def _tree_get(tree: dict, slash_path: str):
    value = tree
    for component in slash_path.split("/"):
        value = value[component]
    return value


def _gradient_summary(tree: dict) -> dict:
    counts = {"none": 0, "numeric_zero": 0, "nonzero": 0}
    max_abs = 0.0
    for value in jax.tree_util.tree_leaves(tree):
        if value is None:
            counts["none"] += 1
            continue
        array = np.asarray(value)
        max_abs = max(max_abs, float(np.max(np.abs(array), initial=0.0)))
        counts["numeric_zero" if not np.any(array) else "nonzero"] += 1
    return {"counts": counts, "max_abs": max_abs}


def _max_tree_abs_diff(left, right) -> float:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    return max(
        (float(np.max(np.abs(np.asarray(a) - np.asarray(b)), initial=0.0))
         for a, b in zip(left_leaves, right_leaves, strict=True)),
        default=0.0,
    )


def _gpu_memory_snapshot() -> dict:
    stats = jax.devices()[0].memory_stats() or {}
    keys = ("bytes_in_use", "peak_bytes_in_use", "bytes_reserved", "peak_bytes_reserved",
            "pool_bytes", "peak_pool_bytes", "bytes_limit")
    return {key: int(stats[key]) for key in keys if key in stats}


def _compare_gradients(mode: str, grads: dict, arrays: dict, manifest: dict) -> dict:
    report = {"groups": {}, "max_relative_l2": 0.0, "min_cosine": 1.0, "max_abs_error": 0.0}
    states = manifest["gradient_states"]
    for stream in ("prefix", "expert"):
        group = {"reference_counts": {"none": 0, "numeric_zero": 0, "nonzero": 0},
                 "jax_counts": {"none": 0, "numeric_zero": 0, "nonzero": 0}}
        prefix = stream + "__"
        for reference_name in sorted(name for name in states if name.startswith(mode + "__" + prefix)):
            parameter_name = reference_name[len(mode) + 2 :]
            state = states[reference_name]
            group["reference_counts"][state] += 1
            ref = arrays["grad__" + reference_name]
            jax_grad = np.asarray(_tree_get(grads, _parameter_key(parameter_name)))
            grad_state = "numeric_zero" if not np.any(jax_grad) else "nonzero"
            group["jax_counts"][grad_state] += 1
            abs_error = float(np.max(np.abs(jax_grad - ref), initial=0.0))
            report["max_abs_error"] = max(report["max_abs_error"], abs_error)
            ref_norm = float(np.linalg.norm(ref.ravel()))
            if state == "nonzero" and ref_norm >= 1e-6:
                rel = float(np.linalg.norm((jax_grad - ref).ravel()) / max(ref_norm, 1e-12))
                actual_norm = float(np.linalg.norm(jax_grad.ravel()))
                cosine = float(np.dot(jax_grad.ravel(), ref.ravel()) /
                               max(actual_norm * ref_norm, 1e-30))
                report["max_relative_l2"] = max(report["max_relative_l2"], rel)
                report["min_cosine"] = min(report["min_cosine"], cosine)
                assert rel <= 1e-3, f"{mode}/{parameter_name}: relative gradient error {rel}"
                assert cosine >= 0.9999, f"{mode}/{parameter_name}: gradient cosine {cosine}"
            else:
                assert abs_error <= 1e-6, f"{mode}/{parameter_name}: near-zero gradient error {abs_error}"
        report["groups"][stream] = group
    return report


def test_g3_debug_one_layer_reference_components() -> None:
    arrays, manifest, evidence = _fixture()
    assert jax.default_backend() == "gpu", "G3 debug fixture must use the GPU backend"
    gpu_memory_highwater = {"after_backend_init": _gpu_memory_snapshot()}
    assert manifest["pi05_fallback_dispatch"]["same_batch"] is True
    assert manifest["pi05_fallback_dispatch"]["reduction"] == "none"
    assert manifest["pi05_fallback_dispatch"]["scope"].startswith("wrapper delegation only")
    params = _params(arrays)
    prefix = jnp.asarray(arrays["input__prefix"])
    suffix1 = jnp.asarray(arrays["input__suffix_k1"])
    cond1 = jnp.asarray(arrays["input__condition"])
    mask1 = jnp.asarray(arrays["input__mask_k1"])
    pos1 = jnp.asarray(arrays["input__positions_k1"])
    target = jnp.asarray(arrays["input__flow_target_suffix"])

    run = lambda p, x, c, m, pos, ki: joint_layer(
        p, prefix, x, m, pos, c, num_heads=HEADS, num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM, knowledge_insulation=ki,
    )
    run_on = jax.jit(lambda p, x, c, m, pos: run(p, x, c, m, pos, True))
    run_off = jax.jit(lambda p, x, c, m, pos: run(p, x, c, m, pos, False))
    prefix_on, suffix_on = run_on(params, suffix1, cond1, mask1, pos1)
    prefix_off, suffix_off = run_off(params, suffix1, cond1, mask1, pos1)
    suffix_on.block_until_ready()
    np.testing.assert_allclose(prefix_on, arrays["expected__k1_prefix"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(suffix_on, arrays["expected__k1_suffix_ki_on"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(prefix_off, arrays["expected__k1_prefix_ki_off"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(suffix_off, arrays["expected__k1_suffix_ki_off"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(suffix_on, suffix_off, atol=1e-5, rtol=1e-4)

    # Direct RoPE and non-degenerate AdaRMS parity against the real components.
    rope_positions = jnp.asarray(arrays["rope__positions"])
    q_actual = rope(jnp.asarray(arrays["rope__q_input"]), rope_positions)
    k_actual = rope(jnp.asarray(arrays["rope__k_input"]), rope_positions)
    np.testing.assert_allclose(q_actual, arrays["rope__q_expected"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(k_actual, arrays["rope__k_expected"], atol=1e-6, rtol=1e-6)
    ada_value, ada_gate = ada_rms(
        suffix1,
        params["expert"]["input_norm"]["dense_weight"],
        params["expert"]["input_norm"]["dense_bias"],
        cond1,
    )
    np.testing.assert_allclose(ada_value, arrays["adarms__input_value"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(ada_gate, arrays["adarms__input_gate"], atol=1e-5, rtol=1e-4)
    assert float(jnp.max(jnp.abs(ada_gate))) > 0.1

    # Prefix KV reuse equals the fused result; changing any prefix identity field rejects it.
    p_mask = mask1[:, :prefix.shape[1], :prefix.shape[1]]
    prefix_cached, prefix_key, prefix_value = prefix_layer_with_cache(
        params, prefix, pos1[:, :prefix.shape[1]], p_mask,
        num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM,
    )
    cache = make_prefix_cache(prefix, pos1[:, :prefix.shape[1]], p_mask, prefix_key, prefix_value)
    assert prefix_cache_matches(cache, prefix, pos1[:, :prefix.shape[1]], p_mask)
    changed_prefix = prefix.at[:, 0, 0].add(1.0)
    assert not prefix_cache_matches(cache, changed_prefix, pos1[:, :prefix.shape[1]], p_mask)
    suffix_cached = expert_layer_with_cache(
        params, suffix1, pos1[:, prefix.shape[1]:], mask1[:, prefix.shape[1]:, :],
        cond1, cache["key"], cache["value"], num_heads=HEADS, num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM, knowledge_insulation=True,
    )
    np.testing.assert_allclose(prefix_cached, prefix_on, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(suffix_cached, suffix_on, atol=1e-5, rtol=1e-4)

    # K=5 uses one prefix and five independent full-visibility blocks.
    suffix5 = jnp.asarray(arrays["input__suffix_k5"])
    cond5 = jnp.asarray(arrays["input__condition_k5"])
    mask5 = jnp.asarray(arrays["input__mask_k5"])
    pos5 = jnp.asarray(arrays["input__positions_k5"])
    prefix5, suffix_joint5 = run_on(params, suffix5, cond5, mask5, pos5)
    suffix_joint5.block_until_ready()
    np.testing.assert_allclose(prefix5, arrays["expected__k5_prefix"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(suffix_joint5, arrays["expected__k5_suffix"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(prefix5, prefix_on, atol=1e-5, rtol=1e-4)
    independent = []
    for draw in range(5):
        start = draw * CHUNK
        _, value = run_on(params, suffix5[:, start : start + CHUNK], cond1, mask1, pos1)
        independent.append(value)
    independent = jnp.stack(independent, axis=1)
    np.testing.assert_allclose(independent, suffix_joint5.reshape(1, 5, CHUNK, -1),
                               atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(independent, arrays["expected__k5_independent_suffix"],
                               atol=1e-5, rtol=1e-4)

    # The Flow fixture checks interpolation, noise-action target, actual-D reduction,
    # and a mixed predict_actions batch using the reference reduction helper.
    actions = jnp.asarray(arrays["flow__actions"])
    noise = jnp.asarray(arrays["flow__noise"])
    times = jnp.asarray(arrays["flow__time"])
    x_t, flow_target = flow_matching_inputs(actions, noise, times)
    np.testing.assert_allclose(x_t, arrays["flow__x_t_expected"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(flow_target, arrays["flow__target_expected"], atol=1e-6, rtol=1e-6)
    predict = jnp.asarray(arrays["flow__predict_actions"])
    velocity = jnp.asarray(arrays["flow__velocity"])
    loss_mean = flow_mse_loss(velocity, flow_target, 6, predict, "mean")
    loss_none = flow_mse_loss(velocity, flow_target, 6, predict, "none")
    np.testing.assert_allclose(loss_mean, arrays["flow__mean_expected"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(loss_none, arrays["flow__none_expected"], atol=1e-6, rtol=1e-6)
    for action_dim in (6, 12, 14, 32):
        routed_mean = flow_mse_loss(velocity, flow_target, action_dim, predict, "mean")
        routed_none = flow_mse_loss(velocity, flow_target, action_dim, predict, "none")
        np.testing.assert_allclose(
            routed_mean, arrays[f"flow__routed_mean_expected_d{action_dim}"],
            atol=1e-6, rtol=1e-6,
        )
        np.testing.assert_allclose(
            routed_none, arrays[f"flow__routed_none_expected_d{action_dim}"],
            atol=1e-6, rtol=1e-6,
        )

    # Integrated PI0.5 action/time embedding -> joint layer -> final norms -> velocity.
    flow_params = {
        "layers": [{"prefix": params["prefix"], "expert": params["expert"]}],
        "suffix_embedding": params["suffix_embedding"],
        "action_out_proj": params["action_out_proj"],
        "final_norms": params["final_norms"],
    }
    flow_actions = jnp.asarray(arrays["forward__actions"])
    flow_noise_k5 = jnp.asarray(arrays["forward__noise_k5"])
    flow_time_k5 = jnp.asarray(arrays["forward__time_k5"])
    flow_k5 = joint_flow_forward(
        flow_params, prefix, flow_actions, flow_noise_k5, flow_time_k5, mask5, pos5,
        actual_action_dim=6, draws=5, num_heads=HEADS, num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM, chunk_size=CHUNK, knowledge_insulation=True,
        predict_actions=jnp.asarray([True]), reduction="mean", use_prefix_cache=False,
    )
    flow_k5["velocity"].block_until_ready()
    np.testing.assert_allclose(flow_k5["x_t"], arrays["forward__x_t_expected_k5"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(flow_k5["target"], arrays["forward__target_expected_k5"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(flow_k5["suffix_embeddings"], arrays["forward__suffix_expected_k5"],
                               atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k5["condition"], arrays["forward__condition_expected_k5"],
                               atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k5["suffix"], arrays["forward__suffix_expected_k5_final"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k5["prefix"], arrays["forward__prefix_expected_k5"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k5["velocity"], arrays["forward__velocity_expected_k5"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k5["flow_loss"], arrays["forward__flow_loss_expected_k5"], atol=1e-5, rtol=1e-4)
    assert len(flow_k5["prefix_layer_outputs"]) == 1

    flow_k5_cached = joint_flow_forward(
        flow_params, prefix, flow_actions, flow_noise_k5, flow_time_k5, mask5, pos5,
        actual_action_dim=6, draws=5, num_heads=HEADS, num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM, chunk_size=CHUNK, knowledge_insulation=True,
        predict_actions=jnp.asarray([True]), reduction="mean", use_prefix_cache=True,
    )
    flow_k5_cached["velocity"].block_until_ready()
    np.testing.assert_allclose(flow_k5_cached["prefix"], flow_k5["prefix"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k5_cached["suffix"], flow_k5["suffix"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k5_cached["velocity"], flow_k5["velocity"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k5_cached["flow_loss"], flow_k5["flow_loss"], atol=1e-6, rtol=1e-6)
    for fused, cached in zip(flow_k5["prefix_layer_outputs"], flow_k5_cached["prefix_layer_outputs"], strict=True):
        np.testing.assert_allclose(cached, fused, atol=1e-5, rtol=1e-4)
    for fused, cached in zip(flow_k5["suffix_layer_outputs"], flow_k5_cached["suffix_layer_outputs"], strict=True):
        np.testing.assert_allclose(cached, fused, atol=1e-5, rtol=1e-4)
    gpu_memory_highwater["after_k5_fused_and_cached_forward"] = _gpu_memory_snapshot()

    flow_k1 = joint_flow_forward(
        flow_params, prefix, flow_actions, flow_noise_k5[0], flow_time_k5[0], mask1, pos1,
        actual_action_dim=6, draws=1, num_heads=HEADS, num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM, chunk_size=CHUNK, knowledge_insulation=True,
        predict_actions=jnp.asarray([True]), reduction="mean", use_prefix_cache=False,
    )
    np.testing.assert_allclose(flow_k1["x_t"], arrays["forward__x_t_expected_k1"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(flow_k1["target"], arrays["forward__target_expected_k1"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(flow_k1["suffix_embeddings"], arrays["forward__suffix_expected_k1"],
                               atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k1["condition"], arrays["forward__condition_expected_k1"],
                               atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k1["prefix"], arrays["forward__prefix_expected_k1"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k1["suffix"], arrays["forward__suffix_expected_k1_final"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k1["velocity"], arrays["forward__velocity_expected_k1"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k1["flow_loss"], arrays["forward__flow_loss_expected_k1"], atol=1e-5, rtol=1e-4)

    flow_k1_cached = joint_flow_forward(
        flow_params, prefix, flow_actions, flow_noise_k5[0], flow_time_k5[0], mask1, pos1,
        actual_action_dim=6, draws=1, num_heads=HEADS, num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM, chunk_size=CHUNK, knowledge_insulation=True,
        predict_actions=jnp.asarray([True]), reduction="mean", use_prefix_cache=True,
    )
    flow_k1_cached["velocity"].block_until_ready()
    np.testing.assert_allclose(flow_k1_cached["prefix"], flow_k1["prefix"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k1_cached["suffix"], flow_k1["suffix"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k1_cached["velocity"], flow_k1["velocity"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(flow_k1_cached["flow_loss"], flow_k1["flow_loss"], atol=1e-6, rtol=1e-6)
    gpu_memory_highwater["after_k1_fused_and_cached_forward"] = _gpu_memory_snapshot()

    def integrated_flow_loss(tree, ki: bool, use_cache: bool):
        return joint_flow_forward(
            tree, prefix, flow_actions, flow_noise_k5[0], flow_time_k5[0], mask1, pos1,
            actual_action_dim=6, draws=1, num_heads=HEADS, num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM, chunk_size=CHUNK, knowledge_insulation=ki,
            predict_actions=jnp.asarray([True]), reduction="mean", use_prefix_cache=use_cache,
        )["flow_loss"]

    flow_grad_on = jax.jit(jax.grad(lambda tree: integrated_flow_loss(tree, True, False)))(flow_params)
    flow_grad_off = jax.jit(jax.grad(lambda tree: integrated_flow_loss(tree, False, False)))(flow_params)
    flow_grad_on_cached = jax.jit(jax.grad(lambda tree: integrated_flow_loss(tree, True, True)))(flow_params)
    flow_grad_off_cached = jax.jit(jax.grad(lambda tree: integrated_flow_loss(tree, False, True)))(flow_params)
    jax.block_until_ready(flow_grad_off_cached["layers"][0]["expert"]["mlp"]["down_proj"])
    flow_on_layers = {"prefix": flow_grad_on["layers"][0]["prefix"],
                      "expert": flow_grad_on["layers"][0]["expert"]}
    flow_off_layers = {"prefix": flow_grad_off["layers"][0]["prefix"],
                       "expert": flow_grad_off["layers"][0]["expert"]}
    flow_on_report = _compare_gradients(
        "flow_ki_on", flow_on_layers, arrays,
        {"gradient_states": manifest["flow_gradient_states"]},
    )
    flow_off_report = _compare_gradients(
        "flow_ki_off", flow_off_layers, arrays,
        {"gradient_states": manifest["flow_gradient_states"]},
    )
    flow_on_vlm = _gradient_summary(flow_on_layers["prefix"])
    flow_off_vlm = _gradient_summary(flow_off_layers["prefix"])
    flow_on_expert = _gradient_summary(flow_on_layers["expert"])
    flow_cache_on_error = _max_tree_abs_diff(flow_grad_on, flow_grad_on_cached)
    flow_cache_off_error = _max_tree_abs_diff(flow_grad_off, flow_grad_off_cached)
    assert flow_on_vlm["max_abs"] <= 1e-10, f"integrated KI-on Flow→VLM gradient leaked: {flow_on_vlm}"
    assert flow_off_vlm["max_abs"] > 1e-7, f"integrated KI-off positive control is ineffective: {flow_off_vlm}"
    assert flow_on_expert["max_abs"] > 1e-7 and np.isfinite(flow_on_expert["max_abs"])
    assert flow_cache_on_error <= 1e-6 and flow_cache_off_error <= 1e-6
    gpu_memory_highwater["after_ki_on_off_fused_cached_backward"] = _gpu_memory_snapshot()

    independent_velocity = []
    for draw in range(5):
        one = joint_flow_forward(
            flow_params, prefix, flow_actions, flow_noise_k5[draw], flow_time_k5[draw], mask1, pos1,
            actual_action_dim=6, draws=1, num_heads=HEADS, num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM, chunk_size=CHUNK, knowledge_insulation=True,
            predict_actions=jnp.asarray([True]), reduction="mean", use_prefix_cache=False,
        )
        independent_velocity.append(one["velocity"])
    independent_velocity = jnp.stack(independent_velocity, axis=1)
    np.testing.assert_allclose(independent_velocity, flow_k5["velocity"], atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(independent_velocity, arrays["forward__velocity_expected_independent"],
                               atol=1e-5, rtol=1e-4)

    # Compare reference and JAX gradients, preserving the distinction between
    # autograd None, a numeric zero, and a nonzero gradient.
    def loss_fn(tree, ki):
        _, predicted = joint_layer(
            tree, prefix, suffix1, mask1, pos1, cond1,
            num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM,
            knowledge_insulation=ki,
        )
        return jnp.mean(jnp.square(predicted - target))

    grad_on = jax.jit(jax.grad(lambda tree: loss_fn(tree, True)))(params)
    grad_off = jax.jit(jax.grad(lambda tree: loss_fn(tree, False)))(params)
    jax.block_until_ready(grad_off["expert"]["mlp"]["down_proj"])
    on_report = _compare_gradients("ki_on", grad_on, arrays, manifest)
    off_report = _compare_gradients("ki_off", grad_off, arrays, manifest)
    on_vlm = _gradient_summary(grad_on["prefix"])
    off_vlm = _gradient_summary(grad_off["prefix"])
    on_expert = _gradient_summary(grad_on["expert"])
    assert on_vlm["max_abs"] <= 1e-10, f"KI-on Flow→VLM gradient leaked: {on_vlm}"
    assert off_vlm["max_abs"] > 1e-7, f"KI-off positive control is ineffective: {off_vlm}"
    assert on_expert["max_abs"] > 1e-7, f"expert Flow gradient is not active: {on_expert}"
    assert np.isfinite(on_expert["max_abs"]) and np.isfinite(off_vlm["max_abs"])
    diagnostics = {
        "evidence_class": "debug_only_not_full_architecture_gate_evidence",
        "backend": jax.default_backend(),
        "fixture_sha256": _npz_hash(evidence / "oracle/run1/g3_layer_fixture.npz"),
        "reference_source_sha256": manifest["reference_source_sha256"],
        "ada_rms_gate_max_abs": float(jnp.max(jnp.abs(ada_gate))),
        "ki_on": {"reference_vs_jax": on_report, "jax_prefix": on_vlm, "jax_expert": on_expert},
        "ki_off": {"reference_vs_jax": off_report, "jax_prefix": off_vlm,
                   "reference_positive_control_max_abs": max(
                       float(np.max(np.abs(arrays["grad__" + name]), initial=0.0))
                       for name in manifest["gradient_states"]
                       if name.startswith("ki_off__prefix__")
                   )},
        "flow": {"mean": float(loss_mean), "none": np.asarray(loss_none).tolist(),
                 "actual_action_dim": 6, "padding_dim": 26,
                 "routed_dimensions_checked": [6, 12, 14, 32]},
        "pi05_fallback_dispatch": manifest["pi05_fallback_dispatch"],
        "integrated_flow": {
            "k1_loss": float(flow_k1["flow_loss"]),
            "k5_loss": float(flow_k5["flow_loss"]),
            "ki_on_k5_cached_velocity_max_abs_error": float(np.max(np.abs(
                np.asarray(flow_k5_cached["velocity"]) - np.asarray(flow_k5["velocity"])
            ))),
            "k5_independent_velocity_max_abs_error": float(np.max(np.abs(
                np.asarray(independent_velocity) - arrays["forward__velocity_expected_independent"]
            ))),
        },
        "integrated_flow_gradients": {
            "ki_on": {"reference_vs_jax": flow_on_report, "jax_prefix": flow_on_vlm,
                      "jax_expert": flow_on_expert},
            "ki_off": {"reference_vs_jax": flow_off_report, "jax_prefix": flow_off_vlm},
            "cache_vs_fused_gradient_max_abs": {
                "ki_on": flow_cache_on_error,
                "ki_off": flow_cache_off_error,
            },
        },
        "memory": {
            "scope": "compact debug process; allocator peaks are cumulative high-water marks",
            "gpu_allocator_highwater_by_phase": gpu_memory_highwater,
            "host_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "host_peak_rss_unit": "bytes; Linux ru_maxrss KiB converted by 1024",
        },
        "k5": {"draws": 5, "prefix_computations": 1, "max_independent_error": float(
            np.max(np.abs(np.asarray(independent) - arrays["expected__k5_independent_suffix"])))},
    }
    (evidence / "g3_debug_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
