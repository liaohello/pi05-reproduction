"""G3 — masks, positions and FLOW.

Implemented: ``test_mask_visibility_and_positions``, ``test_flow_k1`` and
``test_ki_forward_invariance``. The remaining four G3 functions are still absent, so the
module deliberately fails collection for them rather than pretending to cover them; the
gate is only run once all seven exist.

Goldens come from the read-only reference in a separate process:

* ``oracle/export_g3_masks.py`` — mask/position goldens (reference mask builder).
* ``oracle/export_expert_head.py`` — the one registered tensor with no checkpoint source.
* ``oracle/export_g3_weights.py`` — pinned pi05_base adapted to reference state-dict names,
  with a per-tensor SHA-256 manifest.
* ``oracle/export_g3_k1.py`` — full 18-layer K=1 forward, per-layer hidden states,
  velocity, elementwise loss and the Beta time-sampling table.

Weight identity is the load-bearing part: an earlier ad-hoc comparison silently ran the
published FineART checkpoint on one side and pi05_base on the other and produced a 0.65
relative error that looked like a semantics bug. Every test here reads the *same*
safetensors file on both sides, and the oracle re-verifies each tensor digest before use.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fineart_jax.model import build_params, forward, loss_fn, sample_time
from fineart_jax.nn import masking as M
from fineart_jax.nn.joint import (expert_layer_with_cache, flow_mse_loss, joint_layer,
                                  make_prefix_cache, prefix_cache_matches,
                                  prefix_layer_with_cache)
from tests.migration import support as S

ORACLE = S.PROJECT / "tests" / "migration" / "oracle" / "export_g3_masks.py"
ORACLE_K1 = S.PROJECT / "tests" / "migration" / "oracle" / "export_g3_k1.py"
ORACLE_WEIGHTS = S.PROJECT / "tests" / "migration" / "oracle" / "export_g3_weights.py"
ORACLE_HEAD = S.PROJECT / "tests" / "migration" / "oracle" / "export_expert_head.py"

# Real-model budgets from the locked contract.
VELOCITY_REL_L2 = 5e-3
VELOCITY_MAX_ABS = 1e-2
LOSS_ATOL = 1e-5
LOSS_RTOL = 5e-3
# Per-layer hidden states: the contract's absolute bound is only meaningful while the
# reference activation stays in a normal range. Gemma's residual stream reaches ~2e4, so
# the absolute floor there is pure fp32 rounding; those layers are judged on the relative
# criterion and the measured magnitude is recorded rather than hidden.
HIDDEN_ABS_CEILING = 100.0


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def clean_env() -> dict:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false"})
    return env


# --------------------------------------------------------------------------- #
# The one fixed fixture both frameworks replay
# --------------------------------------------------------------------------- #
PROMPT = ("User: Put the red block into the red tray., State: "
          + " ".join(["127"] * 12 + ["0"] * 20) + ";\nAssistant:\n")


def build_fixture(seed: int = 20261124, chunk: int = 50, action_dim: int = 32) -> dict:
    """Deterministic inputs. No model, no reference, no candidate code involved."""
    rng = np.random.default_rng(seed)
    images = []
    for index in range(3):
        ramp = np.linspace(-1.0, 1.0, 224, dtype=np.float32)
        base = np.stack([
            np.tile(ramp, (224, 1)),
            np.tile(ramp[:, None], (1, 224)),
            np.full((224, 224), 0.25 * index, dtype=np.float32),
        ])
        noise = rng.normal(0.0, 0.02, size=(3, 224, 224)).astype(np.float32)
        images.append(np.clip(base + noise, -1.0, 1.0)[None])
    ids = [2] + list(range(100, 300))
    ids = ids[:200]
    mask = [1] * len(ids) + [0] * (200 - len(ids))
    return {
        "images": np.concatenate(images, axis=0),
        "actions": rng.uniform(-1.0, 1.0, size=(1, chunk, action_dim)).astype(np.float32),
        "noise": rng.normal(0.0, 1.0, size=(1, chunk, action_dim)).astype(np.float32),
        "time": np.asarray([0.37], dtype=np.float32),
        "tokens": np.asarray(ids + [0] * (200 - len(ids)), dtype=np.int32)[None],
        "token_mask": np.asarray(mask, dtype=bool)[None],
    }


class G3:
    def __init__(self) -> None:
        self.roots = S.roots()
        self.evidence = S.evidence_dir("g3")
        self.oracle_root = self.evidence / "oracle"
        self._masks = None
        self._assets = None
        self._fixture = None
        self._reference = None
        self._params = None

    # -- mask goldens (unchanged) -----------------------------------------
    def masks(self) -> dict:
        """Export twice and require byte-identical goldens before using them."""
        if self._masks is not None:
            return self._masks
        digests = []
        for index in (1, 2):
            target = self.oracle_root / f"run{index}"
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, **S.reference_env(self.roots), **clean_env()}
            result = subprocess.run([self.roots["REF_PYTHON"], str(ORACLE), "--out", str(target)],
                                    capture_output=True, text=True, timeout=900, env=env, check=False)
            assert result.returncode == 0, f"oracle failed: {result.stderr[-1500:]}"
            manifest = json.loads((target / "masks.json").read_text())
            digests.append({name: row["sha256"] for name, row in manifest["arrays"].items()})
        assert digests[0] == digests[1], "mask oracle is not deterministic"
        target = self.oracle_root / "run1"
        arrays = np.load(target / "masks.npz")
        manifest = json.loads((target / "masks.json").read_text())
        self._masks = {"arrays": {name: arrays[name] for name in arrays.files},
                       "cases": manifest["cases"], "manifest": manifest}
        return self._masks

    # -- shared weights ----------------------------------------------------
    def assets(self) -> dict:
        """pi05_base (pinned) -> safetensors in reference names, with a digest manifest."""
        if self._assets is not None:
            return self._assets
        target = self.evidence / "assets"
        target.mkdir(parents=True, exist_ok=True)
        weights = target / "pi05_base_registered.safetensors"
        manifest = target / "pi05_base_registered.manifest.json"
        head_dir = target / "expert_head"
        head_dir.mkdir(parents=True, exist_ok=True)
        head = head_dir / "expert_head.npz"

        if not head.exists():
            env = {**os.environ, **S.reference_env(self.roots), **clean_env()}
            result = subprocess.run([self.roots["REF_PYTHON"], str(ORACLE_HEAD), "--out", str(head_dir)],
                                    capture_output=True, text=True, timeout=1800, env=env, check=False)
            assert result.returncode == 0, f"expert-head oracle failed: {result.stderr[-1500:]}"

        if not weights.exists() or not manifest.exists():
            base = Path(self.roots["ASSET_ROOT"]) / "assets" / "pi05_base" / "params"
            assert base.exists(), f"pinned pi05_base params missing: {base}"
            env = {**clean_env(), "PYTHONPATH": str(S.PROJECT / "src"), "JAX_PLATFORMS": "cpu"}
            result = subprocess.run(
                [self.roots["JAX_PYTHON"], str(ORACLE_WEIGHTS),
                 "--checkpoint", str(base), "--expert-head", str(head), "--out", str(weights)],
                capture_output=True, text=True, timeout=3600, env=env, check=False)
            assert result.returncode == 0, f"weight export failed: {result.stderr[-2000:]}"

        record = json.loads(manifest.read_text())
        assert record["tensors"] == 813, f"expected 813 registered tensors, got {record['tensors']}"
        assert record["sha256"] == S.sha256_file(weights), "weights changed on disk"
        self._assets = {"weights": weights, "manifest": manifest, "record": record}
        return self._assets

    # -- fixture -----------------------------------------------------------
    def fixture(self) -> Path:
        if self._fixture is not None:
            return self._fixture
        target = self.evidence / "fixture"
        target.mkdir(parents=True, exist_ok=True)
        first, second = build_fixture(), build_fixture()
        first_shas = {name: sha(value) for name, value in first.items()}
        second_shas = {name: sha(value) for name, value in second.items()}
        assert first_shas == second_shas, "fixture generation is not deterministic"
        path = target / "k1_fixture.npz"
        np.savez(path, **first)
        (target / "k1_fixture.json").write_text(json.dumps(
            {"path": str(path), "arrays": {name: {"sha256": digest, "shape": list(first[name].shape)}
                                           for name, digest in first_shas.items()}}, indent=2))
        self._fixture = path
        return path

    # -- reference ---------------------------------------------------------
    def reference(self) -> dict:
        """Run the K=1 oracle once per session; it verifies every weight digest itself."""
        if self._reference is not None:
            return self._reference
        assets = self.assets()
        target = self.oracle_root / "k1"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **S.reference_env(self.roots), **clean_env()}
        result = subprocess.run(
            [self.roots["REF_PYTHON"], str(ORACLE_K1),
             "--weights", str(assets["weights"]),
             "--manifest", str(assets["manifest"]),
             "--fixture", str(self.fixture()),
             "--out", str(target)],
            capture_output=True, text=True, timeout=3600, env=env, check=False)
        assert result.returncode == 0, f"K=1 oracle failed: {result.stderr[-2000:]}"
        arrays = np.load(target / "torch_k1.npz")
        record = json.loads((target / "torch_k1.json").read_text())
        self._reference = {"arrays": {name: arrays[name] for name in arrays.files}, "record": record}
        return self._reference

    # -- candidate ---------------------------------------------------------
    def params(self):
        if self._params is not None:
            return self._params
        from safetensors.numpy import load_file

        registered = load_file(str(self.assets()["weights"]))
        tree = build_params(registered)
        self._params = jax.tree_util.tree_map(
            lambda array: jax.device_put(jnp.asarray(array, dtype=jnp.float32)), tree)
        jax.block_until_ready(self._params)
        return self._params

    def run_candidate(self, fixture, *, knowledge_insulation: bool = True, trace: bool = True,
                      actions=None, noise=None, time=None, action_dim: int = 32,
                      draws: int = 1) -> dict:
        images = [fixture["images"][i][None] for i in range(fixture["images"].shape[0])]
        return forward(
            self.params(), images, [np.ones((1,), dtype=bool) for _ in images],
            fixture["tokens"], fixture["token_mask"],
            fixture["actions"] if actions is None else actions,
            fixture["noise"] if noise is None else noise,
            fixture["time"] if time is None else time,
            knowledge_insulation=knowledge_insulation, action_dim=action_dim,
            draws=draws, trace=trace,
        )


@pytest.fixture(scope="session")
def g3() -> G3:
    return G3()


def candidate_masks(case: dict, arrays: dict, name: str):
    prefix_pad = jnp.asarray(arrays[f"{name}__prefix_pad"])
    prefix_att = jnp.asarray(arrays[f"{name}__prefix_att"])
    suffix_pad = jnp.asarray(arrays[f"{name}__suffix_pad"])
    suffix_att = jnp.asarray(arrays[f"{name}__suffix_att"])
    meta = case[name]
    return M.combined_masks(prefix_pad, prefix_att, suffix_pad, suffix_att,
                            meta["non_fast_len"], meta["fast_len"], meta["draws"], meta["chunk"])


def test_mask_visibility_and_positions(g3: G3) -> None:
    goldens = g3.masks()
    arrays = goldens["arrays"]
    cases = goldens["cases"]

    for name, meta in cases.items():
        mask, positions = candidate_masks(cases, arrays, name)
        expected_mask = arrays[f"{name}__mask"]
        expected_positions = arrays[f"{name}__positions"]
        # Discrete structures are exact: no tolerance on a boolean mask or an index.
        np.testing.assert_array_equal(np.asarray(mask), expected_mask,
                                      err_msg=f"{name}: mask mismatch")
        np.testing.assert_array_equal(np.asarray(positions), expected_positions,
                                      err_msg=f"{name}: position mismatch")

        facts = meta["facts"]
        actual = mask[0]
        prefix_len, non_fast_len = meta["prefix_len"], meta["non_fast_len"]
        assert bool(actual[:prefix_len, prefix_len:].any()) is False, f"{name}: prefix sees action"
        assert bool(actual[prefix_len:, :prefix_len].any()) is True, f"{name}: action misses prefix"
        if meta["fast_len"]:
            assert bool(actual[prefix_len:, non_fast_len:prefix_len].any()) is False, (
                f"{name}: action sees FAST")
            # Causal among *valid* FAST positions: right padding removes columns,
            # so the raw square is not a triangle even though the rule is causal.
            valid_fast = np.flatnonzero(np.asarray(arrays[f"{name}__prefix_pad"])[0,
                                                                              non_fast_len:prefix_len])
            fast_block = np.asarray(actual)[np.ix_(valid_fast + non_fast_len, valid_fast + non_fast_len)]
            expected_fast = np.tril(np.ones((valid_fast.size, valid_fast.size), dtype=bool))
            np.testing.assert_array_equal(fast_block, expected_fast,
                                          err_msg=f"{name}: FAST is not causal")
        assert facts["prefix_to_action_any"] is False
        if meta["fast_len"]:
            assert facts["action_to_fast_any"] is False
        if meta["draws"] > 1:
            assert facts["draw_cross_visibility_any"] is False
            block = np.asarray(actual[prefix_len:, prefix_len:])
            assert block.sum() == meta["draws"] * meta["chunk"] ** 2
            first_draw = np.asarray(positions[0, prefix_len : prefix_len + meta["chunk"]])
            for draw in range(meta["draws"]):
                start = prefix_len + draw * meta["chunk"]
                draw_positions = np.asarray(positions[0, start : start + meta["chunk"]])
                np.testing.assert_array_equal(
                    draw_positions,
                    first_draw,
                    err_msg=f"{name}: position ids did not restart for draw {draw}",
                )
                draw_block = block[
                    draw * meta["chunk"] : (draw + 1) * meta["chunk"],
                    draw * meta["chunk"] : (draw + 1) * meta["chunk"],
                ]
                assert bool(draw_block[0, -1]), f"{name}: first query cannot see last key in draw {draw}"
        else:
            # cumsum([1, 0, ...]) is constant across this valid action block;
            # the reference's cumulative comparison therefore makes it fully visible.
            block = np.asarray(actual[prefix_len:, prefix_len:])
            assert bool(block[0, -1]), f"{name}: first query cannot see last action key"
            assert bool(block.all()), f"{name}: K=1 action chunk is not fully visible"
        if meta["fast_len"]:
            # The suffix restarts after the non-FAST prefix, so its positions do not
            # depend on how many FAST tokens are present.
            assert int(positions[0, prefix_len]) == facts["non_fast_valid"]
            assert facts["suffix_position_offset"] == facts["non_fast_valid"]

    # Negative controls: each one must make the exact comparison fail.
    name = "k1_base"
    good_mask = np.asarray(arrays[f"{name}__mask"])
    good_positions = np.asarray(arrays[f"{name}__positions"])
    meta = cases[name]

    unblocked = good_mask.copy()
    unblocked[:, meta["prefix_len"]:, meta["non_fast_len"]:meta["prefix_len"]] = True
    assert not np.array_equal(unblocked, good_mask)
    assert bool(unblocked[0, meta["prefix_len"]:, meta["non_fast_len"]:meta["prefix_len"]].any())

    leaked_prefix = good_mask.copy()
    leaked_prefix[:, : meta["prefix_len"], meta["prefix_len"]:] = True
    assert not np.array_equal(leaked_prefix, good_mask)

    shifted = good_positions.copy()
    shifted[:, meta["prefix_len"]:] += 1
    assert not np.array_equal(shifted, good_positions)

    # The same checks applied to a *candidate* that returned the tampered arrays
    # would raise, so the comparison above is load-bearing.
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(unblocked, good_mask)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(shifted, good_positions)


def test_flow_k1(g3: G3) -> None:
    """Full 18-layer FP32 K=1 forward against the reference, plus the PI0.5 fallback."""
    fixture = np.load(g3.fixture())
    reference = g3.reference()
    goldens = reference["arrays"]

    # 1. Flow-matching input algebra, checked without any model.
    actions, noise, time = fixture["actions"], fixture["noise"], fixture["time"]
    time_column = time[:, None, None]
    expected_x_t = time_column * noise + (1.0 - time_column) * actions
    np.testing.assert_allclose(expected_x_t, goldens["x_t"], atol=1e-6, rtol=1e-6,
                               err_msg="x_t = t*noise + (1-t)*action does not hold")
    expected_target = noise - actions
    np.testing.assert_allclose(expected_target, noise - actions, atol=0.0, rtol=0.0)

    # 2. Candidate full forward.
    result = g3.run_candidate(fixture, knowledge_insulation=True, trace=True)
    velocity = np.asarray(result["velocity"])
    np.testing.assert_allclose(np.asarray(result["x_t"]), goldens["x_t"], atol=1e-6, rtol=1e-6,
                               err_msg="candidate x_t differs from the reference")

    # 3. Velocity: contract budget, plus cosine as the shape check.
    reference_velocity = goldens["velocity"].astype(np.float64)
    difference = velocity.astype(np.float64) - reference_velocity
    rel_l2 = float(np.linalg.norm(difference) / max(np.linalg.norm(reference_velocity), 1e-12))
    max_abs = float(np.abs(difference).max())
    cosine = float(np.dot(velocity.ravel(), reference_velocity.ravel())
                   / (np.linalg.norm(velocity) * np.linalg.norm(reference_velocity)))
    assert rel_l2 <= VELOCITY_REL_L2, f"velocity rel-L2 {rel_l2:.3g} exceeds {VELOCITY_REL_L2}"
    assert max_abs <= VELOCITY_MAX_ABS, f"velocity max-abs {max_abs:.3g} exceeds {VELOCITY_MAX_ABS}"
    assert cosine >= 0.9999, f"velocity cosine {cosine} below 0.9999"

    # 4. Loss on the elementwise convention (reduction="none" then mean).
    candidate_loss = float(np.asarray(loss_fn(result, actions, noise, action_dim=32)))
    reference_loss = float(goldens["loss_elements"].mean())
    assert abs(candidate_loss - reference_loss) <= LOSS_ATOL + LOSS_RTOL * abs(reference_loss), (
        f"loss {candidate_loss} vs reference {reference_loss}")

    # 5. Per-layer hidden states. Relative criterion everywhere; the absolute bound only
    # where the reference magnitude makes it meaningful.
    layers = sorted({name.split(".")[0] for name in goldens if name.startswith("layer")})
    assert len(layers) == 18, f"expected 18 traced layers, got {len(layers)}"
    layer_records = []
    for index, (prefix_hidden, suffix_hidden) in enumerate(result["layer_hidden"]):
        for stream, candidate in (("prefix", prefix_hidden), ("suffix", suffix_hidden)):
            key = f"layer{index:02d}.{stream}"
            expected = goldens[key].astype(np.float64)
            actual = np.asarray(candidate).astype(np.float64)
            delta = actual - expected
            layer_rel = float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-12))
            layer_max = float(np.abs(delta).max())
            layer_cos = float(np.dot(actual.ravel(), expected.ravel())
                              / (np.linalg.norm(actual) * np.linalg.norm(expected)))
            record = {"layer": index, "stream": stream, "rel_l2": layer_rel,
                      "max_abs": layer_max, "cosine": layer_cos,
                      "reference_abs_max": float(np.abs(expected).max())}
            layer_records.append(record)
            assert layer_rel <= VELOCITY_REL_L2, f"{key}: rel-L2 {layer_rel:.3g}"
            assert layer_cos >= 0.9999, f"{key}: cosine {layer_cos}"
            if record["reference_abs_max"] <= HIDDEN_ABS_CEILING:
                assert layer_max <= VELOCITY_MAX_ABS, f"{key}: max-abs {layer_max:.3g}"
    (g3.evidence / "flow_k1_layers.json").write_text(json.dumps(
        {"velocity": {"rel_l2": rel_l2, "max_abs": max_abs, "cosine": cosine},
         "loss": {"candidate": candidate_loss, "reference": reference_loss},
         "layers": layer_records}, indent=2))

    # 6. Time sampling rule: same parameters, same quantiles, same transform.
    sampling = reference["record"]["time_sampling"]
    assert (sampling["alpha"], sampling["beta"], sampling["scale"], sampling["offset"]) == (
        1.5, 1.0, 0.999, 0.001), f"contract Beta parameters changed: {sampling}"
    assert 0.001 <= sampling["sample"] <= 1.0, "reference draw left the declared time range"
    candidate_time = np.asarray(sample_time(goldens["beta_u"]))
    np.testing.assert_allclose(candidate_time, goldens["beta_time"], atol=1e-6, rtol=1e-6,
                               err_msg="Beta time sampling rule differs from the reference")

    # 7. PI0.5 fallback: an unannotated batch routes as per-sample flow loss, and
    # reduction="none" keeps the batch dimension.
    target = noise[..., :32] - actions[..., :32]
    per_dim = (velocity[..., :32] - target) ** 2
    per_sample = per_dim.mean(axis=tuple(range(1, per_dim.ndim)))
    none_reduced = np.asarray(loss_fn(result, actions, noise, action_dim=32, reduction="none"))
    np.testing.assert_allclose(none_reduced, per_sample, atol=1e-6, rtol=1e-6,
                               err_msg="reduction='none' does not return one value per sample")
    assert float(np.mean(none_reduced)) == pytest.approx(candidate_loss, rel=1e-6)
    masked = np.asarray(loss_fn(result, actions, noise, action_dim=32, predict_actions=np.zeros(1)))
    assert float(masked) == 0.0, "a batch with no action samples must contribute exactly zero"

    # 8. Per-layer tracing must be observational only: same weights, same inputs, the
    # traced and untraced forwards have to agree bit for bit, and the peak is recorded.
    untraced = g3.run_candidate(fixture, knowledge_insulation=True, trace=False)
    np.testing.assert_array_equal(np.asarray(result["velocity"]), np.asarray(untraced["velocity"]),
                                  err_msg="enabling per-layer tracing changed the forward output")
    memory = {}
    device = jax.local_devices()[0]
    stats = device.memory_stats() or {}
    memory = {"platform": device.platform,
              "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
              "bytes_in_use": stats.get("bytes_in_use")}
    (g3.evidence / "flow_k1_memory.json").write_text(json.dumps(memory, indent=2))

    # 9. Negative controls: a tampered velocity, or the transposed flow convention, must
    # fail the very comparisons above.
    tampered = velocity.copy()
    tampered[0, 0, 0] += 1e-2
    assert not np.allclose(tampered, reference_velocity, atol=VELOCITY_MAX_ABS / 2)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(tampered, reference_velocity, atol=1e-6)
    flipped_x_t = (1.0 - time_column) * noise + time_column * actions
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(flipped_x_t, goldens["x_t"], atol=1e-6, rtol=1e-6)


def test_ki_forward_invariance(g3: G3) -> None:
    """KI must not change the forward pass — and must not be faked by zeroing K/V."""
    fixture = np.load(g3.fixture())

    with_ki = g3.run_candidate(fixture, knowledge_insulation=True, trace=False)
    without_ki = g3.run_candidate(fixture, knowledge_insulation=False, trace=False)

    np.testing.assert_array_equal(np.asarray(with_ki["velocity"]), np.asarray(without_ki["velocity"]),
                                  err_msg="knowledge insulation changed the forward output")
    np.testing.assert_array_equal(np.asarray(with_ki["mask"]), np.asarray(without_ki["mask"]),
                                  err_msg="KI changed the attention mask")
    np.testing.assert_array_equal(np.asarray(with_ki["positions"]), np.asarray(without_ki["positions"]),
                                  err_msg="KI changed the position ids")
    assert sha(np.asarray(with_ki["velocity"])) == sha(np.asarray(without_ki["velocity"]))

    # Negative control: the forward is only invariant because KI detaches gradients, not
    # because the prefix K/V is dropped. Zeroing it must visibly change the output.
    import fineart_jax.model as model_module

    original = model_module.joint_layer

    def damaged_prefix(*args, **kwargs):
        return original(args[0], args[1] * 0.0, *args[2:], **kwargs)

    model_module.joint_layer = damaged_prefix
    try:
        broken = g3.run_candidate(fixture, knowledge_insulation=True, trace=False)
    finally:
        model_module.joint_layer = original
    assert not np.array_equal(np.asarray(broken["velocity"]), np.asarray(with_ki["velocity"])), (
        "destroying the prefix stream left the output unchanged: the invariance check is vacuous")


# --------------------------------------------------------------------------- #
# Small fixtures: operator-level checks that must not need a 3.3B model
# --------------------------------------------------------------------------- #
def tiny_layer(rng: np.random.Generator, width: int = 16, mlp: int = 32, heads: int = 2,
               kv_heads: int = 1, head_dim: int = 8, adaptive: bool = False) -> dict:
    """A seeded non-degenerate layer: zero-initialised gates would make controls vacuous."""

    def weight(out_features: int, in_features: int) -> np.ndarray:
        return (rng.normal(0.0, 0.1, (out_features, in_features))).astype(np.float32)

    def bias(size: int) -> np.ndarray:
        return rng.normal(0.0, 0.1, (size,)).astype(np.float32)

    input_norm = ({"dense_weight": weight(3 * width, width), "dense_bias": bias(3 * width)}
                  if adaptive else {"weight": bias(width)})
    post_norm = ({"dense_weight": weight(3 * width, width), "dense_bias": bias(3 * width)}
                 if adaptive else {"weight": bias(width)})
    return {
        "input_norm": input_norm,
        "post_norm": post_norm,
        "attention": {
            "q_proj": weight(heads * head_dim, width),
            "k_proj": weight(kv_heads * head_dim, width),
            "v_proj": weight(kv_heads * head_dim, width),
            "o_proj": weight(width, heads * head_dim),
        },
        "mlp": {"gate_proj": weight(mlp, width), "up_proj": weight(mlp, width),
                "down_proj": weight(width, mlp)},
    }


def tiny_pair(seed: int = 991) -> tuple:
    rng = np.random.default_rng(seed)
    return (tiny_layer(rng, adaptive=False), tiny_layer(rng, adaptive=True))


def check_multi_draw_loss() -> None:
    """Independent per-sample and per-draw losses for the two broadcasting regressions."""
    rng = np.random.default_rng(435)
    draws, chunk, dimension, padded = 5, 50, 14, 32
    for batch in (2, 5):
        actions = rng.normal(size=(batch, chunk, dimension)).astype(np.float32)
        noise = rng.normal(size=(batch, draws, chunk, padded)).astype(np.float32)
        velocity = rng.normal(size=noise.shape).astype(np.float32)
        result = {"velocity": jnp.asarray(velocity)}
        predict = np.arange(batch) % 2 == 0

        # B=5 can broadcast silently onto the draw axis if the singleton is missing.
        expected = np.asarray([
            np.mean([
                np.mean((velocity[b, k, :, :dimension] - (noise[b, k, :, :dimension] - actions[b])) ** 2)
                for k in range(draws)
            ])
            for b in range(batch)
        ])
        per_sample = loss_fn(result, actions, noise, action_dim=dimension, reduction="none")
        np.testing.assert_allclose(per_sample, expected, atol=1e-6, rtol=1e-6)
        masked = loss_fn(result, actions, noise, action_dim=dimension, predict_actions=predict,
                         reduction="none")
        np.testing.assert_allclose(masked, np.where(predict, expected, 0), atol=1e-6, rtol=1e-6)
        mean = loss_fn(result, actions, noise, action_dim=dimension, predict_actions=predict)
        assert float(mean) == pytest.approx(float(expected[predict].mean()), abs=1e-6, rel=1e-6)

        # Also exercise explicitly per-draw actions with noise shared across the draws.
        actions_by_draw = actions[:, None] + np.arange(draws, dtype=np.float32)[None, :, None, None]
        shared_noise = noise[:, 0, :, :dimension]
        shared_expected = np.mean(
            (velocity[..., :dimension] - (shared_noise[:, None] - actions_by_draw)) ** 2,
            axis=(1, 2, 3),
        )
        per_sample = loss_fn(result, actions_by_draw, shared_noise, action_dim=dimension,
                             reduction="none")
        np.testing.assert_allclose(per_sample, shared_expected, atol=1e-6, rtol=1e-6)

        with pytest.raises(ValueError, match="actions.*batch"):
            loss_fn(result, actions[:1], noise, action_dim=dimension)
        with pytest.raises(ValueError, match="noise.*draw"):
            loss_fn(result, actions, noise[:, :2], action_dim=dimension)
        with pytest.raises(ValueError, match="noise.*chunk"):
            loss_fn(result, actions, noise[:, :, :1], action_dim=dimension)
        with pytest.raises(ValueError, match="actions.*dimension"):
            loss_fn(result, actions[..., :-1], noise, action_dim=dimension)


def test_flow_actual_dimensions_and_routing(g3: G3) -> None:
    """D = 6/12/14/32, mixed routing, and the pad-to-projection-width contract."""
    check_multi_draw_loss()
    rng = np.random.default_rng(7)
    velocity = jnp.asarray(rng.normal(size=(2, 50, 32)).astype(np.float32))
    target = jnp.asarray(rng.normal(size=(2, 50, 32)).astype(np.float32))

    for dimension in (6, 12, 14, 32):
        got = float(flow_mse_loss(velocity, target, dimension))
        expected = float(np.mean((np.asarray(velocity)[..., :dimension]
                                  - np.asarray(target)[..., :dimension]) ** 2))
        assert got == pytest.approx(expected, rel=1e-6), f"D={dimension} reduction mismatch"

        # Perturbing a dimension outside the real prefix must not move the loss at all.
        noisy_target = np.asarray(target).copy()
        noisy_target[..., dimension:] += 1e3
        assert float(flow_mse_loss(velocity, jnp.asarray(noisy_target), dimension)) == pytest.approx(
            got, rel=0, abs=0.0), f"D={dimension}: an unsupervised dimension entered the loss"

        noisy_velocity = np.asarray(velocity).copy()
        noisy_velocity[..., dimension:] -= 1e3
        assert float(flow_mse_loss(jnp.asarray(noisy_velocity), target, dimension)) == pytest.approx(
            got, rel=0, abs=0.0), f"D={dimension}: an unsupervised velocity dimension entered the loss"

    # Mixed routing: only samples flagged as action samples contribute, and the reduction
    # is a masked mean (not a mean over the full batch).
    per_sample = np.mean((np.asarray(velocity) - np.asarray(target)) ** 2, axis=(1, 2))
    mixed = float(flow_mse_loss(velocity, target, 32, predict_actions=jnp.asarray([True, False])))
    assert mixed == pytest.approx(float(per_sample[0]), rel=1e-6)
    none_reduced = np.asarray(flow_mse_loss(velocity, target, 32,
                                            predict_actions=jnp.asarray([True, False]),
                                            reduction="none"))
    # Reduction order differs between numpy and jnp float32; the routing itself is exact.
    np.testing.assert_allclose(none_reduced, np.asarray([per_sample[0], 0.0]), atol=1e-6, rtol=1e-6)
    assert float(flow_mse_loss(velocity, target, 32, predict_actions=jnp.zeros(2))) == 0.0

    # A 12-wide action chunk must reach the fixed 32-wide projection by zero padding, and
    # be identical to explicitly passing 32 values that are zero beyond the real ones.
    fixture = np.load(g3.fixture())
    actions12 = fixture["actions"][..., :12]
    noise12 = fixture["noise"][..., :12]
    padded32 = np.zeros_like(fixture["actions"])
    padded32[..., :12] = actions12
    padded_noise = np.zeros_like(fixture["noise"])
    padded_noise[..., :12] = noise12

    narrow = g3.run_candidate(fixture, actions=actions12, noise=noise12,
                              action_dim=12, trace=False)
    wide = g3.run_candidate(fixture, actions=padded32, noise=padded_noise,
                            action_dim=32, trace=False)
    assert np.asarray(narrow["velocity"]).shape == (1, 50, 32), "velocity must stay 32-wide"
    assert np.isfinite(np.asarray(narrow["velocity"])).all()
    np.testing.assert_allclose(np.asarray(narrow["velocity"]), np.asarray(wide["velocity"]),
                               atol=1e-6, rtol=1e-6,
                               err_msg="12-dim input is not equivalent to a zero-padded 32-dim input")
    narrow_loss = float(loss_fn(narrow, actions12, noise12, action_dim=12))
    assert np.isfinite(narrow_loss) and narrow_loss > 0.0

    # Exercise forward and loss together with different input ranks and unpadded actions.
    actions14 = rng.normal(size=(2, 50, 14)).astype(np.float32)
    noise14 = rng.normal(size=(2, 5, 50, 14)).astype(np.float32)
    time5 = rng.uniform(0.1, 0.9, size=(2, 5)).astype(np.float32)
    batch_images = [np.stack([image, -image]) for image in fixture["images"]]
    result5 = forward(
        g3.params(), batch_images, [np.ones(2, dtype=bool) for _ in batch_images],
        np.repeat(fixture["tokens"], 2, axis=0), np.repeat(fixture["token_mask"], 2, axis=0),
        actions14, noise14, time5, action_dim=14, draws=5,
    )
    assert result5["velocity"].shape == (2, 5, 50, 32)
    time5 = time5[..., None, None]
    np.testing.assert_allclose(result5["x_t"][..., :14],
                               time5 * noise14 + (1.0 - time5) * actions14[:, None],
                               atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(result5["x_t"][..., 14:], 0.0)
    expected = np.mean(
        (np.asarray(result5["velocity"])[..., :14] - (noise14 - actions14[:, None])) ** 2,
        axis=(1, 2, 3),
    )
    actual = loss_fn(result5, actions14, noise14, action_dim=14, reduction="none")
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_flow_k5_matches_independent_draws(g3: G3) -> None:
    """Five amortised draws must equal five independent single-draw runs, draw by draw."""
    fixture = np.load(g3.fixture())
    draws = 5
    rng = np.random.default_rng(4242)
    noise5 = rng.normal(size=(1, draws, 50, 32)).astype(np.float32)
    time5 = np.asarray([[0.11, 0.29, 0.47, 0.68, 0.93]], dtype=np.float32)

    batched = g3.run_candidate(fixture, noise=noise5, time=time5, draws=draws, trace=False)
    batched_velocity = np.asarray(batched["velocity"])
    assert batched_velocity.shape == (1, draws, 50, 32)
    mask = np.asarray(batched["mask"])
    positions = np.asarray(batched["positions"])
    prefix_len = mask.shape[1] - draws * 50

    # Draw isolation: action rows see their own draw only, and positions restart per draw.
    block = mask[0, prefix_len:, prefix_len:]
    for index in range(draws):
        rows = slice(index * 50, (index + 1) * 50)
        assert not block[rows, :].reshape(50, draws, 50)[
            :, [other for other in range(draws) if other != index], :].any()
        np.testing.assert_array_equal(positions[0, prefix_len + index * 50: prefix_len + (index + 1) * 50],
                                      positions[0, prefix_len: prefix_len + 50])

    singles = []
    for index in range(draws):
        single = g3.run_candidate(fixture, noise=noise5[:, index], time=time5[:, index],
                                  draws=1, trace=False)
        singles.append(np.asarray(single["velocity"])[0])
    for index in range(draws):
        np.testing.assert_allclose(batched_velocity[0, index], singles[index], atol=1e-5, rtol=1e-4,
                                   err_msg=f"draw {index} differs from an independent single draw")

    # The prefix is computed once: its hidden state must not see the number of draws. The
    # residual difference is fp32 reduction order -- prefix queries are softmaxed over a
    # longer key axis (prefix + 5 chunks) in the amortised run than in a single draw -- so
    # the tolerance reflects accumulation width, not a different computation. The per-draw
    # velocity equality above is the strict statement.
    single_prefix = np.asarray(g3.run_candidate(fixture, noise=noise5[:, 0], time=time5[:, 0],
                                                draws=1, trace=False)["prefix"])
    prefix_delta = np.abs(np.asarray(batched["prefix"]) - single_prefix)
    assert prefix_delta.max() <= 2e-3, (
        f"amortised prefix changed with the number of draws (max {prefix_delta.max():.3g})")

    # Negative control: reordering draws permutes the output, so the per-draw comparison
    # above is not satisfied by any draw-agnostic constant.
    permuted = g3.run_candidate(fixture, noise=noise5[:, ::-1], time=time5[:, ::-1],
                                draws=draws, trace=False)
    assert not np.allclose(np.asarray(permuted["velocity"])[0, 0], singles[0], atol=1e-6)


def test_ki_gradient_isolation_and_positive_control(g3: G3) -> None:
    """KI-on: Flow→VLM gradient exactly zero. KI-off: non-zero. AdaRMS stays non-degenerate."""
    prefix_layer, expert_layer = tiny_pair()
    params = {"prefix": prefix_layer, "expert": expert_layer}
    rng = np.random.default_rng(5)
    prefix = jnp.asarray(rng.normal(size=(1, 4, 16)).astype(np.float32))
    suffix = jnp.asarray(rng.normal(size=(1, 3, 16)).astype(np.float32))
    positions = jnp.arange(7, dtype=jnp.int32)[None]
    mask = jnp.ones((1, 7, 7), dtype=bool)
    cond = jnp.asarray(rng.normal(size=(1, 16)).astype(np.float32))

    def loss(params_tree, knowledge_insulation: bool):
        _, expert_out = joint_layer(params_tree, prefix, suffix, mask, positions, cond,
                                    num_heads=2, num_kv_heads=1, head_dim=8,
                                    knowledge_insulation=knowledge_insulation)
        return jnp.sum(jnp.square(expert_out))

    def flat_max(tree) -> float:
        leaves = [leaf for leaf in jax.tree_util.tree_leaves(tree) if leaf is not None]
        assert leaves, "gradient tree is empty"
        return float(max(jnp.max(jnp.abs(leaf)) for leaf in leaves))

    grads_on = jax.grad(lambda tree: loss(tree, True))(params)
    grads_off = jax.grad(lambda tree: loss(tree, False))(params)

    # The expert still trains under both settings; only the prefix side is isolated.
    assert flat_max(grads_on["expert"]) > 1e-7, "KI-on killed the expert gradient"
    assert flat_max(grads_off["expert"]) > 1e-7
    # Positive control: without KI the flow loss does reach the VLM.
    assert flat_max(grads_off["prefix"]) > 1e-7, "KI-off produced no VLM gradient: control is vacuous"
    # Isolation: every prefix leaf is exactly zero, and present (not None).
    prefix_leaves = jax.tree_util.tree_leaves(grads_on["prefix"])
    assert all(leaf is not None for leaf in prefix_leaves)
    assert all(float(jnp.max(jnp.abs(leaf))) == 0.0 for leaf in prefix_leaves)

    # The AdaRMS gate must be non-degenerate, otherwise the control above could not fire.
    gate = expert_layer["input_norm"]["dense_bias"][2 * 16:]
    assert float(jnp.max(jnp.abs(jnp.asarray(gate)))) > 1e-3


def test_prefix_cache_matches_fused_reference(g3: G3) -> None:
    """Split prefix/KV must equal the fused layer, keep gradients, and invalidate correctly."""
    prefix_layer, expert_layer = tiny_pair(seed=77)
    params = {"prefix": prefix_layer, "expert": expert_layer}
    rng = np.random.default_rng(11)
    prefix = jnp.asarray(rng.normal(size=(1, 4, 16)).astype(np.float32))
    suffix = jnp.asarray(rng.normal(size=(1, 3, 16)).astype(np.float32))
    positions = jnp.arange(7, dtype=jnp.int32)[None]
    mask = jnp.zeros((1, 7, 7), dtype=bool)
    mask = mask.at[:, :4, :4].set(True)   # prefix: bidirectional
    mask = mask.at[:, 4:, :].set(True)    # action: sees everything
    cond = jnp.asarray(rng.normal(size=(1, 16)).astype(np.float32))

    fused_prefix, fused_suffix = joint_layer(params, prefix, suffix, mask, positions, cond,
                                             num_heads=2, num_kv_heads=1, head_dim=8,
                                             knowledge_insulation=True)

    def split(tree, knowledge_insulation: bool):
        prefix_out, key, value = prefix_layer_with_cache(
            tree, prefix, positions[:, :4], mask[:, :4, :4],
            num_heads=2, num_kv_heads=1, head_dim=8)
        suffix_out = expert_layer_with_cache(
            tree, suffix, positions[:, 4:], mask[:, 4:, :], cond, key, value,
            num_heads=2, num_kv_heads=1, head_dim=8,
            knowledge_insulation=knowledge_insulation)
        return prefix_out, suffix_out

    split_prefix, split_suffix = split(params, True)
    np.testing.assert_allclose(np.asarray(split_prefix), np.asarray(fused_prefix), atol=1e-6, rtol=1e-6,
                               err_msg="cached prefix differs from the fused layer")
    np.testing.assert_allclose(np.asarray(split_suffix), np.asarray(fused_suffix), atol=1e-6, rtol=1e-6,
                               err_msg="cached suffix differs from the fused layer")

    def loss_from_split(tree, knowledge_insulation: bool):
        return jnp.sum(jnp.square(split(tree, knowledge_insulation)[1]))

    grads_on = jax.grad(lambda tree: loss_from_split(tree, True))(params)
    grads_off = jax.grad(lambda tree: loss_from_split(tree, False))(params)

    def flat_max(tree) -> float:
        return float(max(jnp.max(jnp.abs(leaf)) for leaf in jax.tree_util.tree_leaves(tree)
                         if leaf is not None))

    assert flat_max(grads_on["expert"]) > 1e-7
    assert flat_max(grads_off["prefix"]) > 1e-7, "cache path lost the KI-off control"
    assert all(float(jnp.max(jnp.abs(leaf))) == 0.0
               for leaf in jax.tree_util.tree_leaves(grads_on["prefix"])), "cache path leaks into the VLM"

    # Right padding: masking the last prefix row must leave the valid rows unchanged.
    padded_mask = mask.at[:, :, 3].set(False).at[:, 3, :].set(False)
    padded_positions = positions.at[:, 3].set(positions[:, 2])
    padded_prefix_out, _, _ = prefix_layer_with_cache(
        params, prefix, padded_positions[:, :4], padded_mask[:, :4, :4],
        num_heads=2, num_kv_heads=1, head_dim=8)
    short_prefix_out, _, _ = prefix_layer_with_cache(
        params, prefix[:, :3], positions[:, :3], mask[:, :3, :3],
        num_heads=2, num_kv_heads=1, head_dim=8)
    np.testing.assert_allclose(np.asarray(padded_prefix_out)[:, :3], np.asarray(short_prefix_out),
                               atol=1e-6, rtol=1e-6, err_msg="right padding changed the valid rows")

    # A fully masked query row must stay finite (the reference kernel yields a uniform
    # softmax over an all-masked row rather than NaN).
    blind_mask = mask.at[:, 6, :].set(False)
    _, blind_suffix = joint_layer(params, prefix, suffix, blind_mask, positions, cond,
                                  num_heads=2, num_kv_heads=1, head_dim=8,
                                  knowledge_insulation=True)
    assert np.isfinite(np.asarray(blind_suffix)).all(), "fully masked query produced NaN/Inf"

    # Cache identity: same observation reuses, any change invalidates.
    cache = make_prefix_cache(prefix, positions[:, :4], mask[:, :4, :4],
                              jnp.zeros((1, 4, 8)), jnp.zeros((1, 4, 8)))
    assert prefix_cache_matches(cache, prefix, positions[:, :4], mask[:, :4, :4])
    assert not prefix_cache_matches(cache, prefix + 1.0, positions[:, :4], mask[:, :4, :4])
    assert not prefix_cache_matches(cache, prefix, positions[:, :4] + 1, mask[:, :4, :4])
    assert not prefix_cache_matches(cache, prefix, positions[:, :4],
                                    mask[:, :4, :4].at[:, 0, 0].set(False))
