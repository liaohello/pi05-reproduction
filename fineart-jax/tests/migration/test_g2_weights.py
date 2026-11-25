"""G2 — strict weight adaptation and base operators (draft)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tests.migration import support as S

ORACLE = S.PROJECT / "tests" / "migration" / "oracle" / "export_g2_ops.py"
HEAD_ORACLE = S.PROJECT / "tests" / "migration" / "oracle" / "export_expert_head.py"


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def assert_real(actual, expected, label: str) -> None:
    """Frozen real-model bucket for full-width operator outputs.

    These operators use the default 2048/1024 widths and the real vocabulary;
    backend trig/matmul last-bit differences are judged by the contract's
    real-prefix row, not by the tiny-model row. Structural quantities stay exact.
    """
    limits = S.contract()["thresholds"]["real_prefix_velocity_logits"]
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    delta = np.abs(actual - expected)
    rel = np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-12)
    assert rel <= limits["relative_l2_max"] and delta.max() <= limits["max_abs"], (
        f"{label}: relative_l2={rel:.3e} max_abs={delta.max():.3e}"
    )


def clean_env() -> dict:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false"})
    return env


class G2:
    def __init__(self):
        self.roots = S.roots()
        self.evidence = S.evidence_dir("g2")
        self.checkpoint = Path(self.roots["ASSET_ROOT"]) / "assets" / "pi05_base" / "params"
        self.audit = self.evidence / "audit"
        self.oracle_root = self.evidence / "oracle"
        self._goldens = {}
        self._expert_head = None

    def reference_env(self) -> dict:
        return {**S.reference_env(self.roots), **clean_env()}

    def export(self, group: str, script: Path) -> dict:
        if group in self._goldens:
            return self._goldens[group]
        runs = []
        for index in (1, 2):
            target = self.oracle_root / group / f"run{index}"
            if (target / f"{group}.json").is_file():
                # A completed double export is reused verbatim: the determinism
                # proof is exactly the run1/run2 comparison below.
                runs.append(target)
                continue
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)
            cmd = [self.roots["REF_PYTHON"], str(script), "--group", group, "--out", str(target)] if script == ORACLE else [
                self.roots["REF_PYTHON"], str(script), "--out", str(target)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                                    env=self.reference_env(), check=False)
            if result.returncode:
                raise AssertionError(f"G2 oracle {group} failed\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
            runs.append(target)
        first = json.loads((runs[0] / f"{group}.json").read_text())
        second = json.loads((runs[1] / f"{group}.json").read_text())
        assert first == second, f"G2 oracle {group} is not deterministic"
        arrays = dict(np.load(runs[0] / f"{group}.npz"))
        for name, info in first["arrays"].items():
            assert list(arrays[name].shape) == info["shape"]
            assert sha(arrays[name]) == info["sha256"]
        payload = {"arrays": arrays, "manifest": first}
        self._goldens[group] = payload
        return payload

    def golden(self, group: str) -> dict:
        return self.export(group, ORACLE)

    def expert_head(self) -> np.ndarray:
        if self._expert_head is None:
            payload = self.export("expert_head", HEAD_ORACLE)
            self._expert_head = payload["arrays"]["head"]
        return self._expert_head

    def audits(self):
        torch_audit = self.audit / "torch_params.json"
        orbax_audit = self.audit / "orbax_params.json"
        if not torch_audit.is_file() or not orbax_audit.is_file():
            raise AssertionError("G2 reference audits are missing from the evidence directory")
        return json.loads(torch_audit.read_text()), json.loads(orbax_audit.read_text())


@pytest.fixture(scope="session")
def g2() -> G2:
    return G2()


# --------------------------------------------------------------------------- #
# 1. Strict parameter mapping
# --------------------------------------------------------------------------- #
def test_strict_parameter_mapping(g2: G2):
    from fineart_jax import weights as W

    torch_audit, orbax_audit = g2.audits()
    checkpoint = W.load_checkpoint(g2.checkpoint)
    assert set(checkpoint) == {row["name"] for row in orbax_audit["logical_tensors"]}
    registered = W.map_to_registered(checkpoint, g2.expert_head())
    expected = {row["name"]: row for row in torch_audit["parameter_rows"]}
    assert set(registered) == set(expected), "strict parameter set differs from the reference"
    for name, array in registered.items():
        row = expected[name]
        assert list(array.shape) == row["shape"], f"shape mismatch: {name}"
        assert str(array.dtype) == "float32", f"dtype mismatch: {name}"
        assert array.size == row["numel"], f"element count mismatch: {name}"
    manifest = W.strict_manifest(checkpoint, g2.expert_head())
    assert manifest["count"] == len(expected) == 813
    assert all(row["source"] is None or row["name"].endswith("gemma_expert.lm_head.weight")
               for row in manifest["parameters"])
    W.validate_registered(registered, list(expected.values()))
    W.validate_expert_head(g2.expert_head(), g2.export("expert_head", HEAD_ORACLE)["manifest"])

    # Negative controls must go through the production mapping/validation API.
    first = next(iter(expected))
    dropped = {key: value for key, value in registered.items() if key != first}
    with pytest.raises(ValueError):
        W.validate_registered(dropped, list(expected.values()))
    extra = dict(registered)
    extra["not.a.parameter"] = np.zeros(1, dtype=np.float32)
    with pytest.raises(ValueError):
        W.validate_registered(extra, list(expected.values()))
    victim = next(name for name in expected if len(expected[name]["shape"]) == 2
                  and expected[name]["shape"][0] != expected[name]["shape"][1])
    transposed = dict(registered)
    transposed[victim] = registered[victim].T
    with pytest.raises(ValueError):
        W.validate_registered(transposed, list(expected.values()))
    wrong_dtype = dict(registered)
    wrong_dtype[victim] = wrong_dtype[victim].astype(np.float64)
    with pytest.raises(ValueError):
        W.validate_registered(wrong_dtype, list(expected.values()))
    wrong_head = g2.expert_head().copy()
    wrong_head[0, 0] += 1.0
    with pytest.raises(ValueError):
        W.validate_expert_head(wrong_head, g2.export("expert_head", HEAD_ORACLE)["manifest"])


# --------------------------------------------------------------------------- #
# 2. Full-checkpoint reversible roundtrip
# --------------------------------------------------------------------------- #
def test_weight_roundtrip(g2: G2):
    from fineart_jax import weights as W

    _, audit = g2.audits()
    checkpoint = W.load_checkpoint(g2.checkpoint)
    registered = W.map_to_registered(checkpoint, g2.expert_head())
    shapes = {row["name"]: tuple(row["shape"]) for row in audit["logical_tensors"]}
    restored = W.map_to_checkpoint(registered, shapes)
    assert set(restored) == set(checkpoint)
    for name, original in checkpoint.items():
        rebuilt = restored[name]
        assert rebuilt.shape == original.shape and rebuilt.dtype == original.dtype
        assert sha(rebuilt) == sha(original), f"roundtrip bytes changed: {name}"
    embed = registered["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"]
    head = registered["model.paligemma_with_expert.paligemma.lm_head.weight"]
    assert sha(embed) == sha(head), "the declared VLM head clone was not preserved"


# --------------------------------------------------------------------------- #
# 3. QKV, GQA, RoPE and output projection
# --------------------------------------------------------------------------- #
def test_qkv_rope_projection(g2: G2):
    from fineart_jax.nn import attention as A

    golden = g2.golden("qkv")
    a = golden["arrays"]
    x, wq, wkv, wo = (jnp.asarray(a[key]) for key in ("x", "w_q", "w_kv", "w_o"))
    positions = jnp.asarray(a["positions"])
    q, k, v = A.split_qkv(x, wq, wkv, 8, 1, 256)
    q, k, v = A.prepare_qkv(q, k, v, positions, 256)
    encoded = jnp.asarray(a["encoded"])
    projected = A.output_projection(encoded, wo)
    assert_real(q, a["q_after_rope_scale"], "q")
    assert_real(k, a["k_after_rope"], "k")
    assert_real(v, a["v"], "v")
    assert_real(encoded, a["encoded"], "encoded")
    assert_real(projected, a["projected"], "projected")

    # Negative control: altering the RoPE positions must change the result.
    _, shifted, _ = A.prepare_qkv(*A.split_qkv(x, wq, wkv, 8, 1, 256), positions + 1, 256)
    assert not np.allclose(np.asarray(shifted), a["k_after_rope"])


# --------------------------------------------------------------------------- #
# 4. RMSNorm and AdaRMS
# --------------------------------------------------------------------------- #
def test_rmsnorm_adarms(g2: G2):
    from fineart_jax.nn import norms as N

    golden = g2.golden("norm")
    a = golden["arrays"]
    x = jnp.asarray(a["x"])
    plain = N.rms_norm(x, jnp.asarray(a["plain_weight"]), eps=1e-6)
    np.testing.assert_allclose(np.asarray(plain), a["plain_out"], rtol=1e-4, atol=1e-5)
    w, b = jnp.asarray(a["dense_weight"]), jnp.asarray(a["dense_bias"])
    for name in ("per_sample", "per_token"):
        value, gate = N.ada_rms(x, w, b, jnp.asarray(a[f"{name}__cond"]), eps=1e-6)
        np.testing.assert_allclose(np.asarray(value), a[f"{name}__out"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.asarray(gate), a[f"{name}__gate"], rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError):
        N.ada_rms(x, w, b, jnp.zeros((2, 7), dtype=jnp.float32), eps=1e-6)


# --------------------------------------------------------------------------- #
# 5. Vision patch / position / projector and prefix assembly
# --------------------------------------------------------------------------- #
def _vision_params(g2):
    """Pinned pi05_base vision subtree exported by the external asset tool."""
    from safetensors.numpy import load_file

    weights = load_file(os.environ["G2_VISION_WEIGHTS"])
    params = {key: jnp.asarray(value) for key, value in weights.items()
              if not key.startswith("projector.")}
    return params, jnp.asarray(weights["projector.weight"]), jnp.asarray(weights["projector.bias"])


def test_vision_prefix_embeddings(g2: G2):
    from fineart_jax.nn import vision as V

    golden = g2.golden("vision")
    a = golden["arrays"]
    params, projector_weight, projector_bias = _vision_params(g2)
    for count in (1, 2, 3):
        images = jnp.asarray(a[f"image_c{count}"])
        patches = V.patch_embed(images, params["embeddings.patch_embedding.weight"],
                                params["embeddings.patch_embedding.bias"])
        with_pos = V.add_position(patches, params["embeddings.position_embedding.weight"])
        assert_real(patches, a[f"patches_c{count}"], f"patch_c{count}")
        assert_real(with_pos, a[f"with_pos_c{count}"], f"with_pos_c{count}")
        final = V.vision_tower(images, params)
        assert_real(final, a[f"final_c{count}"], f"final_c{count}")
        projected = V.project(final, projector_weight, projector_bias)
        assert_real(projected, a[f"projected_c{count}"], f"projected_c{count}")
        text = jnp.asarray(a[f"text_c{count}"])
        text_mask = jnp.asarray(a[f"text_mask_c{count}"])
        image_mask = jnp.asarray(a[f"image_mask_c{count}"])
        prefix, mask = V.assemble_prefix([projected], [image_mask], text, text_mask, 257152)
        assert_real(prefix, a[f"prefix_tokens_c{count}"], f"prefix_c{count}")
        np.testing.assert_array_equal(np.asarray(mask), a[f"prefix_mask_c{count}"])
    # Full-tower intermediate and post-norm.
    images = jnp.asarray(a["image_c1"])
    _, hidden = V.vision_tower(images, params, return_hidden_states=True)
    assert_real(hidden[0], a["hidden0_c1"], "hidden0")
    assert_real(V.vision_tower(images, params), a["hidden_last_c1"], "hidden_last_postnorm")
    # B=2, C=3: camera axis concatenated, batch preserved, one false camera mask.
    images = jnp.asarray(a["image_b2c3"]).reshape(6, 3, 224, 224)
    final = V.vision_tower(images, params).reshape(2, 3, 256, 1152)
    projected = V.project(final.reshape(2, 3 * 256, 1152), projector_weight, projector_bias)
    image_mask = jnp.asarray(a["image_mask_b2c3"])
    text = jnp.asarray(a["text_b2c3"])
    text_mask = jnp.asarray(a["text_mask_b2c3"])
    prefix, mask = V.assemble_prefix(
        [projected[:, index * 256 : (index + 1) * 256, :] for index in range(3)],
        [image_mask[:, index, :] for index in range(3)],
        text, text_mask, 257152,
    )
    assert_real(prefix, a["prefix_tokens_b2c3"], "prefix_b2c3")
    np.testing.assert_array_equal(np.asarray(mask), a["prefix_mask_b2c3"])
    # The 224x224 FP32 pixels are the same saved arrays on both sides; no resize.
    assert a["image_c1"].shape == (1, 3, 224, 224)


# --------------------------------------------------------------------------- #
# 6. Embedding / head tying and dtype
# --------------------------------------------------------------------------- #
def test_embedding_head_tying_and_dtype(g2: G2):
    from fineart_jax.nn import embedding as E

    golden = g2.golden("embedding")
    a = golden["arrays"]
    table = jnp.asarray(a["table"])
    ids = jnp.asarray(a["token_ids"])
    embedded = E.embed_tokens(ids, table)
    logits = E.head_logits(embedded, jnp.asarray(a["head_small"]))
    np.testing.assert_allclose(np.asarray(embedded), a["embedded"], rtol=1e-4, atol=1e-5)
    assert_real(logits, a["logits_small"], "head logits")
    np.testing.assert_array_equal(np.asarray(a["head"]), np.asarray(a["table"]))
    assert len(a["token_ids"][0]) == 5 and a["token_ids"][0][-1] == 257151
    assert str(embedded.dtype) == "float32" and str(logits.dtype) == "float32"
