"""G0 — environment, provenance and reference baseline.

These tests must run where the assets and the GPU live, under the candidate
interpreter. Reference numbers always come from a separate read-only PyTorch
process; nothing here lets the candidate define its own expectation.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.migration import support as S


# --------------------------------------------------------------------------- #
# 1. Source and asset identity
# --------------------------------------------------------------------------- #
def test_source_and_asset_pins():
    """Every pinned byte is re-read and compared; unacquired assets stay blocked."""
    roots = S.roots()
    record = S.lock()
    assert record["schema"] == "fineart-jax-lock/v1", "lock schema must be the G0 one"

    reference = record["reference"]
    acquisition = reference["acquisition"]
    # A directory name is not a revision: an export without VCS history must say so.
    assert acquisition["vcs"] is None, "do not claim a VCS revision this export cannot prove"
    assert acquisition["public_pin"] == "blocked", "the public acquisition path is unresolved and must stay blocked"
    assert acquisition["public_pin_reason"].strip(), "a blocked pin needs a written reason"

    entries = S.walk_tree(Path(roots["REFERENCE_ROOT"]))
    aggregate, count, size = S.aggregate(entries)
    assert count == reference["file_count"], f"file count moved: {count} != {reference['file_count']}"
    assert size == reference["byte_count"], f"byte total moved: {size} != {reference['byte_count']}"
    assert aggregate == reference["aggregate_sha256"], "reference tree aggregate SHA-256 does not match the pin"

    locked = {item["path"]: item for item in reference["files"]}
    live = {item["path"]: item for item in entries}
    assert set(locked) == set(live), "reference file set changed since the pin was recorded"
    drifted = [
        path
        for path, item in live.items()
        if locked[path].get("sha256") != item.get("sha256")
        or locked[path].get("symlink_to") != item.get("symlink_to")
    ]
    assert not drifted, "reference bytes drifted: " + ", ".join(sorted(drifted)[:5])

    assets = record["assets"]
    assert "pi05_base" in assets, "the checkpoint asset must be pinned before any weight gate"
    base = assets["pi05_base"]
    assert base.get("status") is None, "pi05_base must be pinned, not blocked: " + str(base.get("reason"))
    assert base["declared_mismatches"] == [], "declared SHA256SUMS disagrees with the bytes on disk"
    assert base["params"]["file_count"] > 0 and base["params"]["byte_count"] > 0

    params_root = Path(roots["ASSET_ROOT"]) / "assets" / "pi05_base" / "params"
    live_params = []
    for path in sorted(params_root.rglob("*")):
        if path.is_file():
            live_params.append(
                {
                    "path": str(path.relative_to(params_root)),
                    "bytes": path.stat().st_size,
                    "sha256": S.sha256_file(path),
                }
            )
    params_aggregate, params_count, params_bytes = S.aggregate(live_params)
    assert params_count == base["params"]["file_count"], "checkpoint file count changed"
    assert params_bytes == base["params"]["byte_count"], "checkpoint byte total changed"
    assert params_aggregate == base["params"]["aggregate_sha256"], "checkpoint bytes drifted from the pin"
    pinned_params = {item["path"]: item["sha256"] for item in base["params"]["files"]}
    assert pinned_params == {item["path"]: item["sha256"] for item in live_params}

    assert "tokenizers" in assets, "tokenizer identity is required for G1"
    tokenizers = assets["tokenizers"]
    live_tokenizers = S.walk_tree(Path(roots["ASSET_ROOT"]) / "tokenizers")
    token_aggregate, token_count, token_bytes = S.aggregate(live_tokenizers)
    assert token_count == tokenizers["file_count"]
    assert token_bytes == tokenizers["byte_count"]
    assert token_aggregate == tokenizers["aggregate_sha256"], "tokenizer files drifted"

    openpi = record["openpi"]
    assert openpi["status"] in {"pinned_subset", "blocked"}, "OpenPI must be either pinned or explicitly blocked"
    if openpi["status"] == "blocked":
        assert openpi["reason"].strip(), "a blocked OpenPI reference needs a reason"
    else:
        assert openpi["subpaths"], "a pinned OpenPI reference needs at least one hashed subpath"
        for name, value in openpi["subpaths"].items():
            assert value.get("status") != "missing", "pinned OpenPI subpath is missing: " + name

    gaps = {item["id"] for item in record.get("blocked", [])}
    assert "fineart_mid_trained_checkpoint" in gaps, (
        "the absence of a FineART-mid-trained checkpoint must be recorded as a blocked asset, "
        "not silently replaced by the pi0.5 base checkpoint"
    )
    for item in record["blocked"]:
        assert item["reason"].strip(), "every blocked entry needs a reason"

    S.assert_no_placeholders(record.get("environment_gaps", {}), "environment_gaps")


# --------------------------------------------------------------------------- #
# 2. Locked, independent environments
# --------------------------------------------------------------------------- #
def test_locked_environments():
    """Two independent Python 3.12 environments, each rebuildable from its frozen lock."""
    roots = S.roots()
    record = S.lock()
    environments = record.get("environments", {})
    assert set(environments) >= {"ref", "jax"}, "both environments must be locked"

    jax_identity = S.python_identity(roots["JAX_PYTHON"])
    ref_identity = S.python_identity(roots["REF_PYTHON"])
    for name, identity in (("jax", jax_identity), ("ref", ref_identity)):
        assert identity["version"].startswith("3.12."), f"{name} must run Python 3.12, got {identity['version']}"
    assert Path(jax_identity["prefix"]).resolve() != Path(ref_identity["prefix"]).resolve(), (
        "the reference and the candidate must not share an environment"
    )

    for name, python in (("jax", roots["JAX_PYTHON"]), ("ref", roots["REF_PYTHON"])):
        live = S.pip_freeze(python)
        locked = environments[name]["freeze"]
        assert sorted(live) == sorted(locked), (
            f"{name} environment drifted from its frozen lock; "
            f"missing={sorted(set(locked) - set(live))[:5]} extra={sorted(set(live) - set(locked))[:5]}"
        )

    jax_packages = S.freeze_map(environments["jax"]["freeze"])
    ref_packages = S.freeze_map(environments["ref"]["freeze"])

    # The frameworks must not leak into each other's environment.
    assert "torch" not in jax_packages, "the candidate environment must not carry PyTorch"
    assert "jax" not in ref_packages and "jaxlib" not in ref_packages, (
        "the reference environment must not carry JAX"
    )
    assert "torch" in ref_packages and "jax" in jax_packages

    # A shared tokenizer stack version is required: the contract forbids mixing them.
    assert ref_packages.get("transformers") == jax_packages.get("transformers"), (
        "reference and candidate must not run different transformers versions"
    )
    assert ref_packages.get("transformers", "").startswith("5.22."), (
        "the reference requires the transformers 5.22 series, got " + str(ref_packages.get("transformers"))
    )
    assert "sentencepiece" in ref_packages and "sentencepiece" in jax_packages, (
        "FAST supervision needs sentencepiece in both environments"
    )

    baseline = record.get("base_python", {})
    assert baseline.get("version", "").startswith("3.12."), "the shared interpreter must be pinned"
    assert baseline.get("sha256"), "the interpreter archive must be pinned by SHA-256"
    assert baseline.get("source_url", "").startswith("https://"), "the interpreter needs a recorded source"
    assert baseline.get("verified_sha256") is True, "the pinned interpreter hash must have been verified after download"

    cuda = record.get("cuda_runtime", {})
    assert str(cuda.get("torch_cuda", "")).startswith("12."), "the reference must use a CUDA 12 runtime"
    assert cuda.get("cudnn"), "cuDNN version must be recorded"
    assert "cuda12" in json.dumps(environments["jax"]).lower() or "nvidia" in json.dumps(environments["jax"]).lower(), (
        "the candidate lock must record that JAX runs on CUDA 12"
    )


# --------------------------------------------------------------------------- #
# 3. Real GPU execution in both frameworks
# --------------------------------------------------------------------------- #
_TORCH_PROBE = r"""
import json, os
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
info = {
  "torch": torch.__version__,
  "torch_cuda": torch.version.cuda,
  "cudnn": torch.backends.cudnn.version(),
  "available": torch.cuda.is_available(),
  "device_count": torch.cuda.device_count(),
}
if not info["available"]:
    info["error"] = "no CUDA device"
    info["finite"] = False
    print(json.dumps(info)); raise SystemExit(0)
info["device_name"] = torch.cuda.get_device_name(0)
info["capability"] = list(torch.cuda.get_device_capability(0))
info["driver"] = torch.cuda.get_device_properties(0).name
a = torch.randn(512, 512, device="cuda", dtype=torch.float32, requires_grad=True)
b = torch.randn(512, 512, device="cuda", dtype=torch.float32)
loss = (a @ b).pow(2).mean()
loss.backward()
torch.cuda.synchronize()
info["loss"] = float(loss.detach())
info["grad_finite"] = bool(torch.isfinite(a.grad).all())
info["output_finite"] = bool(torch.isfinite(loss).all())
info["device_of_result"] = "cuda" if loss.device.type == "cuda" else loss.device.type
info["memory_allocated"] = int(torch.cuda.memory_allocated(0))
print(json.dumps(info))
"""

_JAX_PROBE = r"""
import json
import jax, jax.numpy as jnp
devices = jax.devices()
info = {
  "jax": jax.__version__,
  "jaxlib": jax.lib.__version__ if hasattr(jax, "lib") else None,
  "backend": jax.default_backend(),
  "devices": [{"platform": d.platform, "kind": d.device_kind} for d in devices],
}
def f(a, b):
    return jnp.mean((a @ b) ** 2)
grad = jax.grad(lambda x: f(x, b))(a)
out = jax.jit(f)(a, b)
out.block_until_ready()
info["loss"] = float(out)
info["output_finite"] = bool(jnp.isfinite(out))
info["grad_finite"] = bool(jnp.isfinite(grad).all())
try:
    info["result_platforms"] = sorted({d.platform for d in out.devices()})
except Exception:
    info["result_platforms"] = []
print(json.dumps(info))
"""


def test_real_cuda_execution():
    """Both frameworks must execute real kernels on the GPU; CPU fallback fails."""
    roots = S.roots()
    record = S.lock()

    torch_info = S.run_python(roots["REF_PYTHON"], _TORCH_PROBE, timeout=900)
    assert torch_info["available"] and torch_info["device_count"] >= 1, "torch sees no CUDA device"
    assert torch_info["device_of_result"] == "cuda", "torch matrix multiply did not run on the GPU"
    assert torch_info["output_finite"] and torch_info["grad_finite"], "torch produced non-finite values"
    assert "V100" in torch_info["device_name"], "unexpected device: " + torch_info["device_name"]
    assert torch_info["capability"] == [7, 0], "compute capability must be 7.0, got " + str(torch_info["capability"])

    jax_probe = _JAX_PROBE.replace(
        "import jax, jax.numpy as jnp",
        "import jax, jax.numpy as jnp\n"
        "a = jnp.asarray(__import__('numpy').random.RandomState(0).randn(512,512).astype('float32'))\n"
        "b = jnp.asarray(__import__('numpy').random.RandomState(1).randn(512,512).astype('float32'))",
    )
    jax_info = S.run_python(roots["JAX_PYTHON"], jax_probe, timeout=900)
    assert jax_info["backend"] == "gpu", "JAX backend is " + str(jax_info["backend"]) + "; CPU fallback is forbidden"
    platforms = [device["platform"] for device in jax_info["devices"]]
    assert "gpu" in platforms, "JAX exposes no GPU device"
    assert jax_info["output_finite"] and jax_info["grad_finite"], "JAX produced non-finite values"
    if jax_info.get("result_platforms"):
        assert jax_info["result_platforms"] == ["gpu"], (
            "JAX jitted result landed on " + str(jax_info["result_platforms"])
        )

    device_profile = S.contract()["device_profile"]
    assert torch_info["device_name"] == device_profile["gpu"], (
        "the measured device does not match the locked device profile"
    )

    evidence = S.evidence_dir("g0") / "cuda_execution.json"
    evidence.write_text(
        json.dumps({"torch": torch_info, "jax": jax_info}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    locked = record.get("cuda_runtime", {})
    assert locked.get("torch") == torch_info["torch"], "locked torch version differs from the runtime"
    assert str(locked.get("torch_cuda")) == str(torch_info["torch_cuda"]), "locked CUDA runtime differs"
    assert str(locked.get("cudnn")) == str(torch_info["cudnn"]), "locked cuDNN differs"


# --------------------------------------------------------------------------- #
# 4. Resource budget against the real default parameter set
# --------------------------------------------------------------------------- #
_BUDGET_PROBE = r"""
import json
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig
from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy
from lerobot.utils.constants import ACTION

config = FineARTVLAConfig(
    device="cpu",
    dtype=torch.float32,
    input_features={
        "observation.images.base": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(32,)),
    },
    output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,))},
)
with torch.device("cpu"):
    policy = FineARTVLAPolicy(config)
total = 0
grouped = {"backbone": 0, "action_expert": 0, "vision": 0, "other": 0}
for name, parameter in policy.named_parameters():
    count = parameter.numel()
    total += count
    if "vision_tower" in name or "multi_modal_projector" in name:
        grouped["vision"] += count
    elif "action_expert" in name or "action_in_proj" in name or "action_out_proj" in name:
        grouped["action_expert"] += count
    elif name.startswith("model.paligemma_with_expert.paligemma"):
        grouped["backbone"] += count
    else:
        grouped["other"] += count
print(json.dumps({"total": total, "groups": grouped, "device": "cpu"}))
"""


def test_hardware_resource_budget():
    """Record the real default parameter budget and keep G7 honest about feasibility."""
    roots = S.roots()
    record = S.lock()
    contract = S.contract()

    counts = S.run_python(roots["REF_PYTHON"], _BUDGET_PROBE, timeout=1800, env_extra=S.reference_env(roots))
    total = counts["total"]
    assert total > 1_000_000_000, "parameter count is implausibly small for the default profile: " + str(total)

    gpu_capacity = contract["device_profile"]["vram_bytes"]
    # FP32 weights + gradients + two AdamW moments. Activations, KV cache and
    # allocator slack are excluded on purpose: they only make the verdict worse.
    optimizer_bytes = 16 * total
    weights_bytes = 4 * total

    pinned_bytes = record["assets"]["pi05_base"]["params"]["byte_count"]
    ratio = pinned_bytes / float(weights_bytes)
    assert 0.5 <= ratio <= 1.5, (
        "pinned checkpoint bytes and the analytical FP32 weight size disagree by "
        f"{ratio:.2f}x; the parameter count or the asset identity is wrong"
    )

    budget = {
        "parameter_count": total,
        "parameter_groups": counts["groups"],
        "weights_fp32_bytes": weights_bytes,
        "optimizer_state_bytes": optimizer_bytes,
        "total_training_bytes": optimizer_bytes,
        "vram_bytes": gpu_capacity,
        "g7_fits_vram": optimizer_bytes <= gpu_capacity,
        "headroom_bytes": gpu_capacity - optimizer_bytes,
        "excluded": "activations, KV cache, temporaries, allocator slack",
    }
    evidence = S.evidence_dir("g0") / "resource_budget.json"
    evidence.write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")

    locked = record["resource_budget"]
    assert locked["parameter_count"] == total, "the recorded parameter count is stale"
    assert locked["g7_fits_vram"] == budget["g7_fits_vram"], "the recorded G7 verdict contradicts the measurement"
    assert isinstance(locked["g7_fits_vram"], bool), "the G7 verdict must be a recorded boolean, not a placeholder"
    # G0 may legitimately pass while G7 is blocked; what it may not do is claim the opposite.
    if not budget["g7_fits_vram"]:
        assert record.get("blocked"), "an over-budget GPU must produce an explicit G7 entry in blocked"

    assert contract["g7_resource_eligibility"]["accounting"].startswith("16 * N"), (
        "the budget accounting rule must stay as locked"
    )
    # The checkpoint bytes and the constructed model disagree; that must stay visible.
    assert locked.get("checkpoint_param_equivalent_fp32"), "the checkpoint-equivalent count must be recorded"
    assert locked["checkpoint_gap_note"].strip(), "an unexplained parameter-count gap is not acceptable"
    assert "checkpoint_vs_model_parameter_gap" in locked.get("g7_open_items", []), (
        "the unresolved count gap must be carried as an explicit G7 open item"
    )
    if locked["checkpoint_vs_model_ratio"] < 0.95:
        assert locked["g7_fits_vram"] is False, (
            "the smaller checkpoint-equivalent count must not be used to declare G7 feasible"
        )


# --------------------------------------------------------------------------- #
# 5. Reference repeatability
# --------------------------------------------------------------------------- #
def _export_fixture(python: str, destination: Path, reference_root: str) -> dict:
    oracle = S.PROJECT / "tests" / "migration" / "oracle" / "export_tiny_reference.py"
    environment = {
        "PYTHONPATH": str(Path(reference_root) / "src"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [python, str(oracle), "--out", str(destination)],
        capture_output=True,
        text=True,
        timeout=3600,
        env={**os.environ, **environment},
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "reference export failed\n--- stdout ---\n" + result.stdout[-6000:]
            + "\n--- stderr ---\n" + result.stderr[-6000:]
        )
    provenance = json.loads((destination / "reference.provenance.json").read_text(encoding="utf-8"))
    return provenance


def test_reference_repeatability():
    """Prove the oracle replays before anyone treats its numbers as expectations."""
    roots = S.roots()
    record = S.lock()
    evidence = S.evidence_dir("g0")

    first_dir = evidence / "oracle-run-1"
    second_dir = evidence / "oracle-run-2"
    first = _export_fixture(roots["REF_PYTHON"], first_dir, roots["REFERENCE_ROOT"])
    second = _export_fixture(roots["REF_PYTHON"], second_dir, roots["REFERENCE_ROOT"])

    assert first["array_sha256"] == second["array_sha256"], (
        "the oracle is not deterministic: repeated runs disagree, so no golden may be exported yet"
    )

    arrays = np.load(str(first_dir / "reference.npz"))
    for name in arrays.files:
        assert np.isfinite(arrays[name]).all(), "reference fixture contains non-finite values: " + name

    # Non-degeneracy: a constant or zero output would make later gates vacuous.
    loss = float(arrays["output.loss"])
    assert np.isfinite(loss) and loss > 0.0, "reference loss must be a positive finite number"
    measured = {key: value for key, value in first["grad_norms"].items() if value is not None}
    assert set(measured) >= {"backbone", "action_expert"}, (
        "the gradient groups the flow path must reach are missing: " + str(sorted(measured))
    )
    grad_total = sum(measured.values())
    assert grad_total > 0.0, "all reference gradients are zero; the fixture cannot discriminate"
    for group, value in measured.items():
        assert np.isfinite(value), "non-finite gradient norm for group " + group
    assert measured["backbone"] > 0.0, "the VLM backbone received no gradient in the reference step"
    assert measured["action_expert"] > 0.0, "the action expert received no gradient in the reference step"

    source_aggregate = record["reference"]["aggregate_sha256"]
    assert first["config"]["knowledge_insulation"] is True, "the G0 fixture must exercise knowledge insulation"

    locked_fixture = record.get("reference_fixture", {})
    assert locked_fixture.get("oracle_sha256"), "the oracle script must be pinned by hash"
    oracle_path = S.PROJECT / "tests" / "migration" / "oracle" / "export_tiny_reference.py"
    assert locked_fixture["oracle_sha256"] == S.sha256_file(oracle_path), (
        "the oracle script changed after its fixture was recorded; the fixture is stale"
    )
    assert locked_fixture.get("source_aggregate_sha256") == source_aggregate, (
        "the fixture was exported from a different reference tree than the current pin"
    )
    assert locked_fixture.get("array_sha256") == first["array_sha256"], (
        "the exported fixture does not match the recorded golden hashes"
    )
    assert locked_fixture.get("loss_terms") == first["loss_terms"], "recorded loss terms differ from the fixture"
    assert locked_fixture.get("environment", {}).get("torch") == first["torch"], (
        "the fixture was produced by a different torch build than the locked environment"
    )


# --------------------------------------------------------------------------- #
# 6. The acceptance contract is locked
# --------------------------------------------------------------------------- #
EXPECTED_THRESHOLDS = {
    "data_preprocessing": {"atol": 1e-06, "rtol": 1e-06},
    "tiny_numeric": {"atol": 1e-05, "rtol": 0.0001},
    "tiny_gradient": {"relative_l2_max": 0.001, "cosine_min": 0.9999, "near_zero_max_abs": 1e-06},
    "knowledge_insulation": {"flow_to_vlm_max_abs": 1e-10, "ki_off_positive_control_min_abs": 1e-07},
    "optimizer_single_step": {"atol": 1e-07, "rtol": 0.0001},
    "trajectory_20_steps": {"atol": 1e-05, "rtol": 0.001},
    "real_prefix_velocity_logits": {"relative_l2_max": 0.005, "max_abs": 0.01},
    "real_losses": {"atol": 1e-05, "rtol": 0.005},
    "denoised_actions": {"mae_max": 0.001, "max_abs": 0.01},
}

EXPECTED_DEFAULTS = {
    "flow_loss_weight": 10.0,
    "text_loss_weight": 1.0,
    "fast_action_loss_weight": 1.0,
    "text_ce_z_loss_weight": 0.0001,
    "knowledge_insulation": True,
    "flow_num_repeats": 5,
    "fast_skip_tokens": 1152,
    "max_action_tokens": 256,
    "chunk_size": 50,
    "tokenizer_max_length": 200,
    "max_action_dim": 32,
    "max_state_dim": 32,
    "num_inference_steps": 10,
    "optimizer_lr": 2.5e-05,
    "optimizer_betas": [0.9, 0.95],
    "optimizer_eps": 1e-08,
    "optimizer_weight_decay": 0.01,
    "optimizer_grad_clip_norm": 1.0,
    "scheduler_warmup_steps": 1000,
    "scheduler_decay_steps": 30000,
    "scheduler_decay_lr": 2.5e-06,
    "lm_head_lr_scale": 1.0,
    "backbone_lr_scale": 1.0,
    "action_expert_lr_scale": 1.0,
    "conditioning_lr_scale": 1.0,
}


def test_acceptance_contract_locked():
    """The contract may not drift, soften a threshold, or hide a TODO."""
    contract = S.contract()

    S.assert_no_placeholders(contract["thresholds"], "thresholds")
    S.assert_no_placeholders(contract["defaults"], "defaults")
    S.assert_no_placeholders(contract["rejection_table"], "rejection_table")

    for name, expected in EXPECTED_THRESHOLDS.items():
        assert name in contract["thresholds"], "missing threshold class: " + name
        assert contract["thresholds"][name] == expected, (
            f"threshold '{name}' was changed: {contract['thresholds'][name]} != {expected}"
        )

    for name, expected in EXPECTED_DEFAULTS.items():
        assert name in contract["defaults"], "missing default: " + name
        assert contract["defaults"][name] == expected, (
            f"default '{name}' was changed: {contract['defaults'][name]} != {expected}"
        )

    assert contract["defaults"]["loss_weight_triplet"] == [10.0, 1.0, 1.0], "the 10/1/1 trio is the default"
    assert contract["defaults"]["recipe_blend"] == {"high_level_subtask": 0.3, "low_level_execution": 0.7}

    matrix = contract["config_matrix"]
    assert matrix["action_dims"] == [6, 12, 14, 32], "real action dimensions must cover the required matrix"
    assert matrix["camera_counts"] == [1, 2, 3]
    assert matrix["flow_draws"] == [1, 5]
    assert sorted(matrix["times"]) == [0.001, 0.5, 1.0]
    assert matrix["fast_skip_offsets_under_test"] == [1152, 128], (
        "both the default FAST offset and the checkpoint-declared offset must be tested"
    )
    assert matrix["reduction_modes"] == ["mean", "none"]
    assert set(matrix["routing_cases"]) >= {
        "text_only", "flow_only", "flow_plus_fast", "all_three", "text_all_ignored",
        "no_action_batch", "missing_fast_fields", "all_targets_disabled", "recipe_none_fallback",
    }

    shapes = contract["shapes"]
    assert shapes["tiny_model"]["paligemma_variant"] == "gemma_2b", (
        "the smallest valid language tower is gemma_2b; a 300m VLM fails the 2048 projection concat"
    )
    assert shapes["tiny_model"]["action_expert_variant"] == "gemma_300m"
    assert shapes["tiny_model"]["vocab_size"] == 257152, "the real vocabulary size must not be lowered"
    assert shapes["real_batch"] == {
        "batch_size": 1, "text_tokens_max": 200, "fast_tokens_max": 256,
        "chunk_size": 50, "state_dim": 32, "action_dim": 32,
    }

    assert contract["training_contract"]["batch_size"] == 1
    assert contract["training_contract"]["optimizer_updates_g7"] == 100

    rejected = contract["rejection_table"]["rejected_config_combinations"]
    assert "enable_fast_action_loss=true with recipe=None" in rejected
    assert "train_expert_only=true with text_loss_weight>0 or enable_fast_action_loss" in rejected
    assert "flow_num_repeats<1" in rejected
    assert "cpu_fallback" in contract["rejection_table"]["unsupported_in_fp32_default_profile"]

    anti_cheat = contract["anti_cheat"]["per_stage_negative_controls"]
    for stage in ("G1", "G2", "G3", "G4", "G5", "G6", "G7"):
        assert anti_cheat.get(stage), "every stage needs a negative control: " + stage

    assert contract["device_profile"]["gpu"] == "Tesla V100S-PCIE-32GB"
    assert contract["device_profile"]["dtype"] == "float32"
    assert contract["device_profile"]["cpu_fallback"] == "forbidden"
