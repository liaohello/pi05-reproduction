"""Fail-closed contracts shared by the clean60 export and training chain."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

FEATURE_NAMES = [
    "left_shoulder_pan.pos", "left_shoulder_lift.pos", "left_elbow_flex.pos",
    "left_wrist_flex.pos", "left_wrist_roll.pos", "left_gripper.pos",
    "right_shoulder_pan.pos", "right_shoulder_lift.pos", "right_elbow_flex.pos",
    "right_wrist_flex.pos", "right_wrist_roll.pos", "right_gripper.pos",
]
FEATURE_UNITS = ["degree"] * 5 + ["percent"] + ["degree"] * 5 + ["percent"]
CAMERAS = [
    "observation.images.central",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
]
ACTION_SEMANTICS = "absolute joint target applied at that control frame"
CLEAN_SOURCE = "collection.clean_layout_filter_v1"
READY_GATES = (
    "source_protection",
    "layout_geometry_ik",
    "episodes_replay_and_success",
    "family_split",
    "normalization_and_inference_consistency",
    "pretrained_weight_loading",
    "parameter_tree_and_lora",
    "gradients_and_parameter_delta",
    "multi_gpu_compute_and_sharding",
    "full_update_peak_and_throughput",
    "checkpoint_full_restore_continuation",
    "configuration_and_launch_gate",
    "security_and_targeted_tests",
)


def training_config_problems(config: dict,
                             expected_data: dict | None = None) -> list[str]:
    """Check the immutable formal-run settings before command construction."""
    problems: list[str] = []
    registered = expected_data or {"view_episodes": 60, "train": 48, "validation": 12,
                                   "clean": 60, "recovery": 0}
    initial = config.get("initialisation") or {}
    optimisation = config.get("optimisation") or {}
    checkpointing = config.get("checkpointing") or {}
    expected_initial = {
        "resume_previous_so101_checkpoint": False,
        "state_mode": "on",
        "internal_action_dim": 32,
        "state_action_dim": 12,
        "action_horizon": 10,
        "paligemma_variant": "gemma_2b_lora",
        "action_expert_variant": "gemma_300m",
    }
    for key, value in expected_initial.items():
        if initial.get(key) != value:
            problems.append(f"initialisation.{key} is not {value!r}")
    if initial.get("base_checkpoint") not in {"pi05_libero", "pi05_base"}:
        problems.append("initialisation.base_checkpoint is not a registered pi05 base")
    expected_optimisation = {
        "seed": 42,
        "primary_batch_size": 8,
        "planned_updates": 25000,
        "warmup_steps": 500,
        "peak_lr": 5e-5,
        "decay_steps": 25000,
        "decay_lr": 5e-6,
        "optimizer": "AdamW",
        "gradient_clip_norm": 1.0,
        "dtype": "bfloat16",
    }
    for key, value in expected_optimisation.items():
        if optimisation.get(key) != value:
            problems.append(f"optimisation.{key} is not {value!r}")
    if optimisation.get("ema_decay") is not None:
        problems.append("optimisation.ema_decay must be absent or null")
    if "fallback_batch_sizes" in optimisation:
        problems.append("batch fallback field is forbidden for the formal run")
    if optimisation.get("allow_batch_fallback") is not False:
        problems.append("batch fallback must be explicitly disabled")
    if optimisation.get("allow_update_override") is not False:
        problems.append("formal update override must be explicitly disabled")
    save_interval = checkpointing.get("save_interval")
    if save_interval not in {3000, 5000}:
        problems.append("checkpointing.save_interval is not 3000 or 5000")
    if checkpointing.get("step_axis") != "TrainState.step":
        problems.append("checkpointing.step_axis is not TrainState.step")
    expected_evals = list(range(0, 25000, save_interval)) + [25000] if save_interval in {3000, 5000} else []
    if checkpointing.get("eval_checkpoints") != expected_evals:
        problems.append("checkpointing.eval_checkpoints do not match the save interval")
    data = config.get("data") or {}
    for key, value in registered.items():
        if data.get(key) != value:
            problems.append(f"data.{key} is not {value}")
    if data.get("perturbations_enabled") is not False:
        problems.append("data.perturbations_enabled must be false")
    runtime = config.get("runtime_binding") or {}
    image = runtime.get("image")
    if not isinstance(image, str) or "@sha256:" not in image:
        problems.append("runtime_binding.image is not pinned by digest")
    for key in ("openpi_source_sha256", "base_checkpoint_sha256",
                "sentencepiece_asset_sha256"):
        if not _hex(runtime.get(key)):
            problems.append(f"runtime_binding.{key} is not a SHA-256")
    return problems


def sha256_json(value: Any) -> str:
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Stable hash of a file or directory using relative paths and file bytes."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"tree contains no files: {path.name}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _hex(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def view_contract_problems(
    meta: dict,
    *,
    protocol_binding: dict | None = None,
    expected_counts: tuple[int, int, int] | None = None,
    require_clean: bool = True,
) -> list[str]:
    """Return every contract violation without inferring a missing category."""
    problems: list[str] = []
    rows = meta.get("episodes")
    if not isinstance(rows, list) or not rows:
        return ["meta.episodes must be a non-empty list"]
    if require_clean and int(meta.get("schema_version", 0)) < 2:
        problems.append("clean view requires metadata schema_version >= 2")
    actual = (
        len(rows),
        sum(row.get("split") == "train" for row in rows),
        sum(row.get("split") == "validation" for row in rows),
    )
    if expected_counts is not None and actual != tuple(expected_counts):
        problems.append(f"episode counts {actual} differ from expected {tuple(expected_counts)}")
    if len({row.get("split") for row in rows}) != 2:
        problems.append("view must contain exactly the registered two splits")

    if meta.get("state_action_dim") != 12:
        problems.append("state_action_dim must be 12")
    if meta.get("feature_names") != FEATURE_NAMES:
        problems.append("feature_names differ from the left-6/right-6 contract")
    if meta.get("feature_units") != FEATURE_UNITS:
        problems.append("feature_units differ from degree/degree/.../percent contract")
    if meta.get("cameras") != CAMERAS:
        problems.append("cameras differ from the three registered RGB streams")
    if meta.get("action_semantics") != ACTION_SEMANTICS:
        problems.append("action_semantics does not name the applied current-frame target")

    clean_profile = meta.get("clean_profile")
    if require_clean:
        if not isinstance(clean_profile, dict):
            problems.append("clean view has no explicit clean_profile provenance")
        else:
            expected = {
                "category": "clean",
                "source": CLEAN_SOURCE,
                "perturbation_free": True,
                "episodes_bound": len(rows),
            }
            for key, value in expected.items():
                if clean_profile.get(key) != value:
                    problems.append(f"clean_profile.{key} is not {value!r}")
            if not _hex(clean_profile.get("protocol_binding_sha256")):
                problems.append("clean_profile lacks a protocol binding hash")

    seeds, scenes = [], []
    for index, row in enumerate(rows):
        prefix = f"episode row {index}"
        if require_clean:
            if row.get("category") != "clean":
                problems.append(f"{prefix}: category is not explicitly clean")
            if row.get("subgroup") != "clean":
                problems.append(f"{prefix}: subgroup is not explicitly clean")
            if "perturbation_frame" not in row:
                problems.append(f"{prefix}: perturbation_frame field is absent")
            elif row["perturbation_frame"] is not None:
                problems.append(f"{prefix}: perturbation_frame is not null")
        if row.get("max_state_error_vs_trace") != 0.0:
            problems.append(f"{prefix}: physical replay state error is not zero")
        if row.get("max_action_error_vs_trace") != 0.0:
            problems.append(f"{prefix}: physical replay action error is not zero")
        frames = row.get("frames")
        if not isinstance(frames, int) or frames < 10:
            problems.append(f"{prefix}: fewer than one 10-frame action chunk")
        if row.get("state_shape") != [frames, 12] or row.get("action_shape") != [frames, 12]:
            problems.append(f"{prefix}: state/action shapes are not aligned [frames,12]")
        if row.get("image_shape") != [480, 640, 3]:
            problems.append(f"{prefix}: image_shape is not [480,640,3]")
        seeds.append(row.get("seed"))
        scenes.append(row.get("scene_sha256"))
    if len(set(seeds)) != len(rows):
        problems.append("seeds are not unique")
    if len(set(scenes)) != len(rows) or not all(_hex(value) for value in scenes):
        problems.append("scene hashes are not 60 unique SHA-256 identities")

    if protocol_binding is not None:
        if not protocol_binding.get("passed"):
            problems.append("protocol_binding.passed is not true")
        if protocol_binding.get("episodes_bound") != len(rows):
            problems.append("protocol binding does not cover every episode")
        checks = protocol_binding.get("checks") or {}
        if checks.get("object_counts") != [4]:
            problems.append("protocol binding does not certify four objects per scene")
        if checks.get("routes") != ["direct"]:
            problems.append("protocol binding does not certify direct sorting only")
        if checks.get("perturbation_free") is not True:
            problems.append("protocol binding does not certify perturbation-free scenes")
        distances = checks.get("minimum_pairwise_distance_m") or {}
        if distances.get("episodes_below") != 0:
            problems.append("protocol binding does not certify the spacing threshold")
        if not isinstance(clean_profile, dict) or clean_profile.get("protocol_binding_sha256") != sha256_json(
                protocol_binding):
            problems.append("clean_profile protocol binding hash mismatch")
    return problems


def admission_contract_problems(
    admission: dict,
    *,
    meta: dict,
    split: dict,
    train_stats_sha256: str,
) -> list[str]:
    problems: list[str] = []
    rows = meta.get("episodes", [])
    if not admission.get("passed") or not admission.get("data_gate", {}).get("passed"):
        problems.append("data admission gate did not pass")
    if admission.get("data_gate", {}).get("clean_source") != CLEAN_SOURCE:
        problems.append("data admission does not bind the explicit clean source")
    totals = admission.get("totals") or {}
    expected = {
        "episodes": len(rows),
        "train": sum(row.get("split") == "train" for row in rows),
        "validation": sum(row.get("split") == "validation" for row in rows),
        "clean": len(rows),
        "recovery": 0,
    }
    for key, value in expected.items():
        if totals.get(key) != value:
            problems.append(f"admission totals.{key} is not {value}")
    expected_hashes = {
        "view_sha256": meta.get("view_sha256"),
        "split_sha256": sha256_json(split),
        "train_stats_sha256": train_stats_sha256,
    }
    for key, value in expected_hashes.items():
        if admission.get(key) != value:
            problems.append(f"admission {key} mismatch")
    return problems


def norm_audit_problems(
    audit: dict,
    *,
    view_sha256: str,
    norm_stats_sha256: str,
    base_checkpoint_sha256: str,
    training_config_sha256: str,
    expected_train: int = 48,
    expected_validation: int = 12,
) -> list[str]:
    problems: list[str] = []
    if int(audit.get("schema_version", 0)) < 3:
        problems.append("normalization audit schema_version must be >= 3")
    if not audit.get("passed"):
        problems.append("normalization audit did not pass")
    bindings = {
        "view_sha256": view_sha256,
        "norm_stats_sha256": norm_stats_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "training_config_sha256": training_config_sha256,
    }
    for key, value in bindings.items():
        if audit.get(key) != value:
            problems.append(f"normalization audit {key} mismatch")
    binding = audit.get("binding") or {}
    if (binding.get("train_episodes"), binding.get("validation_episodes")) != (
            expected_train, expected_validation):
        problems.append(
            f"normalization audit does not bind the {expected_train}/{expected_validation} split")
    statistics = audit.get("statistics") or {}
    if statistics.get("recomputed_from_train_only_matches_view") is not True:
        problems.append("normalization statistics are not train-only and exact")
    if statistics.get("all_values_finite") is not True:
        problems.append("normalization statistics contain non-finite values")
    if statistics.get("near_constant_dims_std_lt_1e-6") != {"state": [], "actions": []}:
        problems.append("state/action statistics contain near-constant dimensions")
    transform = audit.get("transform") or {}
    if transform.get("internal_action_dim") != 32:
        problems.append("runtime internal action dimension is not 32")
    if float(transform.get("worst_recovery_error", float("inf"))) > 1e-4:
        problems.append("action normalization round-trip error exceeds 1e-4")
    for field in ("pad_first_12_max_abs_error", "padded_tail_max_abs"):
        if transform.get(field) != 0.0:
            problems.append(f"12->32 action padding check {field} is not zero")
    if transform.get("clipping_applied") is not False:
        problems.append("normalization audit applied clipping")
    if transform.get("inference_output_execution_order_verified") is not True:
        problems.append("inference output transforms were not executed in runtime order")
    loader = audit.get("loader_consistency") or {}
    if loader.get("episodes_covered") != expected_train:
        problems.append(
            f"real loader comparison did not cover all {expected_train} train episodes")
    if float(loader.get("action_max_abs_error_vs_runtime_pipeline", float("inf"))) > 1e-6:
        problems.append("real loader actions differ from the runtime transform pipeline")
    if float(loader.get("state_max_abs_error_vs_runtime_pipeline", float("inf"))) > 1e-6:
        problems.append("real loader state differs from the runtime transform pipeline")
    semantics = audit.get("semantics") or {}
    if semantics.get("feature_units") != FEATURE_UNITS:
        problems.append("normalization audit units differ from the 12-D contract")
    if semantics.get("action_horizon") != 10 or semantics.get("action_shape") != [10, 12]:
        problems.append("normalization audit does not bind a 10x12 action chunk")
    return problems


def scene_audit_problems(audit: dict, *, view_sha256: str,
                         expected_counts: tuple[int, int, int] = (60, 48, 12),
                         expected_attempts: tuple[int, int, int] = (175, 60, 115),
                         scene_key: str = "all_60_scenes_reconstructed_hash_exact",
                         ) -> list[str]:
    problems = []
    if not audit.get("passed") or audit.get("training_view_sha256") != view_sha256:
        problems.append("scene audit failed or is bound to another view")
    summary = audit.get("summary") or {}
    expected_true = (
        scene_key,
        "all_four_objects_per_episode",
        "all_object_ik_records_complete",
        "all_region_keepout_checks_passed",
        "all_original_object_predicates_passed",
        "all_physical_success_rows",
        "all_physical_replay_exact",
    )
    for key in expected_true:
        if summary.get(key) is not True:
            problems.append(f"scene audit {key} is not true")
    total, train, validation = expected_counts
    if (summary.get("episodes"), summary.get("train"), summary.get("validation")) != (
            total, train, validation):
        problems.append(f"scene audit does not cover the {total}/{train}/{validation} split")
    if summary.get("unique_scene_hashes") != total:
        problems.append(f"scene audit does not bind {total} unique scene hashes")
    if summary.get("near_duplicate_families_crossing_split"):
        problems.append("scene families cross the train/validation split")
    if (summary.get("attempts"), summary.get("successes"), summary.get("failures")) != tuple(
            expected_attempts):
        problems.append("scene audit attempt accounting is not "
                        f"{expected_attempts[0]}/{expected_attempts[1]}/{expected_attempts[2]}")
    return problems


def data_audit_problems(audit: dict, *, view_sha256: str,
                        expected_counts: tuple[int, int, int] = (60, 48, 12),
                        expected_attempts: tuple[int, int, int] = (175, 60, 115),
                        ) -> list[str]:
    problems = []
    if not audit.get("passed") or audit.get("view_sha256") != view_sha256:
        problems.append("data audit failed or is bound to another view")
    attempts = audit.get("attempts") or {}
    if (attempts.get("total"), attempts.get("successes"), attempts.get("failures")) != tuple(
            expected_attempts):
        problems.append("data audit attempt accounting is not "
                        f"{expected_attempts[0]}/{expected_attempts[1]}/{expected_attempts[2]}")
    rows = audit.get("episodes") or []
    total, train, validation = expected_counts
    if (len(rows), sum(row.get("split") == "train" for row in rows),
            sum(row.get("split") == "validation" for row in rows)) != (total, train, validation):
        problems.append(f"data audit does not cover {total} episodes with {train}/{validation}")
    if not all(row.get("state_action_trace_exact") for row in rows):
        problems.append("data audit did not prove state/action trace alignment")
    if not all(row.get("strict_predicate_passed") for row in rows):
        problems.append("data audit did not prove every original strict predicate")
    if audit.get("images_decoded") is not True:
        problems.append("data audit did not decode all three camera streams")
    if audit.get("actions_reapplied") is not True:
        problems.append("data audit did not re-apply every action")
    return problems


def stats_recompute_problems(report: dict, *, norm_stats_sha256: str,
                             view_sha256: str,
                             expected_train_ids: list[int] | None = None,
                             expected_validation_ids: list[int] | None = None,
                             ) -> list[str]:
    problems = []
    if not report.get("passed") or report.get("view_sha256") != view_sha256:
        problems.append("statistics recomputation is failed or bound to another view")
    if report.get("new_norm_stats_sha256") != norm_stats_sha256:
        problems.append("recomputed statistics hash differs from norm_stats")
    train_ids = expected_train_ids if expected_train_ids is not None else list(range(48))
    validation_ids = (expected_validation_ids if expected_validation_ids is not None
                      else list(range(48, 60)))
    if report.get("train_episodes") != train_ids:
        problems.append("statistics were not computed from the registered train episodes only")
    if report.get("validation_episodes_excluded") != validation_ids:
        problems.append("registered validation episodes were not explicitly excluded")
    near = report.get("near_constant_dimensions_std_lt_1e-6") or {}
    if near.get("state") != [] or near.get("actions") != []:
        problems.append("recomputed statistics contain near-constant dimensions")
    return problems


def ready_manifest_problems(
    ready: dict,
    *,
    view_sha256: str,
    norm_stats_sha256: str,
    norm_audit_sha256: str,
    training_config_sha256: str,
    base_checkpoint_sha256: str,
    scene_audit_sha256: str | None = None,
    data_audit_sha256: str | None = None,
    stats_recompute_sha256: str | None = None,
) -> list[str]:
    problems: list[str] = []
    if ready.get("status") != "READY" or ready.get("mode") != "PREPARE_ONLY":
        problems.append("ready manifest is not a PREPARE_ONLY READY decision")
    if ready.get("formal_training_performed") is not False:
        problems.append("ready manifest does not certify that formal training is still unstarted")
    gates = ready.get("gates") or {}
    for name in READY_GATES:
        if gates.get(name, {}).get("passed") is not True:
            problems.append(f"ready gate {name} is not true")
    bindings = ready.get("bindings") or {}
    expected = {
        "view_sha256": view_sha256,
        "norm_stats_sha256": norm_stats_sha256,
        "norm_audit_sha256": norm_audit_sha256,
        "training_config_sha256": training_config_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
    }
    if scene_audit_sha256 is not None:
        expected["scene_audit_sha256"] = scene_audit_sha256
    if data_audit_sha256 is not None:
        expected["data_audit_sha256"] = data_audit_sha256
    if stats_recompute_sha256 is not None:
        expected["stats_recompute_sha256"] = stats_recompute_sha256
    for key, value in expected.items():
        if bindings.get(key) != value:
            problems.append(f"ready binding {key} mismatch")
    return problems


def tree_structure_problems(
    reference: dict[str, dict],
    other: dict[str, dict],
) -> list[str]:
    """Compare parameter path/shape/dtype metadata without importing JAX."""
    problems = []
    missing = sorted(set(reference) - set(other))
    extra = sorted(set(other) - set(reference))
    if missing:
        problems.append(f"{len(missing)} parameter paths are missing")
    if extra:
        problems.append(f"{len(extra)} unexpected parameter paths are present")
    for name in sorted(set(reference) & set(other)):
        if tuple(reference[name].get("shape", ())) != tuple(other[name].get("shape", ())):
            problems.append(f"{name}: parameter shape mismatch")
        if reference[name].get("dtype") != other[name].get("dtype"):
            problems.append(f"{name}: parameter dtype mismatch")
    return problems
