"""Pure, fail-closed checks for the Scene1 PI-FAST v2 plan."""
from __future__ import annotations

import math

IMAGE_DIGEST = (
    "openpi-server@sha256:"
    "405af52051682d5e13169d1de1523fd0adc116eceae4fc314cadfacd45b1ad52"
)


def plan_problems(plan: dict, *, formal: bool) -> list[str]:
    problems: list[str] = []
    status = plan.get("status")
    allowed = {"draft_requires_audit", "ready_for_training", "running"}
    if status not in allowed:
        problems.append(f"unexpected status {status!r}")
    if formal and (
            status != "ready_for_training"
            or plan.get("execution_authorized") is not True):
        problems.append("formal execution requires ready_for_training and authorization")
    if plan.get("plan_id") != "scene1-pifast-ki-v2":
        problems.append("plan_id is not scene1-pifast-ki-v2")
    if plan.get("supersedes") != (
            "configs/so101/scene1_pifast_ki_training_v1.json"):
        problems.append("v2 does not explicitly supersede v1")
    runtime = plan.get("runtime") or {}
    expected_runtime = {
        "image": IMAGE_DIGEST,
        "single_process": True,
    }
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            problems.append(f"runtime.{key} is not {expected!r}")
    device_count = runtime.get("device_count")
    fsdp_devices = runtime.get("fsdp_devices")
    if device_count not in {2, 4} or fsdp_devices not in {2, 4}:
        problems.append("runtime must use one process with 2 or 4 FSDP devices")
    elif (
            device_count % fsdp_devices
            or runtime.get("data_parallel_replicas")
            != device_count // fsdp_devices):
        problems.append("runtime data-parallel replica count is inconsistent")

    initialization = plan.get("initialization") or {}
    if not (initialization.get("pi05_base") or {}).get("tree_sha256"):
        problems.append("pi05_base tree hash is missing")
    if (initialization.get("pi0_fast_base") or {}).get("role") != "comparison_only":
        problems.append("pi0_fast_base is not limited to comparison")
    if initialization.get("resume_existing_checkpoint") is not False:
        problems.append("checkpoint resume is not disabled for initialization")
    fast_adapter = initialization.get("fast_adapter") or {}
    if fast_adapter.get("shape") != [2048, 2048]:
        problems.append("FAST residual is not [2048,2048]")
    if fast_adapter.get("initialization") != "zero":
        problems.append("FAST residual is not zero-initialized")

    scope = plan.get("model_scope") or {}
    if scope.get("language") != {
            "backbone": "gemma_2b",
            "non_lora": "frozen",
            "attention_lora": True,
            "mlp_lora": True,
            "rank": 16,
            "alpha": 16.0}:
        problems.append("language LoRA/freeze scope differs")
    expert = scope.get("action_expert") or {}
    for key, expected in (
        ("backbone", "gemma_300m"),
        ("non_lora", "frozen"),
        ("attention_lora", True),
        ("mlp_lora", True),
        ("rank", 64),
        ("alpha", 64.0),
    ):
        if expert.get(key) != expected:
            problems.append(f"model_scope.action_expert.{key} is not {expected!r}")
    if scope.get("vision_encoder") != "full_trainable":
        problems.append("vision encoder is not fully trainable")
    if (scope.get("tied_input_embedding") or {}).get("state") != "frozen":
        problems.append("tied input embedding is not frozen")
    projections = scope.get("projections") or {}
    for name in (
        "action_in_proj", "action_out_proj",
        "time_mlp_in", "time_mlp_out",
    ):
        if projections.get(name) != "kernel_and_bias_trainable":
            problems.append(f"{name} kernel and bias are not fully trainable")

    data = plan.get("data") or {}
    for key, expected in (
        ("episodes", 64),
        ("train_episodes", 52),
        ("validation_episodes", 12),
        ("train_red", 26),
        ("train_blue", 26),
        ("validation_red", 6),
        ("validation_blue", 6),
        ("family_crosses_split", 0),
    ):
        if data.get(key) != expected:
            problems.append(f"data.{key} is not {expected!r}")
    interface = plan.get("interface") or {}
    for key, expected in (
        ("state_dim", 12),
        ("action_dim", 12),
        ("internal_action_dim", 32),
        ("action_horizon", 16),
        ("n_action_steps", 16),
        ("state_mode", "on"),
        ("delta_transform", False),
        ("flow_loss_dimensions", 12),
        ("padded_dimensions_in_flow_loss", False),
        ("state_text_dimensions", 12),
        ("fast_action_dimensions", 12),
        ("inference_output_dimensions", 12),
    ):
        if interface.get(key) != expected:
            problems.append(f"interface.{key} is not {expected!r}")
    norm = plan.get("normalization") or {}
    for key, expected in (
        ("method", "quantile"),
        ("grouping", "combined_train_red_blue_single_group"),
        ("state_and_action_separate", True),
        ("fit_split", "train_only"),
        ("frame_level_unique_records", True),
        ("overlapping_chunks_reused_for_stats", False),
        ("validation_and_inference_use_train_stats", True),
        ("red_blue_use_same_stats", True),
    ):
        if norm.get(key) != expected:
            problems.append(f"normalization.{key} is not {expected!r}")

    fast = plan.get("fast") or {}
    for key, expected in (
        ("processor_commit", "ec4d7aa71691cac0b8bed6942be45684db2110f4"),
        ("vocabulary_size", 2048),
        ("dct_scale", 10),
        ("min_token", -354),
        ("padded_language_vocabulary", 257152),
        ("language_pad_id", 0),
        ("language_eos_id", 1),
        ("max_token_len", 128),
        ("truncation", "forbidden_fail_closed"),
        ("real_horizon_16_length_audit", "pass"),
    ):
        if fast.get(key) != expected:
            problems.append(f"fast.{key} is not {expected!r}")

    loss = plan.get("loss") or {}
    if loss.get("flow_weight") != 1.0:
        problems.append("Flow loss weight is not 1.0")
    weight = loss.get("fast_ce_weight")
    if formal and (
            not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0):
        problems.append("formal fast_ce_weight is not frozen to a positive value")
    if loss.get("implicit_default_allowed") is not False:
        problems.append("implicit FAST loss weight remains allowed")

    optimization = plan.get("optimization") or {}
    for key, expected in (
        ("seed", 42),
        ("optimizer", "AdamW"),
        ("beta1", 0.9),
        ("beta2", 0.95),
        ("epsilon", 1e-8),
        ("weight_decay", 1e-10),
        ("global_gradient_clip_norm", 1.0),
        ("gradient_accumulation_steps", 1),
        ("optimizer_updates", 40000),
        ("warmup_updates", 1000),
        ("peak_lr", 2.5e-5),
        ("cosine_decay_updates", 40000),
        ("final_lr", 2.5e-6),
        ("ema_enabled", False),
        ("compute_dtype", "bfloat16"),
        ("data_loader_workers", 8),
    ):
        if optimization.get(key) != expected:
            problems.append(f"optimization.{key} is not {expected!r}")
    batch = optimization.get("global_batch_size")
    if not isinstance(batch, int) or batch < 4 or batch % 4:
        problems.append("selected global batch is not a tested multiple of 4 at least 4")
    elif (
            isinstance(device_count, int)
            and batch % device_count
            or isinstance(device_count, int)
            and optimization.get("per_device_batch") != batch // device_count):
        problems.append("global/per-device batch geometry is inconsistent")

    checkpointing = plan.get("checkpointing") or {}
    if checkpointing.get("save_steps") != list(range(2000, 40001, 2000)):
        problems.append("checkpoint schedule is not 2000..40000")
    if checkpointing.get("full_train_state") is not True:
        problems.append("checkpoint does not save the full TrainState")
    if checkpointing.get("retain_all") is not True:
        problems.append("checkpoint retention is not enabled")

    validation = plan.get("validation") or {}
    for key, expected in (
        ("quick_probe_every_updates", 50),
        ("quick_probe_samples", 256),
        ("full_samples", 8228),
        ("full_batch_size", 16),
        ("full_batches", 515),
        ("full_last_batch_valid_samples", 4),
        ("full_padding_rows", 12),
        ("train_false", True),
        ("gradient_updates", 0),
        ("fixed_flow_time_and_noise", True),
    ):
        if validation.get(key) != expected:
            problems.append(f"validation.{key} is not {expected!r}")

    resource = plan.get("resource") or {}
    if resource.get("minimum_memory_margin_mib") != 100:
        problems.append("minimum GPU memory margin is not 100 MiB")
    if formal and not (
            isinstance(resource.get("wall_limit_s"), int)
            and resource["wall_limit_s"] > 0):
        problems.append("formal wall-clock limit is missing")
    return problems
