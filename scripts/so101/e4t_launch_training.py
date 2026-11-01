#!/usr/bin/env python3
"""Launch the frozen E4-T run from configs/so101/e4t_training.json.

Every hyper-parameter comes from the protocol file so the training command cannot drift from what
was registered, and the private base-checkpoint and tokenizer paths are read from the deploy config
without ever being printed.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "deploy"))
sys.path.insert(0, str(REPO / "scripts" / "so101"))
import common as deploy  # noqa: E402
import so101_common as common  # noqa: E402
import clean60_contracts as clean60  # noqa: E402
from model_scope import resolve_model_scope  # noqa: E402


def model_scope_cli_args(protocol: dict) -> list[str]:
    """Translate registered architecture and freeze scope into runner flags."""
    scope = resolve_model_scope(protocol)
    freeze_flag = ("--freeze-vision-encoder" if scope["freeze_vision_encoder"]
                   else "--no-freeze-vision-encoder")
    return ["--paligemma-variant", scope["paligemma_variant"],
            "--action-expert-variant", scope["action_expert_variant"], freeze_flag]


def default_fsdp_devices(protocol: dict, selected_count: int) -> int:
    """Preserve the legacy single-FSDP default; use registered clean60 topology."""
    selected = protocol.get("devices", {}).get("selected", {})
    if isinstance(selected, dict) and selected.get("fsdp_devices") is not None:
        return int(selected["fsdp_devices"])
    if "devices" in protocol:
        return selected_count
    return 1


def validate_view_artifacts(view: Path, protocol: dict, *,
                            base_checkpoint_sha256: str | None = None,
                            training_config_sha256: str | None = None) -> dict:
    """Validate explicit clean provenance and all frozen data-side bindings."""
    admission_path = view / "e4t_admission.json"
    plan_path = view / "e4t_sampling_plan.json"
    binding_path = view / "protocol_binding.json"
    split_path = view / "splits.json"
    stats_path = view / "assets/so101/norm_stats.json"
    train_stats_path = view / "train_stats.json"
    scene_audit_path = view / "scene_audit.json"
    data_audit_path = view / "data_audit.json"
    stats_recompute_path = view / "stats_recompute.json"
    for path in (admission_path, plan_path, binding_path, split_path, stats_path,
                 train_stats_path, scene_audit_path, data_audit_path,
                 stats_recompute_path, view / "meta.json"):
        if not path.is_file():
            raise SystemExit(f"missing frozen artifact: {path.name}")
    meta = common.load_json(view / "meta.json")
    binding = common.load_json(binding_path)
    split = common.load_json(split_path)
    expected = protocol["data"]
    problems = clean60.view_contract_problems(
        meta, protocol_binding=binding,
        expected_counts=(expected["view_episodes"], expected["train"],
                         expected["validation"]),
        require_clean=True)
    problems.extend(clean60.admission_contract_problems(
        common.load_json(admission_path), meta=meta, split=split,
        train_stats_sha256=common.sha256_file(train_stats_path)))
    norm_stats_sha256 = common.sha256_file(stats_path)
    problems.extend(clean60.scene_audit_problems(
        common.load_json(scene_audit_path), view_sha256=meta["view_sha256"]))
    problems.extend(clean60.data_audit_problems(
        common.load_json(data_audit_path), view_sha256=meta["view_sha256"]))
    problems.extend(clean60.stats_recompute_problems(
        common.load_json(stats_recompute_path), view_sha256=meta["view_sha256"],
        norm_stats_sha256=norm_stats_sha256))
    measured_tree = common.sha256_tree(view / "episodes")["sha256"]
    if measured_tree != meta.get("view_sha256"):
        problems.append("episode tree hash differs from meta.view_sha256")
    plan = common.load_json(plan_path)
    if plan.get("view_sha256") != meta.get("view_sha256"):
        problems.append("sampling plan is bound to another view")
    if plan.get("train_groups_present") != ["clean"] or set(plan.get("group_weights", {})) != {"clean"}:
        problems.append("sampling plan is not the registered clean-only plan")
    if problems:
        raise SystemExit("clean60 view contract failed: " + "; ".join(problems))
    if protocol.get("run", {}).get("require_action_norm_audit"):
        audit_path = view / "action_norm_audit.json"
        if not audit_path.is_file():
            raise SystemExit("action normalization audit is missing")
        if not base_checkpoint_sha256 or not training_config_sha256:
            raise SystemExit("normalization audit validation needs config and base hashes")
        problems = clean60.norm_audit_problems(
            common.load_json(audit_path),
            view_sha256=meta["view_sha256"],
            norm_stats_sha256=norm_stats_sha256,
            base_checkpoint_sha256=base_checkpoint_sha256,
            training_config_sha256=training_config_sha256)
        if problems:
            raise SystemExit("action normalization audit failed: " + "; ".join(problems))
    return {
        "meta": meta, "binding": binding, "split": split, "plan": plan,
        "admission": common.load_json(admission_path),
        "sha256": {
            "scene_audit": common.sha256_file(scene_audit_path),
            "data_audit": common.sha256_file(data_audit_path),
            "stats_recompute": common.sha256_file(stats_recompute_path),
            "norm_stats": norm_stats_sha256,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True, help="external python for the launcher")
    parser.add_argument("--gpu-indices", required=True,
                        help="comma-separated device ids measured to fit this run (2 or 4 cards)")
    parser.add_argument("--required-memory-mib", type=int, default=None,
                        help="measured per-card demand, excluding the safety margin")
    parser.add_argument("--memory-margin-mib", type=int, default=None)
    parser.add_argument("--fsdp-devices", type=int, default=None,
                        help="defaults to the number of selected devices")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-directory", type=Path, default=None,
                        help="unique ignored output directory for this run")
    parser.add_argument("--steps", type=int, default=None, help="override the registered update budget")
    parser.add_argument("--training-config", type=Path,
                        default=REPO / "configs/so101/clean60_training.json")
    parser.add_argument("--ready-manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    training_config = args.training_config.expanduser().resolve()
    protocol = common.load_json(training_config)
    config_problems = clean60.training_config_problems(protocol)
    if config_problems:
        raise SystemExit("registered clean60 configuration failed: "
                         + "; ".join(config_problems))
    optimisation = protocol["optimisation"]
    assets = deploy.load_assets()
    config_sha256 = common.sha256_file(training_config)
    base_sha256 = clean60.sha256_tree(assets["checkpoint"])
    runtime_binding = protocol.get("runtime_binding") or {}
    if base_sha256 != runtime_binding.get("base_checkpoint_sha256"):
        raise SystemExit("base checkpoint differs from the registered runtime binding")
    tokenizer_sha256 = common.sha256_file(assets["tokenizer"])
    if tokenizer_sha256 != runtime_binding.get("sentencepiece_asset_sha256"):
        raise SystemExit("tokenizer differs from the registered runtime binding")
    view = args.view.expanduser().resolve()
    artifacts = validate_view_artifacts(
        view, protocol, base_checkpoint_sha256=base_sha256,
        training_config_sha256=config_sha256)
    admission = view / "e4t_admission.json"
    plan = view / "e4t_sampling_plan.json"
    meta = artifacts["meta"]
    expected = protocol["data"]
    rows = meta["episodes"]
    actual = (len(rows), sum(row["split"] == "train" for row in rows),
              sum(row["split"] == "validation" for row in rows))
    if actual != (expected["view_episodes"], expected["train"], expected["validation"]):
        raise SystemExit(f"training view counts {actual} differ from the selected profile")
    run_settings = protocol.get("run", {})
    if run_settings.get("require_action_norm_audit"):
        audit = common.load_json(view / "action_norm_audit.json")
        if not audit.get("passed") or audit.get("norm_stats_sha256") != common.sha256_file(
                view / "assets/so101/norm_stats.json") or audit.get("view_sha256") != meta["view_sha256"]:
            raise SystemExit("action normalization audit is missing, failed or bound to another view")
    if args.steps is not None:
        raise SystemExit("step override is forbidden for the formal clean60 run")
    steps = optimisation["planned_updates"]
    if steps <= 0:
        raise SystemExit("update budget must be positive")
    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-clean60-25k-" + uuid.uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", run_id):
        raise SystemExit("run id may contain only letters, digits, dot, underscore and dash")
    base_run_dir = REPO / run_settings.get("directory", "runs/so101/e4t")
    run_dir = (args.run_directory.expanduser().resolve() if args.run_directory else
               (base_run_dir / run_id).resolve())
    if not run_dir.is_relative_to((REPO / "runs").resolve()):
        raise SystemExit("run directory must remain inside the ignored runs directory")
    if (run_dir / "config_initial.json").exists():
        raise SystemExit(f"refusing to reuse a run directory with an initial snapshot: {run_dir.name}")
    resource_policy = protocol.get("resource_policy", {})
    demand = (args.required_memory_mib if args.required_memory_mib is not None else
              protocol["resource_policy"]["measured_per_card_demand_mib"])
    margin = (args.memory_margin_mib if args.memory_margin_mib is not None else
              resource_policy.get("memory_margin_mib", 1500))
    optimisation_args = protocol["optimisation"]
    experiment_id = f"{run_settings['exp_name']}_{run_id}"
    argv = [str(args.python), str(REPO / "scripts/so101/e4t_run_training.py"),
            "--mode", "initial", "--run-dir", str(run_dir),
            "--train-view", str(view), "--base-checkpoint", assets["checkpoint"],
            "--tokenizer", assets["tokenizer"], "--exp-name", experiment_id,
            "--image", str(runtime_binding["image"]),
            "--expected-tokenizer-sha256", tokenizer_sha256,
            "--state-mode", protocol["initialisation"]["state_mode"],
            "--gpu-indices", args.gpu_indices,
            "--fsdp-devices", str(args.fsdp_devices if args.fsdp_devices is not None else
                default_fsdp_devices(protocol, len(
                    [value for value in args.gpu_indices.split(",") if value.strip()]))),
            *model_scope_cli_args(protocol),
            "--expected-groups", str(len(protocol["sampling_plan"]["group_weights"])),
            "--required-memory-mib", str(demand),
            "--memory-margin-mib", str(margin),
            "--no-jax-preallocate" if not resource_policy.get("jax_preallocate", False)
            else "--jax-preallocate",
            "--batch-size", str(optimisation["primary_batch_size"]),
            "--steps", str(steps),
            "--save-interval", str(protocol["checkpointing"]["save_interval"]),
            "--data-loader-workers", str(optimisation["data_loader_workers"]),
            "--lr-warmup-steps", str(optimisation["warmup_steps"]),
            "--peak-lr", str(optimisation["peak_lr"]),
            "--decay-steps", str(steps if args.steps is not None else optimisation["decay_steps"]),
            "--decay-lr", str(optimisation["decay_lr"]),
            "--seed", str(optimisation["seed"]),
            "--action-horizon", str(protocol["initialisation"]["action_horizon"]),
            "--optimizer", str(optimisation["optimizer"]),
            "--gradient-clip-norm", str(optimisation["gradient_clip_norm"]),
            "--dtype", str(optimisation["dtype"]),
            "--sampling-plan", str(plan), "--data-admission", str(admission),
            "--training-config", str(training_config),
            "--training-config-sha256", config_sha256,
            "--require-clean-data",
            "--require-action-norm-audit",
            "--registered-base-sha256", base_sha256,
            "--dataset-label", str(view.name),
            "--expected-total-episodes", str(expected["view_episodes"]),
            "--expected-train-episodes", str(expected["train"]),
            "--expected-validation-episodes", str(expected["validation"]),
            "--timeout-seconds", str(optimisation["training_wall_limit_s"]),
            "--run-id", run_id]
    measured_fraction = resource_policy.get("measured_jax_memory_fraction")
    if measured_fraction is not None:
        argv += ["--jax-memory-fraction", str(measured_fraction)]
    if not args.dry_run:
        if not args.ready_manifest or not args.ready_manifest.is_file():
            raise SystemExit("formal clean60 launch requires --ready-manifest")
        ready = common.load_json(args.ready_manifest)
        problems = clean60.ready_manifest_problems(
            ready,
            view_sha256=meta["view_sha256"],
            norm_stats_sha256=common.sha256_file(view / "assets/so101/norm_stats.json"),
            norm_audit_sha256=common.sha256_file(view / "action_norm_audit.json"),
            training_config_sha256=config_sha256,
            base_checkpoint_sha256=base_sha256,
            scene_audit_sha256=artifacts["sha256"]["scene_audit"],
            data_audit_sha256=artifacts["sha256"]["data_audit"],
            stats_recompute_sha256=artifacts["sha256"]["stats_recompute"],
        )
        if problems:
            raise SystemExit("ready manifest failed: " + "; ".join(problems))
        argv += ["--require-ready-manifest", "--ready-manifest", str(args.ready_manifest)]
    elif args.ready_manifest is not None:
        argv += ["--ready-manifest", str(args.ready_manifest)]
    if optimisation_args.get("ema_decay") is not None:
        argv += ["--ema-decay", str(optimisation_args["ema_decay"])]
    shown = [value if value not in (assets["checkpoint"], assets["tokenizer"]) else "<asset>"
             for value in argv]
    print(json.dumps({"argv": shown, "run_id": run_id, "run_directory": str(run_dir),
                      "experiment_id": experiment_id, "view": view.name,
                      "admission_sha256": common.sha256_file(admission),
                      "sampling_plan_sha256": common.sha256_file(plan)}, indent=2))
    if args.dry_run:
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = run_dir / "training_config_snapshot.json"
    if snapshot_path.exists():
        raise SystemExit("refusing to overwrite the registered training config snapshot")
    with training_config.open("rb") as source, snapshot_path.open("xb") as target:
        shutil.copyfileobj(source, target)
    return subprocess.call(argv)


if __name__ == "__main__":
    raise SystemExit(main())
