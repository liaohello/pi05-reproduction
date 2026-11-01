"""Clean60 contracts that hold without the external OpenPI runtime.

Covers the round's parameter plumbing: registered hyper-parameters, the training
scope switches, the multi-device launcher, and the layout filter.
"""
from __future__ import annotations

import ast
import json
import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "so101"))

import clean60_contracts as clean60


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def tree(relative: str) -> ast.Module:
    return ast.parse(source(relative))


def functions(module: ast.Module):
    return {node.name: node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)}


def argparse_flag(text: str, flag: str) -> str:
    for line in text.splitlines():
        if "add_argument" in line and f'"{flag}"' in line:
            return line
    raise AssertionError(f"{flag} is not declared")


class Clean60ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(source("configs/so101/clean60_training.json"))
        self.protocol = json.loads(source("configs/so101/clean60_protocol.json"))

    def test_registered_optimisation_baseline(self) -> None:
        optimisation = self.config["optimisation"]
        self.assertEqual(optimisation["planned_updates"], 25000)
        self.assertEqual(optimisation["decay_steps"], 25000)
        self.assertEqual(optimisation["primary_batch_size"], 8)
        self.assertEqual(optimisation["seed"], 42)
        self.assertEqual(optimisation["warmup_steps"], 500)
        self.assertEqual(optimisation["peak_lr"], 5e-5)
        self.assertEqual(optimisation["decay_lr"], 5e-6)
        self.assertEqual(optimisation["gradient_clip_norm"], 1.0)
        self.assertEqual(optimisation["dtype"], "bfloat16")
        self.assertFalse(optimisation["allow_batch_fallback"])
        self.assertFalse(optimisation["allow_update_override"])
        self.assertNotIn("fallback_batch_sizes", optimisation)
        self.assertEqual(self.config["checkpointing"]["save_interval"], 5000)
        self.assertIsNone(self.config["optimisation"].get("ema_decay"))
        self.assertIn("@sha256:", self.config["runtime_binding"]["image"])
        self.assertEqual(clean60.training_config_problems(self.config), [])

    def test_scope_is_language_lora_plus_full_expert_and_vision(self) -> None:
        initialisation = self.config["initialisation"]
        self.assertEqual(initialisation["paligemma_variant"], "gemma_2b_lora")
        self.assertEqual(initialisation["action_expert_variant"], "gemma_300m")
        self.assertNotIn("lora", initialisation["action_expert_variant"])
        self.assertEqual(initialisation["base_checkpoint"], "pi05_libero")
        self.assertFalse(initialisation["resume_previous_so101_checkpoint"])
        self.assertEqual(initialisation["state_mode"], "on")

    def test_no_stale_history_counts(self) -> None:
        text = json.dumps(self.config)
        for stale in ("30000", "e4t_30k", '"view_episodes": 50', '"train": 40', '"validation": 10'):
            self.assertNotIn(stale, text, stale)
        data = self.config["data"]
        self.assertEqual((data["view_episodes"], data["train"], data["validation"]), (60, 48, 12))
        self.assertEqual(data["recovery"], 0)
        self.assertFalse(data["perturbations_enabled"])

    def test_run_is_isolated_from_history(self) -> None:
        self.assertTrue(self.config["run"]["directory"].startswith("runs/so101/clean60"))
        self.assertNotEqual(self.config["run"]["directory"], "runs/so101/e4t")

    def test_collection_protocol_is_clean_and_spread(self) -> None:
        layout = self.protocol["scene_layout"]
        self.assertEqual(layout["object_count"], 4)
        self.assertGreaterEqual(layout["minimum_pairwise_distance_m"], 0.07)
        self.assertFalse(layout["perturbations_enabled"])
        self.assertFalse(layout["visual_randomization_enabled"])
        self.assertTrue(layout["vary_positions_between_episodes"])
        self.assertFalse(layout["reuse_previous_episodes"])
        split = self.protocol["collection"]["split"]
        self.assertEqual((split["train"], split["validation"]), (48, 12))
        self.assertEqual(self.protocol["collection"]["target_successes"], 60)
        evaluation = set(range(*[self.protocol["evaluation"]["seed_range"][0],
                                 self.protocol["collection"]["direct_seed_range"][1]]))
        self.assertFalse(evaluation & set(range(*[self.protocol["collection"]["direct_seed_range"][0],
                                                 self.protocol["collection"]["direct_seed_range"][1]])))

    def test_admission_numbers_are_measured_not_placeholders(self) -> None:
        devices = self.config["devices"]["capacity_probe"]
        self.assertEqual(devices["fsdp_devices"], 4)
        selected = self.config["devices"]["selected"]
        self.assertEqual((selected["gpu_count"], selected["fsdp_devices"]), (4, 4))
        self.assertEqual(selected["per_device_batch_shard"], 2)
        self.assertGreater(devices["measured_per_card_demand_mib"], 0)
        policy = self.config["resource_policy"]
        self.assertEqual(policy["measured_per_card_demand_mib"], devices["measured_per_card_demand_mib"])
        self.assertEqual(policy["required_free_memory_mib"],
                         policy["measured_per_card_demand_mib"] + policy["memory_margin_mib"])


class LauncherPlumbingTest(unittest.TestCase):
    def test_multi_device_flags_reach_the_trainer(self) -> None:
        text = source("scripts/so101/e4t_run_training.py")
        self.assertIn("--gpu-indices", text)
        self.assertIn("--fsdp-devices", text)
        self.assertIn("--paligemma-variant", text)
        self.assertIn("--action-expert-variant", text)
        self.assertIn("--expected-groups", text)
        self.assertNotIn('"--gpu-index", str(args.gpu_index)', text)

    def test_registered_run_settings_reach_trainstate_config(self) -> None:
        launcher = source("scripts/so101/e4t_launch_training.py")
        runner = source("scripts/so101/e4t_run_training.py")
        trainer = source("scripts/so101/train_so101.py")
        runtime = source("scripts/so101/openpi_so101.py")
        for flag in ("--seed", "--action-horizon", "--optimizer",
                     "--gradient-clip-norm", "--dtype", "--run-id"):
            self.assertIn(flag, launcher + runner + trainer)
        for field in ("action_horizon", "seed", "gradient_clip_norm", "optimizer_name",
                      "ema_decay", "dtype"):
            self.assertIn(field, runtime)
        self.assertIn("XLA_PYTHON_CLIENT_PREALLOCATE={str(args.jax_preallocate).lower()}", runner)
        self.assertIn('"run_id": args.run_id', trainer)

    def test_historical_episode_defaults_are_gone(self) -> None:
        text = source("scripts/so101/e4t_run_training.py")
        for flag in ("--expected-total-episodes", "--expected-train-episodes",
                    "--expected-validation-episodes", "--dataset-label"):
            line = argparse_flag(text, flag)
            self.assertIn("default=None", line, flag)
            self.assertNotIn("default=50", line)
            self.assertNotIn("default=40", line)
            self.assertNotIn("default=10", line)

    def test_launch_wrapper_has_no_stale_exp_name_or_directory(self) -> None:
        text = source("scripts/so101/e4t_launch_training.py")
        self.assertNotIn("e4t_30k_state_on_b8", text)
        self.assertIn("run_settings['exp_name']", text)
        self.assertIn("experiment_id", text)
        self.assertIn("clean60_training.json", text)

    def test_registry_accepts_a_device_list(self) -> None:
        text = source("scripts/so101/e4t_resources.py")
        self.assertIn('"device={args.gpu_index}"', text)
        self.assertIn("memory_free_mib_per_device", text)

    def test_registered_scope_resolves_to_language_lora_full_expert_and_vision(self) -> None:
        from model_scope import resolve_model_scope
        import json as _json
        config = _json.loads(source("configs/so101/clean60_training.json"))
        self.assertEqual(resolve_model_scope(config), {
            "paligemma_variant": "gemma_2b_lora",
            "action_expert_variant": "gemma_300m",
            "freeze_vision_encoder": False})

    def test_freeze_filter_excludes_action_expert_and_lora_from_frozen_scope(self) -> None:
        module = tree("scripts/so101/openpi_so101.py")
        body = ast.dump(functions(module)["make_freeze_filter"])
        self.assertNotIn("get_freeze_filter", body)
        self.assertIn(".*llm.*", body)
        self.assertIn(".*llm.*_1.*", body)
        self.assertIn(".*lora.*", body)
        self.assertIn(".*PaliGemma/img.*", body)
        signature = functions(tree("scripts/so101/openpi_so101.py"))["make_freeze_filter"].args
        defaults = [ast.unparse(value) for value in signature.kw_defaults if value is not None]
        self.assertIn("False", defaults)
        self.assertNotIn("freeze_vision_encoder=True", source("scripts/so101/openpi_so101.py"))
        signature = functions(module)["make_model_config"].args
        names = [argument.arg for argument in list(signature.args) + list(signature.kwonlyargs)]
        self.assertIn("paligemma_variant", names)
        self.assertIn("action_expert_variant", names)

    def test_train_forward_passes_devices_and_variants(self) -> None:
        text = source("scripts/so101/train_so101.py")
        self.assertIn("--fsdp-devices", text)
        self.assertIn("fsdp_devices=args.fsdp_devices", text)
        self.assertIn("**model_scope", text)
        self.assertIn("_runtime_snapshot", text)
        self.assertIn("must be divisible by the OpenPI device count", text)


class SamplingGroupTest(unittest.TestCase):
    def test_plan_builder_and_loader_resolve_groups_identically(self) -> None:
        plan = source("scripts/so101/build_e4t_sampling_plan.py")
        loader = source("scripts/so101/openpi_so101.py")
        self.assertIn('"clean"', plan)
        self.assertIn('row.get("subgroup") or row.get("category") or "clean"', plan)
        self.assertIn('str(episode.get("category", "clean"))', loader)
        self.assertIn('episode.get("subgroup", episode.get("category", "clean"))', loader)

    def test_family_matching_is_role_aware_and_uses_cube_scale_tolerance(self):
        from clean60_scene_audit import _ik_record_complete, _same_family

        left = [
            {"color": "red", "supply_arm": "left", "destination_arm": "left",
             "route": "direct", "source_region": "left_source", "xy": [0.0, 0.0]},
            {"color": "blue", "supply_arm": "right", "destination_arm": "right",
             "route": "direct", "source_region": "right_source", "xy": [0.2, 0.0]},
            {"color": "red", "supply_arm": "left", "destination_arm": "left",
             "route": "direct", "source_region": "left_source", "xy": [0.0, 0.2]},
            {"color": "blue", "supply_arm": "right", "destination_arm": "right",
             "route": "direct", "source_region": "right_source", "xy": [0.2, 0.2]},
        ]
        right = json.loads(json.dumps(left))
        right[0]["xy"] = [0.0125, 0.0]
        matched, distance = _same_family(left, right, 0.0125)
        self.assertTrue(matched)
        self.assertAlmostEqual(distance, 0.0125)
        right[0]["supply_arm"] = "right"
        self.assertEqual(_same_family(left, right, 0.0125), (False, None))
        grasp = {"converged": True, "target_position_m": [0.1, 0.2, 0.3]}
        hover = {"converged": True, "target_position_m": [0.1, 0.2, 0.35]}
        self.assertTrue(_ik_record_complete(grasp, hover, [0.1, 0.2], 0.05, 0.3))
        hover["converged"] = False
        self.assertFalse(_ik_record_complete(grasp, hover, [0.1, 0.2], 0.05, 0.3))


class Clean60BindingTest(unittest.TestCase):
    def exporter_projection(self):
        module = tree("scripts/so101/export_training_view.py")
        node = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == "public_episode")
        environment = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]),
                     "export_training_view.py", "exec"), environment)
        return environment["public_episode"]

    def replay_gate(self):
        module = tree("scripts/so101/export_training_view.py")
        node = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == "replay_audit_passed")
        environment = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]),
                     "export_training_view.py", "exec"), environment)
        return environment["replay_audit_passed"]

    def fixture(self):
        project = self.exporter_projection()
        rows = []
        for index in range(60):
            raw = {
                "episode_index": index, "seed": 12000 + index,
                "scene_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                "split": "train" if index < 48 else "validation",
                "frames": 100, "state_shape": [100, 12], "action_shape": [100, 12],
                "image_shape": [480, 640, 3],
                "max_state_error_vs_trace": 0.0, "max_action_error_vs_trace": 0.0,
                "category": "clean", "subgroup": "clean", "perturbation_frame": None,
                "eligible_windows": 91, "excluded_cross_event_windows": 0,
                "fixed_windows": list(range(16)), "event_windows": list(range(16)),
            }
            rows.append(project(raw, require_clean=True))
        binding = {
            "passed": True, "episodes_bound": 60,
            "checks": {
                "object_counts": [4], "routes": ["direct"], "perturbation_free": True,
                "minimum_pairwise_distance_m": {"episodes_below": 0},
            },
        }
        clean_profile = {
            "category": "clean", "source": clean60.CLEAN_SOURCE,
            "perturbation_free": True, "episodes_bound": 60,
            "protocol_binding_sha256": clean60.sha256_json(binding),
        }
        meta = {
            "schema_version": 2, "episodes": rows,
            "state_action_dim": 12, "feature_names": clean60.FEATURE_NAMES,
            "feature_units": clean60.FEATURE_UNITS, "cameras": clean60.CAMERAS,
            "action_semantics": clean60.ACTION_SEMANTICS,
            "clean_profile": clean_profile,
            "view_sha256": hashlib.sha256(b"view").hexdigest(),
        }
        split = {
            "train": list(range(48)), "validation": list(range(48, 60)),
        }
        admission = {
            "passed": True, "data_gate": {
                "passed": True, "clean_source": clean60.CLEAN_SOURCE},
            "view_sha256": meta["view_sha256"],
            "split_sha256": clean60.sha256_json(split),
            "train_stats_sha256": hashlib.sha256(b"stats").hexdigest(),
            "totals": {"episodes": 60, "train": 48, "validation": 12,
                       "clean": 60, "recovery": 0},
        }
        return meta, binding, split, admission

    def test_exporter_projection_is_accepted_by_real_launcher_contract(self):
        meta, binding, split, admission = self.fixture()
        self.assertEqual(clean60.view_contract_problems(
            meta, protocol_binding=binding,
            expected_counts=(60, 48, 12), require_clean=True), [])
        self.assertEqual(clean60.admission_contract_problems(
            admission, meta=meta, split=split,
            train_stats_sha256=admission["train_stats_sha256"]), [])
        for row in meta["episodes"]:
            self.assertIsNone(row["perturbation_frame"])
            self.assertEqual((row["category"], row["subgroup"]), ("clean", "clean"))

    def test_unknown_or_inferred_clean_source_is_rejected(self):
        meta, binding, _, _ = self.fixture()
        broken = json.loads(json.dumps(meta))
        del broken["clean_profile"]
        self.assertTrue(clean60.view_contract_problems(
            broken, protocol_binding=binding, require_clean=True))
        broken = json.loads(json.dumps(meta))
        del broken["episodes"][0]["category"]
        self.assertTrue(clean60.view_contract_problems(
            broken, protocol_binding=binding, require_clean=True))

    def test_exporter_accepts_both_complete_replay_report_schemas(self):
        gate = self.replay_gate()
        self.assertTrue(gate({"passed": True, "problems": [],
                              "replay": {"total": 60, "passed": 60}}, 60))
        self.assertTrue(gate({"ok": True, "replay_total": 60, "replay_passed": 60}, 60))
        self.assertFalse(gate({"passed": True, "problems": [],
                               "replay": {"total": 60, "passed": 59}}, 60))

    def test_norm_and_ready_schema_fail_on_hash_or_gate_drift(self):
        digest = hashlib.sha256(b"x").hexdigest()
        stats = {
            "mean": [0.0] * 12, "std": [1.0] * 12,
            "q01": [-1.0] * 12, "q99": [1.0] * 12,
        }
        audit = {
            "schema_version": 3, "passed": True,
            "view_sha256": digest, "norm_stats_sha256": digest,
            "base_checkpoint_sha256": digest, "training_config_sha256": digest,
            "binding": {"train_episodes": 48, "validation_episodes": 12},
            "statistics": {
                "recomputed_from_train_only_matches_view": True,
                "all_values_finite": True,
                "near_constant_dims_std_lt_1e-6": {"state": [], "actions": []},
                "recorded_train_only": {"state": stats, "actions": stats},
            },
            "transform": {
                "internal_action_dim": 32, "worst_recovery_error": 0.0,
                "pad_first_12_max_abs_error": 0.0, "padded_tail_max_abs": 0.0,
                "clipping_applied": False,
                "inference_output_execution_order_verified": True,
            },
            "loader_consistency": {
                "episodes_covered": 48,
                "action_max_abs_error_vs_runtime_pipeline": 0.0,
                "state_max_abs_error_vs_runtime_pipeline": 0.0,
            },
            "semantics": {
                "feature_units": clean60.FEATURE_UNITS,
                "action_horizon": 10, "action_shape": [10, 12],
            },
        }
        expected = {
            "view_sha256": digest, "norm_stats_sha256": digest,
            "base_checkpoint_sha256": digest, "training_config_sha256": digest,
        }
        self.assertEqual(clean60.norm_audit_problems(audit, **expected), [])
        wrong = dict(expected, view_sha256="0" * 64)
        self.assertTrue(clean60.norm_audit_problems(audit, **wrong))
        gates = {name: {"passed": True} for name in clean60.READY_GATES}
        ready = {
            "status": "READY", "mode": "PREPARE_ONLY",
            "formal_training_performed": False, "gates": gates,
            "bindings": {
                **expected, "norm_audit_sha256": digest,
            },
        }
        ready_args = {
            **expected, "norm_audit_sha256": digest,
        }
        ready_args["norm_stats_sha256"] = digest
        self.assertEqual(clean60.ready_manifest_problems(ready, **ready_args), [])
        ready["status"] = "BLOCKED"
        self.assertTrue(clean60.ready_manifest_problems(ready, **ready_args))

    def test_parameter_missing_and_shape_mismatch_are_failures(self):
        reference = {
            "action_in_proj/kernel": {"shape": (32, 1024), "dtype": "bfloat16"},
            "action_in_proj/bias": {"shape": (32,), "dtype": "bfloat16"},
        }
        self.assertTrue(clean60.tree_structure_problems(
            reference, {"action_in_proj/kernel": reference["action_in_proj/kernel"]}))
        wrong_shape = dict(reference)
        wrong_shape["action_in_proj/bias"] = {"shape": (31,), "dtype": "bfloat16"}
        self.assertTrue(clean60.tree_structure_problems(reference, wrong_shape))

    def test_resource_policy_has_no_hidden_eighty_percent_cap(self):
        from resource_policy import memory_plan

        device = {"index": 0, "uuid": "GPU-test", "memory_total_mib": 24564,
                  "memory_used_mib": 512, "memory_free_mib": 24052}
        plan = memory_plan([device], demand_mib=18000, margin_mib=512)
        self.assertGreater(plan["jax_memory_fraction"], 0.80)
        with self.assertRaises(RuntimeError):
            memory_plan([device], demand_mib=23600, margin_mib=512)
        self.assertNotIn("min(0.80", source("scripts/so101/e4t_run_training.py"))

    def test_launcher_shell_consumes_current_norm_schema(self):
        text = source("scripts/so101/clean60_prepare_and_launch.sh")
        self.assertIn("internal_action_dim", text)
        self.assertNotIn("internal_dim_from_loader", text)
        self.assertNotIn("validation_leak_into_mean_max_abs", text)
        self.assertIn("--launch", text)
        self.assertIn("prepare-only complete", text)
        self.assertIn("CONTAINER_VIEW=/work/clean60_view", text)
        self.assertIn('--mount "$VIEW":"$CONTAINER_VIEW"', text)
        self.assertNotIn("$HOME/miniconda3/envs/bimanual/bin/python", text)


class LayoutFilterTest(unittest.TestCase):
    class _Obj:
        def __init__(self, xy, route="direct"):
            self.xy, self.route = xy, route

        def as_dict(self):
            return {"xy": list(self.xy), "route": self.route}

    class _Scene:
        def __init__(self, objects, extra=None):
            self.objects, self.extra, self.sha256 = objects, extra or {}, "x"

        def digest(self):
            return "digest"

    def test_near_objects_and_perturbations_are_rejected(self) -> None:
        module = tree("scripts/so101/clean_layout.py")
        self.assertIn("inspect_layout", functions(module))
        import clean_layout

        close = [self._Obj((0.0, 0.0)), self._Obj((0.05, 0.0)),
                 self._Obj((0.0, 0.5)), self._Obj((0.5, 0.5))]
        with self.assertRaises(ValueError):
            clean_layout.inspect_layout(self._Scene(close), 0.08)
        spread = [self._Obj((0.0, 0.0)), self._Obj((0.2, 0.0)),
                  self._Obj((0.0, 0.3)), self._Obj((0.4, 0.4))]
        audit = clean_layout.inspect_layout(self._Scene(spread), 0.08)
        self.assertAlmostEqual(audit["minimum_pairwise_distance_m"], 0.2, places=6)
        with self.assertRaises(ValueError):
            clean_layout.inspect_layout(self._Scene(spread, {"perturbation": [1]}), 0.08)
        with self.assertRaises(ValueError):
            clean_layout.inspect_layout(self._Scene(spread[:3]), 0.08)

    def test_handover_task_is_not_the_registered_task(self) -> None:
        protocol = json.loads(source("configs/so101/clean60_protocol.json"))
        self.assertEqual(protocol["task"]["name"], "direct_sorting")
        self.assertFalse(protocol["task"]["transfer_protocol"])
        self.assertIn("red tray", protocol["task"]["instruction"].lower())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "so101"))
    unittest.main()
