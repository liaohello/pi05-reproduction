from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts/so101"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Scene1PiFastRuntimeContractTests(unittest.TestCase):
    def test_runtime_variants_and_scope_contract_are_explicit(self) -> None:
        text = (SCRIPTS / "scene1_pifast_ki_runtime.py").read_text(
            encoding="utf-8")
        self.assertIn('action_horizon=ACTION_HORIZON', text)
        self.assertIn('action_expert_variant="gemma_300m_lora_r64"', text)
        self.assertIn('PathRegex(".*PaliGemma/llm.*")', text)
        self.assertIn('Not(nnx_utils.PathRegex(".*lora.*"))', text)
        self.assertIn("fast_token_residual/table", text)

    def test_fast_transform_is_local_and_fails_closed(self) -> None:
        text = (SCRIPTS / "scene1_pifast_ki_fast_data.py").read_text(
            encoding="utf-8")
        self.assertIn("local_files_only=True", text)
        self.assertIn("refusing to truncate", text)
        self.assertIn("loss_mask", text)
        self.assertIn("if actions is None:", text)

    def test_training_wrapper_cannot_bypass_plan_or_audit(self) -> None:
        text = (SCRIPTS / "scene1_train_so101.py").read_text(
            encoding="utf-8")
        for marker in (
            "PI-FAST plan hash mismatch",
            "PI-FAST data/FAST audit did not pass",
            "formal PI-FAST training is not authorized by its plan",
            "formal fast_ce_weight differs from the frozen plan",
            'pifast_plan.get("resource", {}).get("minimum_memory_margin_mib", -1) < 100',
            "formal wall-clock limit is not registered",
        ):
            self.assertIn(marker, text)
        self.assertIn("plan_contract_problems(", text)

    def test_validation_uses_flow_only_for_joint_model(self) -> None:
        text = (SCRIPTS / "scene1_validation.py").read_text(
            encoding="utf-8")
        self.assertIn('getattr(model, "compute_flow_loss", model.compute_loss)', text)
        model_text = (SCRIPTS / "scene1_pifast_ki_model.py").read_text(
            encoding="utf-8")
        self.assertIn("axis=(-2, -1)", model_text)
        self.assertIn("jax.lax.stop_gradient", model_text)

    def test_memory_margin_is_one_hundred_mib(self) -> None:
        runtime = load_module("scene1_pifast_ki_contract")
        plan = json.loads((
            ROOT / "configs/so101/scene1_pifast_ki_training_v2.json"
        ).read_text(encoding="utf-8"))
        plan["status"] = "ready_for_training"
        plan["execution_authorized"] = True
        plan["loss"]["fast_ce_weight"] = 0.01
        plan["fast"]["real_horizon_16_length_audit"] = "pass"
        plan["resource"]["minimum_memory_margin_mib"] = 10
        plan["resource"]["wall_limit_s"] = 1
        problems = runtime.plan_problems(plan, formal=True)
        self.assertIn("minimum GPU memory margin is not 100 MiB", problems)
        plan["resource"]["minimum_memory_margin_mib"] = 100
        self.assertNotIn(
            "minimum GPU memory margin is not 100 MiB",
            runtime.plan_problems(plan, formal=True),
        )

    def test_resume_scope_accepts_rank64_variant(self) -> None:
        scope = load_module("scene1_model_scope")
        resolved = scope.resolve_model_scope({
            "model_scope": {
                "paligemma_variant": "gemma_2b_lora",
                "action_expert_variant": "gemma_300m_lora_r64",
                "freeze_vision_encoder": False,
            }
        })
        self.assertEqual(
            resolved["action_expert_variant"], "gemma_300m_lora_r64")

    def test_gpu_probe_binds_base_rng_and_training_arguments(self) -> None:
        probe = (SCRIPTS / "scene1_pifast_ki_gpu_probe.py").read_text(
            encoding="utf-8")
        self.assertIn("--expected-base-tree-sha256", probe)
        self.assertIn("pi05_base tree hash", probe)
        self.assertIn("train_rng, _ = jax.random.split", probe)
        self.assertIn("jax.random.fold_in(train_rng, state.step)", probe)
        launch = (SCRIPTS / "scene1_pifast_ki_launch.py").read_text(
            encoding="utf-8")
        for argument in (
            "--pifast-plan",
            "--pifast-data-audit",
            "--fast-ce-weight",
            "--action-expert-variant\", \"gemma_300m_lora_r64",
            "--no-freeze-vision-encoder",
            "--deterministic-ops",
        ):
            self.assertIn(argument, launch)
        self.assertIn(
            "already has checkpoints", launch)
        self.assertIn('line.split(",", 4)', launch)

    def test_source_freezer_records_every_file(self) -> None:
        text = (SCRIPTS / "scene1_pifast_ki_freeze_source.py").read_text(
            encoding="utf-8")
        self.assertIn("source_manifest.json", text)
        self.assertIn("sha256_file(path)", text)
        self.assertIn("scene1_pifast_ki_training_v2.json", text)


if __name__ == "__main__":
    unittest.main()
