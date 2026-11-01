from __future__ import annotations

import ast
import copy
import hashlib
import io
import inspect
import json
import re
import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "so101"))

import scene1_validation as validation
import scene1_run_training


class Scene1ValidationTests(unittest.TestCase):
    def test_v3_config_preserves_52_12_and_requested_budgets(self) -> None:
        config = json.loads(
            (ROOT / "configs/so101/scene1_training_base_v3.json").read_text(
                encoding="utf-8"))
        self.assertEqual(config["data_binding"]["train_episodes"], 52)
        self.assertEqual(config["data_binding"]["validation_episodes"], 12)
        self.assertEqual(config["optimisation"]["updates"], 25000)
        self.assertEqual(config["optimisation"]["global_batch_size"], 8)
        self.assertEqual(config["optimisation"]["warmup_updates"], 1000)
        self.assertEqual(config["validation_loss"]["probe"]["every_updates"], 50)
        self.assertEqual(config["validation_loss"]["probe"]["samples"], 256)
        self.assertEqual(config["validation_loss"]["full"]["samples"], 8300)
        self.assertEqual(config["validation_loss"]["batch_size"], 16)
        self.assertTrue(config["checkpointing"]["retain_all_saved_steps"])
        self.assertEqual(
            tuple(config["checkpointing"]["checkpoint_steps"]),
            validation.EXPECTED_CHECKPOINT_STEPS,
        )

    def test_full_validation_plan_covers_8300_windows_with_exact_padding(self) -> None:
        plan = validation.fixed_validation_sample_plan(8300, 16, 42)
        self.assertEqual(plan["batch_count"], 519)
        self.assertEqual(plan["padded_count"], 8304)
        self.assertEqual(plan["padding"], 4)
        self.assertEqual(plan["valid_samples_per_batch"][-1], 12)
        self.assertEqual(plan["_sampler_indices"][:8300], list(range(8300)))
        self.assertEqual(plan["_sampler_indices"][8300:], [0, 1, 2, 3])
        self.assertEqual(sum(plan["valid_samples_per_batch"]), 8300)

    def test_probe_selects_256_samples_and_covers_all_twelve_episodes(self) -> None:
        metadata = [
            {"sample_index": episode * 100 + frame, "episode_id": episode}
            for episode in range(12)
            for frame in range(100)
        ]
        first, audit = validation.balanced_probe_indices(metadata, 256, 42)
        second, second_audit = validation.balanced_probe_indices(metadata, 256, 42)
        self.assertEqual(first, second)
        self.assertEqual(audit, second_audit)
        self.assertEqual(len(first), 256)
        self.assertEqual(len(set(first)), 256)
        self.assertEqual(audit["covered_episodes"], 12)
        self.assertEqual(set(audit["per_episode"].values()), {21, 22})
        probe_plan = validation.fixed_sample_plan(first, 16, 42)
        self.assertEqual(probe_plan["batch_count"], 16)
        self.assertEqual(probe_plan["padding"], 0)

    def test_every_fifty_update_and_checkpoint_schedules_are_independent(self) -> None:
        steps = validation.validation_steps(25000, 50)
        self.assertEqual(len(steps), 500)
        self.assertEqual(steps[:3], (50, 100, 150))
        self.assertEqual(steps[-1], 25000)
        config = {
            "checkpointing": {
                "checkpoint_steps": list(validation.EXPECTED_CHECKPOINT_STEPS)
            }
        }
        self.assertEqual(
            validation.checkpoint_steps(config, 25000),
            validation.EXPECTED_CHECKPOINT_STEPS,
        )

    def test_launcher_strict_config_accepts_v3_and_rejects_metric_mixing(self) -> None:
        config = json.loads(
            (ROOT / "configs/so101/scene1_training_base_v3.json").read_text(
                encoding="utf-8"))
        args = SimpleNamespace(
            steps=25000, batch_size=8, lr_warmup_steps=1000,
            peak_lr=5e-5, decay_steps=25000, decay_lr=5e-6,
            ema_decay=0.999, gradient_clip_norm=1.0, fsdp_devices=4,
            memory_margin_mib=10, action_horizon=10, state_mode="on",
            save_interval=1,
            registered_base_sha256=config["runtime_binding"]["base_checkpoint_sha256"],
            freeze_vision_encoder=True, require_validation_loss=True,
            validation_every=50, validation_seed=42, validation_batch_size=16,
            checkpoint_probe=False, validation_benchmark=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            view = Path(temporary)
            (view / "assets/so101").mkdir(parents=True)
            norm = view / "assets/so101/norm_stats.json"
            norm.write_bytes(b"norm")
            config["data_binding"]["norm_stats_sha256"] = hashlib.sha256(
                norm.read_bytes()).hexdigest()
            meta = {
                "view_sha256": config["data_binding"]["view_sha256"],
                "episodes": (
                    [{"split": "train"}] * 52
                    + [{"split": "validation"}] * 12
                ),
            }
            (view / "meta.json").write_text(
                json.dumps(meta), encoding="utf-8")
            self.assertEqual(
                scene1_run_training.strict_training_config_problems(
                    config, args, view), [])
            probe_args = copy.copy(args)
            probe_args.steps = 2
            probe_args.checkpoint_probe = True
            probe_args.validation_benchmark = True
            self.assertEqual(
                scene1_run_training.strict_training_config_problems(
                    config, probe_args, view), [])
            changed = json.loads(json.dumps(config))
            changed["validation_loss"]["probe"]["metric"] = "val_loss_full"
            self.assertTrue(
                scene1_run_training.strict_training_config_problems(
                    changed, args, view))

    def test_weighted_average_excludes_padding_and_rejects_nonfinite(self) -> None:
        result = validation.weighted_validation_loss(
            [[1.0] * 8, [5.0] * 4 + [999.0] * 4],
            [8, 4],
        )
        self.assertEqual(result["samples_evaluated"], 12)
        self.assertAlmostEqual(result["validation_loss"], (8.0 + 20.0) / 12.0)
        with self.assertRaises(FloatingPointError):
            validation.weighted_validation_loss(
                [[float("nan")]], [1])

    def test_validation_call_is_gradient_free_and_train_false(self) -> None:
        source = inspect.getsource(validation.validation_batch_losses)
        self.assertIn("train=False", source)
        self.assertNotIn("value_and_grad", source)
        self.assertNotIn("tx.update", source)
        self.assertNotIn("optax.apply_updates", source)

    def test_trainer_checks_state_tree_identity_around_validation(self) -> None:
        source = (ROOT / "scripts/so101/scene1_train_so101.py").read_text(
            encoding="utf-8")
        self.assertIn("before_state_ids", source)
        self.assertIn("state_tree_unchanged", source)
        self.assertIn("validation replaced a parameter or optimizer leaf", source)

    def test_trainer_runs_full_after_save_and_probe_every_fifty(self) -> None:
        source = (ROOT / "scripts/so101/scene1_train_so101.py").read_text(
            encoding="utf-8")
        save = source.index(
            "original_save_state(manager, state, data_loader, actual_step)")
        wait = source.index("manager.wait_until_finished()", save)
        full = source.index('evaluate_validation(state, "full"', save)
        probe = source.index('evaluate_validation(state, "probe"', save)
        self.assertLess(save, wait)
        self.assertLess(wait, full)
        self.assertLess(full, probe)
        self.assertIn(
            "actual_step % int(args.validation_every) == 0", source)
        self.assertIn("checkpoint retention lost registered steps", source)

    def test_scene1_guard_rewrites_zero_based_loop_step_to_real_update(self) -> None:
        path = ROOT / "scripts/so101/scene1_train_so101.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(item for item in tree.body if isinstance(item, ast.ClassDef)
                    and item.name == "_FiniteOutputGuard")
        namespace = {"io": io, "re": re}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
             namespace)
        output = io.StringIO()
        guard = namespace["_FiniteOutputGuard"](
            output, require_initial_update=True, step_offset=1)
        guard.write("Step 0: loss=0.1, grad_norm=0.2, param_norm=3.0\n")
        self.assertIn("Step 1:", output.getvalue())

    def test_plan_hash_is_bound_and_cannot_be_overwritten(self) -> None:
        plan = {
            "schema_version": 1,
            "sample_count": 8300,
            "seed": 42,
            "_sampler_indices": list(range(8300)) + [0, 1, 2, 3],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, digest = validation.register_plan(
                root, plan, name="validation_full_plan.json", fresh=True)
            second, second_digest = validation.register_plan(
                root, plan, name="validation_full_plan.json", fresh=False)
            self.assertEqual(first, second)
            self.assertEqual(digest, second_digest)
            changed = dict(plan, seed=43)
            with self.assertRaisesRegex(ValueError, "differs"):
                validation.register_plan(
                    root, changed, name="validation_full_plan.json", fresh=False)
            self.assertNotIn("_sampler_indices", json.dumps(first))

    def test_resume_archives_only_records_ahead_of_restored_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, step in (
                ("val_loss_probe_450.json", 450),
                ("val_loss_probe_500.json", 500),
                ("val_loss_probe_550.json", 550),
                ("val_loss_full_500.json", 500),
            ):
                (root / name).write_text(json.dumps({"train_state_step": step}))
            moved = validation.archive_future_validation_records(
                root, 500, archive_name="resume-500")
            self.assertEqual(
                sorted(Path(value).name for value in moved),
                ["val_loss_probe_550.json"],
            )
            self.assertTrue((root / "val_loss_probe_450.json").is_file())
            self.assertTrue((root / "val_loss_probe_500.json").is_file())
            self.assertTrue((root / "val_loss_full_500.json").is_file())
            self.assertFalse((root / "val_loss_probe_550.json").exists())
            self.assertTrue(
                (root / "resume_validation_archive/resume-500/"
                 "val_loss_probe_550.json").is_file())


if __name__ == "__main__":
    unittest.main()
