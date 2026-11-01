"""Contract tests for resuming the existing formal clean60 run."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/so101/e4t_run_training.py"

sys.path.insert(0, str(ROOT / "scripts/so101"))
import e4t_run_training as runner  # noqa: E402


class Clean60ResumeContractTest(unittest.TestCase):
    def test_launcher_exposes_resume_and_authoritative_step(self):
        result = subprocess.run(
            [sys.executable, str(LAUNCHER), "--help"],
            capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("initial,resume,eval", result.stdout.replace(" ", ""))
        self.assertIn("--resume-step", result.stdout)

    def test_resume_command_uses_total_budget_and_restored_step(self):
        args = SimpleNamespace(
            mode="resume", resume_step=5001, run_id="run-resume",
            exp_name="clean60-resume-contract",
            state_mode="on",
            sampling_plan=None, data_admission=None, training_config_sha256=None,
            training_config=None, require_clean_data=False,
            require_action_norm_audit=False, registered_base_sha256=None,
            ema_decay=None, batch_size=8, steps=25000, save_interval=5000,
            data_loader_workers=8, lr_warmup_steps=500, peak_lr=5e-5,
            decay_steps=25000, decay_lr=5e-6, seed=42, action_horizon=10,
            optimizer="AdamW", gradient_clip_norm=1.0, dtype="bfloat16",
            fsdp_devices=4, paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m", freeze_vision_encoder=False,
            expected_groups=1, dataset_label="clean60-view",
            expected_total_episodes=60, expected_train_episodes=48,
            expected_validation_episodes=12,
        )
        command = runner.build_training_command(args, "/work/run", "/work/view")
        self.assertEqual(command[command.index("--mode") + 1], "resume")
        self.assertEqual(command[command.index("--resume-step") + 1], "5001")
        self.assertEqual(command[command.index("--steps") + 1], "25000")
        self.assertEqual(command[command.index("--batch-size") + 1], "8")

    def test_fixed_budget_guard_applies_to_initial_and_resume(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('if args.mode in {"initial", "resume"} and (', text)
        self.assertIn("must keep global batch 8 and 25000 updates", text)


if __name__ == "__main__":
    unittest.main()
