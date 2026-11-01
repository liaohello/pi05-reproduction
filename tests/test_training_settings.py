"""Saved checkpoint settings must survive changes to the launcher's defaults."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.so101.model_scope import resolve_model_scope


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCOPE = {"paligemma_variant": "gemma_2b_lora",
                "action_expert_variant": "gemma_300m_lora",
                "freeze_vision_encoder": True}
CLEAN60_SCOPE = {**LEGACY_SCOPE, "action_expert_variant": "gemma_300m",
                 "freeze_vision_encoder": False}


def load_function(filename, name):
    tree = ast.parse((ROOT / "scripts" / "so101" / filename).read_text(encoding="utf-8"))
    node = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    env = {"Path": Path, "json": json, "resolve_model_scope": resolve_model_scope}
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), env)
    return env[name]


class TrainingSettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.training = load_function("train_so101.py", "_training_settings")
        self.reader = load_function("read_train_state.py", "_checkpoint_settings")
        self.args = SimpleNamespace(mode="resume", state_mode=None, paligemma_variant=None,
                                    action_expert_variant=None, freeze_vision_encoder=None)

    def save(self, document):
        path = self.run / "config_initial.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_initial_training_keeps_requested_clean60_defaults(self):
        self.args.mode = "initial"
        self.assertEqual(self.training(self.args, self.run), {
            "model_scope": CLEAN60_SCOPE, "state_mode": "off"})

    def test_resume_recovers_historical_model_and_optimizer_scope(self):
        self.save({"state_mode": "on", "config": {
            "model": {key: value for key, value in LEGACY_SCOPE.items()
                      if key != "freeze_vision_encoder"},
            "freeze_filter": "All(PathRegex('.*llm.*'), PathRegex('.*PaliGemma/img.*'))"}})
        self.assertEqual(self.training(self.args, self.run), {
            "model_scope": LEGACY_SCOPE, "state_mode": "on"})

    def test_resume_recovers_explicit_clean60_scope(self):
        self.save({"state_mode": "on", "model_scope": CLEAN60_SCOPE})
        self.assertEqual(self.training(self.args, self.run), {
            "model_scope": CLEAN60_SCOPE, "state_mode": "on"})

    def test_resume_refuses_missing_snapshot_and_explicit_scope_changes(self):
        with self.assertRaisesRegex(ValueError, "config_initial"):
            self.training(self.args, self.run)
        self.save({"state_mode": "on", "model_scope": LEGACY_SCOPE})
        self.args.action_expert_variant = "gemma_300m"
        with self.assertRaisesRegex(ValueError, "action_expert_variant"):
            self.training(self.args, self.run)
        self.args.action_expert_variant = None
        self.args.state_mode = "off"
        with self.assertRaisesRegex(ValueError, "state_mode"):
            self.training(self.args, self.run)

    def test_reader_discovers_saved_scope_from_checkpoint_root(self):
        self.save({"state_mode": "on", "model_scope": CLEAN60_SCOPE})
        checkpoint_root = self.run / "checkpoints" / "pi05_so101_e2" / "experiment"
        self.assertEqual(self.reader(checkpoint_root, None, None), {
            "model_scope": CLEAN60_SCOPE, "state_mode": "on"})

    def test_reader_explicit_config_preserves_old_scope_and_rejects_state_conflict(self):
        config = self.save({"state_mode": "off", "model_scope": LEGACY_SCOPE})
        self.assertEqual(self.reader(self.run, config, None), {
            "model_scope": LEGACY_SCOPE, "state_mode": "off"})
        with self.assertRaisesRegex(ValueError, "state_mode"):
            self.reader(self.run, config, "on")

    def test_server_discovers_clean60_sidecars_from_checkpoint(self):
        snapshot = self.save({"state_mode": "on", "model_scope": CLEAN60_SCOPE})
        run_config = self.run / "run_config.json"
        run_config.write_text(json.dumps({"model_scope": CLEAN60_SCOPE}), encoding="utf-8")
        checkpoint = self.run / "checkpoints" / "pi05_so101_e2" / "experiment" / "4999"
        discover = load_function("serve_so101.py", "_saved_config_paths")
        self.assertEqual(discover(checkpoint, None, None), (run_config, snapshot))

    def test_server_keeps_explicit_sidecars(self):
        discover = load_function("serve_so101.py", "_saved_config_paths")
        run_config = self.run / "explicit-run.json"
        snapshot = self.run / "explicit-snapshot.json"
        self.assertEqual(discover(self.run, run_config, snapshot), (run_config, snapshot))


if __name__ == "__main__":
    unittest.main()
