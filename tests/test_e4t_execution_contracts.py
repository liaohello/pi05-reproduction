"""E4-T contracts that can be checked without loading the external OpenPI runtime."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(relative_path: str) -> ast.Module:
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def called(node: ast.AST, attr: str) -> list[ast.Call]:
    return [child for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
            and child.func.attr == attr]


class E4TExecutionContractTest(unittest.TestCase):
    def test_batch_size_reaches_openpi_train_config_and_loader(self):
        launcher = function(module("scripts/so101/train_so101.py"), "main")
        config_calls = called(launcher, "make_train_config")
        self.assertTrue(config_calls)
        batch_arg = next(kw.value for call in config_calls for kw in call.keywords
                         if kw.arg == "batch_size")
        self.assertEqual(ast.dump(batch_arg), ast.dump(ast.Attribute(
            value=ast.Name(id="args", ctx=ast.Load()), attr="batch_size", ctx=ast.Load())))

        adapter = module("scripts/so101/openpi_so101.py")
        config_builder = function(adapter, "make_train_config")
        train_config_calls = called(config_builder, "TrainConfig")
        configured_batch = [kw.value for call in train_config_calls for kw in call.keywords
                            if kw.arg == "batch_size"]
        self.assertTrue(any(isinstance(value, ast.Call)
                            and isinstance(value.func, ast.Name) and value.func.id == "int"
                            and len(value.args) == 1 and isinstance(value.args[0], ast.Name)
                            and value.args[0].id == "batch_size" for value in configured_batch))

        loader = function(adapter, "create_so101_data_loader")
        data_loader_calls = called(loader, "TorchDataLoader")
        batch_values = [kw.value for call in data_loader_calls for kw in call.keywords
                        if kw.arg == "local_batch_size"]
        self.assertTrue(any(isinstance(value, ast.Attribute) and value.attr == "batch_size"
                            and isinstance(value.value, ast.Name) and value.value.id == "config"
                            for value in batch_values))

    def test_training_and_stats_use_only_the_frozen_train_split(self):
        launcher = function(module("scripts/so101/train_so101.py"), "main")
        dataset_calls = called(launcher, "SO101EpisodeDataset")
        train_calls = [call for call in dataset_calls if any(
            kw.arg == "split" and isinstance(kw.value, ast.Constant) and kw.value.value == "train"
            for kw in call.keywords)]
        self.assertTrue(train_calls)
        self.assertTrue(any(kw.arg == "episode_ids" and isinstance(kw.value, ast.Name)
                            and kw.value.id == "train_ids_meta"
                            for call in train_calls for kw in call.keywords))

        exporter = (ROOT / "scripts/so101/export_e4t_view.py").read_text(encoding="utf-8")
        self.assertIn('train_rows = [row for row in export_rows if row["split"] == "train"]', exporter)
        self.assertIn('for row in train_rows:', exporter)
        self.assertIn('data["state"]', exporter)
        self.assertIn('data["action"]', exporter)
        self.assertIn('"train_only_statistics": True', exporter)

    def test_sampler_and_policy_inputs_do_not_receive_diagnostic_labels(self):
        adapter = module("scripts/so101/openpi_so101.py")
        dataset_class = next(node for node in adapter.body
                             if isinstance(node, ast.ClassDef) and node.name == "SO101EpisodeDataset")
        getitem = next(node for node in dataset_class.body
                       if isinstance(node, ast.FunctionDef) and node.name == "__getitem__")
        returned = next(node.value for node in ast.walk(getitem)
                         if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict))
        keys = {key.value for key in returned.keys if isinstance(key, ast.Constant)}
        self.assertEqual(keys, {"observation.state", "action", "task"})
        self.assertTrue(any(key is None for key in returned.keys))  # camera arrays only
        body = ast.get_source_segment((ROOT / "scripts/so101/openpi_so101.py").read_text(), getitem)
        for label in ("category", "subgroup", "split", "perturbation_frame", "sample_metadata"):
            self.assertNotIn(label, body)

    def test_evaluation_and_serving_share_the_so101_preprocessing_contract(self):
        """The open-loop evaluator and the policy server must use one input/inference path."""
        evaluator = (ROOT / "scripts/so101/evaluate_e4t_openloop.py").read_text(encoding="utf-8")
        server = (ROOT / "scripts/so101/serve_so101.py").read_text(encoding="utf-8")
        adapter = (ROOT / "scripts/so101/openpi_so101.py").read_text(encoding="utf-8")
        self.assertIn("policy._input_transform", evaluator)
        self.assertIn("policy.infer", evaluator)
        self.assertIn("openpi_so101.make_train_config", server)
        self.assertIn("_policy_config.create_trained_policy", server)
        self.assertIn("self._policy.infer", server)
        self.assertIn("ACTION_DIM = 12", adapter)
        self.assertIn("action_horizon=10", adapter)
        for camera in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            self.assertIn(camera, adapter)

    def test_prediction_targets_start_at_the_observed_frame(self):
        evaluator = function(module("scripts/so101/evaluate_e4t_openloop.py"), "main")
        source = ast.get_source_segment(
            (ROOT / "scripts/so101/evaluate_e4t_openloop.py").read_text(encoding="utf-8"),
            evaluator)
        self.assertIn("end = start + 10", source)
        self.assertIn("gt = actions_all[start:end].copy()", source)
        self.assertIn("raw_row(view, episode, start, gt)", source)
        self.assertIn('data["state"][start]', source)
        self.assertIn("if inferred.shape != (10, 12)", source)


if __name__ == "__main__":
    unittest.main()
