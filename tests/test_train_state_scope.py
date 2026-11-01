"""Checkpoint audit must report the trainer's filter, rather than duplicate path rules."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


READER = (Path(__file__).resolve().parents[1] / "scripts" / "so101"
          / "read_train_state.py")


class TrainStateScopeTests(unittest.TestCase):
    def summary(self, selected):
        tree = ast.parse(READER.read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "_trainable_summary")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), READER.name, "exec"), namespace)
        criterion = object()
        params = SimpleNamespace(filter=Mock(return_value=SimpleNamespace(
            to_pure_dict=lambda: selected)))

        def flatten(node, sep):
            self.assertEqual(sep, "/")
            self.assertIs(node, selected)
            return node

        traversal = ModuleType("flax.traverse_util")
        traversal.flatten_dict = flatten
        with patch.dict(sys.modules, {"flax.traverse_util": traversal}):
            result = namespace["_trainable_summary"](params, criterion)
        params.filter.assert_called_once_with(criterion)
        return result

    def test_selected_vision_and_action_expert_are_counted(self):
        selected = {
            "PaliGemma/img/kernel": SimpleNamespace(size=12),
            "PaliGemma/llm/attention_1/kernel": SimpleNamespace(size=20),
            "PaliGemma/llm/lora_a": SimpleNamespace(size=3),
            "action_in_proj/kernel": SimpleNamespace(size=4),
        }
        result = self.summary(selected)
        self.assertEqual(result["trainable_parameter_count"], 39)
        self.assertEqual(result["trainable_parameter_tensors"], 4)
        self.assertEqual(result["trainable_parameter_names"], sorted(selected))
        self.assertEqual(result["trainable_rule"], "config.trainable_filter")

    def test_empty_filter_selection_is_not_replaced_by_path_heuristics(self):
        result = self.summary({})
        self.assertEqual(result["trainable_parameter_count"], 0)
        self.assertEqual(result["trainable_parameter_tensors"], 0)
        self.assertEqual(result["trainable_parameter_names"], [])


if __name__ == "__main__":
    unittest.main()
