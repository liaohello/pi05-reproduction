"""Check SO101 runtime validation without importing JAX or allocating devices."""
from __future__ import annotations

import ast
import io
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "so101" / "train_so101.py"


class RuntimeSnapshotContractTests(unittest.TestCase):
    def setUp(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "_runtime_snapshot")
        module = ast.Module(body=[function], type_ignores=[])
        self.snapshot = {}
        self.devices = [SimpleNamespace(id=index, device_kind="MockDevice")
                        for index in range(4)]
        self.make_mesh = Mock(side_effect=lambda fsdp: SimpleNamespace(
            axis_names=("data", "fsdp"), shape={"data": len(self.devices) // fsdp,
                                               "fsdp": fsdp}))
        self.jax_mock = types.ModuleType("jax")
        self.jax_mock.devices = lambda: list(self.devices)
        self.jax_mock.default_backend = lambda: "mock"
        self.sharding_module = types.ModuleType("openpi.training.sharding")
        self.sharding_module.make_mesh = self.make_mesh
        openpi_module = types.ModuleType("openpi")
        openpi_module.__path__ = []
        training_module = types.ModuleType("openpi.training")
        training_module.__path__ = []
        training_module.sharding = self.sharding_module
        openpi_module.training = training_module
        self.modules = {
            "jax": self.jax_mock,
            "openpi": openpi_module,
            "openpi.training": training_module,
            "openpi.training.sharding": self.sharding_module,
        }
        exec(compile(module, str(SCRIPT), "exec"), self.snapshot)

    def call_snapshot(self, fsdp_devices: int, batch_size: int) -> dict:
        config = SimpleNamespace(fsdp_devices=fsdp_devices, batch_size=batch_size,
                                 num_train_steps=25000)
        with patch.dict(sys.modules, self.modules):
            return self.snapshot["_runtime_snapshot"](config)

    def test_four_devices_fsdp_four_preserves_batch_eight_layout(self) -> None:
        result = self.call_snapshot(fsdp_devices=4, batch_size=8)
        self.assertEqual(result["data_parallel_replicas"], 1)
        self.assertEqual(result["batch_per_data_replica"], 8)
        self.assertEqual(result["batch_per_data_shard"], 2)
        self.assertEqual(result["mesh_shape"], {"data": 1, "fsdp": 4})

    def test_four_devices_fsdp_two_preserves_batch_eight_layout(self) -> None:
        result = self.call_snapshot(fsdp_devices=2, batch_size=8)
        self.assertEqual(result["data_parallel_replicas"], 2)
        self.assertEqual(result["batch_per_data_replica"], 4)
        self.assertEqual(result["batch_per_data_shard"], 2)
        self.assertEqual(result["mesh_shape"], {"data": 2, "fsdp": 2})

    def test_invalid_fsdp_is_rejected_before_mesh_creation(self) -> None:
        for value in (0, -1, 3, 5):
            with self.subTest(fsdp_devices=value):
                self.make_mesh.reset_mock()
                with self.assertRaisesRegex(ValueError, "fsdp_devices"):
                    self.call_snapshot(fsdp_devices=value, batch_size=8)
                self.make_mesh.assert_not_called()

    def test_invalid_batch_is_rejected_before_mesh_creation(self) -> None:
        for value in (0, -1, 6):
            with self.subTest(batch_size=value):
                self.make_mesh.reset_mock()
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    self.call_snapshot(fsdp_devices=2, batch_size=value)
                self.make_mesh.assert_not_called()

    def test_missing_devices_is_rejected_before_mesh_creation(self) -> None:
        self.devices = []
        self.make_mesh.reset_mock()
        with self.assertRaisesRegex(ValueError, "at least one JAX device"):
            self.call_snapshot(fsdp_devices=1, batch_size=1)
        self.make_mesh.assert_not_called()


class TrainingSafetyGuardTests(unittest.TestCase):
    def test_nonfinite_metrics_and_zero_first_gradient_stop(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "_FiniteOutputGuard")
        module = ast.Module(body=[node], type_ignores=[])
        namespace = {"io": io, "re": re}
        exec(compile(module, str(SCRIPT), "exec"), namespace)
        guard_class = namespace["_FiniteOutputGuard"]
        guard = guard_class(io.StringIO(), require_initial_update=True)
        guard.write("Step 0: loss=0.1, grad_norm=0.2, param_norm=3.0\n")
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            guard.write("Step 0: loss=nan, grad_norm=0.2, param_norm=3.0\n")
        with self.assertRaisesRegex(RuntimeError, "zero gradient"):
            guard.write("Step 0: loss=0.1, grad_norm=0.0, param_norm=3.0\n")


if __name__ == "__main__":
    unittest.main()
