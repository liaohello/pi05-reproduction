"""Exercise the probe's real argument wiring without importing JAX or using GPUs."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


PROBE = (Path(__file__).resolve().parents[1] / "scripts" / "so101"
         / "clean60_probes" / "memory_shard.py")


class GradientArgumentContractTests(unittest.TestCase):
    def setUp(self) -> None:
        # Execute the actual nested function, JIT construction and call expressions.
        # Distinct sentinels expose rng/state swaps before a device is allocated.
        self.tree = ast.parse(PROBE.read_text(encoding="utf-8"))
        self.rng, self.grads, self.loss, self.norm = (object() for _ in range(4))
        self.state = SimpleNamespace(model_def=object(), params=object())
        self.observation, self.actions = object(), object()
        self.batch = (self.observation, self.actions)
        self.model = SimpleNamespace(train=Mock(), compute_loss=Mock(return_value=self.loss))
        self.merge = Mock(return_value=self.model)
        self.filter, self.diff = object(), object()
        self.diff_state = Mock(return_value=self.diff)

        def value_and_grad(fn, *, argnums):
            self.assertIs(argnums, self.diff)
            return lambda *args: (fn(*args), self.grads)

        def jit(fn, *, in_shardings, out_shardings):
            self.assertEqual(in_shardings, ("replicated", "state", "batch"))
            return fn

        self.env = {
            "nnx": SimpleNamespace(merge=self.merge, DiffState=self.diff_state,
                                   value_and_grad=value_and_grad),
            "jnp": SimpleNamespace(mean=lambda value: value),
            "optax": SimpleNamespace(global_norm=lambda grads: self.norm),
            "jax": SimpleNamespace(
                jit=jit, eval_shape=lambda fn, *args: fn(*args),
                random=SimpleNamespace(key=lambda seed: self.rng,
                                       fold_in=lambda rng, step: rng)),
            "config": SimpleNamespace(seed=42),
            "state": self.state, "batch": self.batch,
            "trainable_filter": self.filter,
            "replicated_sharding": "replicated", "state_sharding": "state",
            "data_sharding": "batch",
        }
        definitions = [node for node in ast.walk(self.tree)
                       if isinstance(node, ast.FunctionDef) and node.name == "grad_step"]
        bindings = [node for node in ast.walk(self.tree)
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "pgrad"
                            for target in node.targets)]
        self.assertEqual(len(definitions), 1)
        self.assertEqual(len(bindings), 1)
        module = ast.Module(body=definitions + bindings, type_ignores=[])
        exec(compile(module, PROBE.name, "exec"), self.env)

    def check_call(self, *, shape_only: bool) -> None:
        calls = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)
                 and ((shape_only and isinstance(node.func, ast.Attribute)
                       and isinstance(node.func.value, ast.Name)
                       and node.func.value.id == "jax" and node.func.attr == "eval_shape")
                      or (not shape_only and isinstance(node.func, ast.Name)
                          and node.func.id == "pgrad"))]
        self.assertEqual(len(calls), 1)
        info, grads = eval(compile(ast.Expression(calls[0]), PROBE.name, "eval"), self.env)
        self.merge.assert_called_once_with(self.state.model_def, self.state.params)
        self.model.train.assert_called_once_with()
        self.model.compute_loss.assert_called_once_with(
            self.rng, self.observation, self.actions, train=True)
        self.diff_state.assert_called_once_with(0, self.filter)
        self.assertIs(info["loss"], self.loss)
        self.assertIs(info["grad_norm"], self.norm)
        self.assertIs(grads, self.grads)

    def test_shape_trace_receives_rng_before_state(self) -> None:
        self.check_call(shape_only=True)

    def test_compiled_call_receives_rng_before_state(self) -> None:
        self.check_call(shape_only=False)


if __name__ == "__main__":
    unittest.main()
