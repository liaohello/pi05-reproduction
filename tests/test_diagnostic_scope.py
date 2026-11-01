from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.so101.model_scope import resolve_model_scope


ROOT = Path(__file__).resolve().parents[1]
SO101 = ROOT / "scripts/so101"
LEGACY_SCOPE = {
    "paligemma_variant": "gemma_2b_lora",
    "action_expert_variant": "gemma_300m_lora",
    "freeze_vision_encoder": True,
}
CURRENT_SCOPE = {
    "paligemma_variant": "gemma_2b_lora",
    "action_expert_variant": "gemma_300m",
    "freeze_vision_encoder": False,
}


def extract_function(filename: str, name: str, namespace: dict):
    tree = ast.parse((SO101 / filename).read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return namespace[name]


class DiagnosticScopeTest(unittest.TestCase):
    def fake_builder(self):
        calls = []

        def make_train_config(**kwargs):
            calls.append(kwargs)
            return kwargs

        return calls, SimpleNamespace(make_train_config=make_train_config)

    def test_offline_diagnostic_forwards_saved_scope_to_real_builder_call(self):
        calls, openpi_so101 = self.fake_builder()
        build = extract_function("e31_offline_diag.py", "_make_offline_config", {
            "openpi_so101": openpi_so101,
            "resolve_model_scope": resolve_model_scope,
        })
        result = build({"model_scope": CURRENT_SCOPE}, mode="serve", exp_name="diag")
        self.assertEqual(result, {"mode": "serve", "exp_name": "diag", **CURRENT_SCOPE})
        self.assertEqual(calls, [result])

    def test_state_check_uses_one_scope_for_both_control_modes(self):
        calls, openpi_so101 = self.fake_builder()
        build = extract_function("e31_state_check.py", "_make_state_mode_config", {
            "openpi_so101": openpi_so101,
        })
        off = build("off", CURRENT_SCOPE, mode="serve", exp_name="state_check")
        on = build("on", CURRENT_SCOPE, mode="serve", exp_name="state_check")
        self.assertEqual([call["state_mode"] for call in calls], ["off", "on"])
        self.assertEqual(off["freeze_vision_encoder"], on["freeze_vision_encoder"])
        self.assertEqual(off["action_expert_variant"], "gemma_300m")
        self.assertEqual(on["action_expert_variant"], "gemma_300m")

    def test_state_check_discovers_nearest_checkpoint_ancestor_config(self):
        scope_for = extract_function("e31_state_check.py", "_scope_for_checkpoint", {
            "Path": Path, "json": json, "resolve_model_scope": resolve_model_scope,
        })
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            checkpoint = run / "checkpoints" / "step_10"
            checkpoint.mkdir(parents=True)
            (run / "config_initial.json").write_text(
                json.dumps({"model_scope": CURRENT_SCOPE}), encoding="utf-8")
            (run / "checkpoints" / "config_initial.json").write_text(
                json.dumps({"model_scope": LEGACY_SCOPE}), encoding="utf-8")
            self.assertEqual(scope_for(checkpoint), LEGACY_SCOPE)
            explicit = run / "explicit.json"
            explicit.write_text(json.dumps({"model_scope": CURRENT_SCOPE}), encoding="utf-8")
            self.assertEqual(scope_for(checkpoint, explicit), CURRENT_SCOPE)
        self.assertEqual(scope_for(Path("/no/snapshot/here")), LEGACY_SCOPE)

    def test_selfcheck_forwards_run_scope_to_real_builder_call(self):
        calls, so101 = self.fake_builder()
        build = extract_function("e4t_selfcheck.py", "_make_selfcheck_config", {
            "so101": so101, "resolve_model_scope": resolve_model_scope,
        })
        result = build({"model_scope": CURRENT_SCOPE}, mode="initial", batch_size=8)
        self.assertEqual(calls, [result])
        self.assertEqual(result["action_expert_variant"], "gemma_300m")
        self.assertFalse(result["freeze_vision_encoder"])

    def test_checkpoint_shape_build_uses_requested_variants_without_freeze_kwarg(self):
        calls = []

        class Model:
            def create(self, _key):
                return object()

        def make_model_config(*args, **kwargs):
            calls.append((args, kwargs))
            return Model()

        fake_jax = SimpleNamespace(
            eval_shape=lambda fn: fn(),
            random=SimpleNamespace(key=lambda _seed: object()),
        )
        fake_nnx = SimpleNamespace(state=lambda _model: SimpleNamespace(
            to_pure_dict=lambda: {"shape": "tree"}))
        build = extract_function("inspect_checkpoint.py", "_model_shapes", {
            "openpi_so101": SimpleNamespace(make_model_config=make_model_config),
            "jax": fake_jax,
            "nnx": fake_nnx,
            "resolve_model_scope": resolve_model_scope,
            "_flatten": lambda value: value,
        })
        result = build(CURRENT_SCOPE, "on")
        self.assertEqual(result, {"shape": "tree"})
        self.assertEqual(calls, [(("on",), {
            "paligemma_variant": "gemma_2b_lora",
            "action_expert_variant": "gemma_300m",
        })])

    def test_checkpoint_base_report_records_scope_used_for_shape_comparison(self):
        array = SimpleNamespace(shape=(2, 3))
        params = {"action_in_proj/kernel": array, "action_out_proj/kernel": array}
        shape_calls = []
        model = SimpleNamespace(restore_params=lambda *_args, **_kwargs: params)

        def model_shapes(scope, state_mode):
            shape_calls.append((scope, state_mode))
            return params

        report_base = extract_function("inspect_checkpoint.py", "_base_report", {
            "_model": model,
            "_flatten": lambda value: value,
            "_model_shapes": model_shapes,
            "_tree_sha": lambda _flat: "test-sha",
            "_array_stats": lambda _array: {"shape": [2, 3]},
            "np": SimpleNamespace(asarray=lambda value: value, ndarray=object),
            "LORA": __import__("re").compile(r".*lora.*"),
            "resolve_model_scope": resolve_model_scope,
        })
        report, _flat = report_base(Path("base"), {
            "model_scope": CURRENT_SCOPE, "state_mode": "on"})
        self.assertEqual(report["model_scope"], CURRENT_SCOPE)
        self.assertEqual(report["state_mode"], "on")
        self.assertEqual(shape_calls, [(CURRENT_SCOPE, "on")])


if __name__ == "__main__":
    unittest.main()
