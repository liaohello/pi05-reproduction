"""Fail-closed contracts for the independent LIBERO pi05 training line."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "libero"))

import libero_data
import train_libero
import build_libero_scene_split
import audit_libero_rlds
import compare_libero_state_digests


def canonical_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class LiberoTrainingProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = ROOT / "configs" / "libero" / "s2_training.json"
        self.document = json.loads(self.path.read_text(encoding="utf-8"))
        self.sha = train_libero.sha256_file(self.path)

    def load(self, document=None):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.json"
            path.write_text(json.dumps(document or self.document), encoding="utf-8")
            return train_libero.load_protocol(path, train_libero.sha256_file(path))

    def test_registered_s2_protocol_is_exact(self) -> None:
        protocol = train_libero.load_protocol(self.path, self.sha)
        self.assertEqual(
            protocol["protocol_id"], "libero-pi05-goal8-s2-v5-validation")
        self.assertEqual(protocol["dataset_task_index"], 10)
        self.assertEqual(protocol["training"]["global_batch_size"], 8)
        self.assertEqual(protocol["training"]["optimizer_updates"], 25000)
        self.assertEqual(protocol["data"]["action_horizon"], 25)
        self.assertEqual(protocol["training"]["ema_decay"], 0.999)
        self.assertTrue(protocol["model"]["freeze_vision_encoder"])
        self.assertEqual(
            train_libero.checkpoint_steps(protocol),
            (500, 1000, 1500, 2000, 3000, 6000, 9000,
             12000, 15000, 18000, 21000, 24000, 25000),
        )
        self.assertEqual(protocol["validation"]["expected_samples"], 895)
        self.assertFalse(protocol["validation"]["train_argument"])
        self.assertEqual(protocol["validation"]["gradient_updates"], 0)
        self.assertEqual(protocol["resource"]["minimum_memory_margin_mib"], 10)

    def test_batch_steps_scope_and_checkpoint_reductions_are_rejected(self) -> None:
        mutations = [
            ("training", "global_batch_size", 4),
            ("training", "optimizer_updates", 5000),
            ("training", "peak_lr", 1e-4),
            ("checkpoint", "save_steps", [500, 1000]),
            ("validation", "train_argument", True),
            ("validation", "gradient_updates", 1),
            ("validation", "norm_stats_source", "validation"),
            ("model", "action_expert_variant", "gemma_300m_lora"),
        ]
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(self.document)
                changed[section][key] = value
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "protocol.json"
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        train_libero.load_protocol(path, train_libero.sha256_file(path))

    def test_protocol_hash_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA256"):
            train_libero.load_protocol(self.path, "0" * 64)

    def test_view_action_semantics_matches_protocol_token(self) -> None:
        self.assertEqual(
            train_libero.action_semantics_token("delta end-effector"),
            "delta_end_effector",
        )


class LiberoViewTests(unittest.TestCase):
    def make_view(self, root: Path, *, split="validation") -> dict:
        norm = root / "assets" / "libero_goal8" / "norm_stats.json"
        norm.parent.mkdir(parents=True)
        norm.write_text('{"norm_stats": {}}\n', encoding="utf-8")
        view = {
            "schema_version": 1,
            "suite": "libero_goal",
            "task_id": 8,
            "dataset_task_index": 10,
            "task": "put the bowl on the plate",
            "train_episode_ids": list(range(39)),
            "validation_episode_ids": list(range(39, 49)),
            "norm_stats": {
                "path": "assets/libero_goal8/norm_stats.json",
                "sha256": train_libero.sha256_file(norm),
            },
        }
        if split != "validation":
            view["train_episode_ids"] = list(range(38)) + [39]
        view["view_sha256"] = canonical_sha(view)
        (root / "view.json").write_text(json.dumps(view), encoding="utf-8")
        return view

    def test_view_requires_disjoint_39_10_and_exact_norm_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_view(root)
            loaded = libero_data.load_view(root)
            self.assertEqual(loaded["train_episode_ids"][-1], 38)

            norm = root / "assets" / "libero_goal8" / "norm_stats.json"
            norm.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "norm stats SHA256"):
                libero_data.load_view(root)

    def test_episode_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            view = self.make_view(root)
            view["train_episode_ids"] = list(range(38)) + [39]
            view["validation_episode_ids"] = list(range(38, 48))
            view["view_sha256"] = canonical_sha(
                {key: value for key, value in view.items() if key != "view_sha256"})
            (root / "view.json").write_text(json.dumps(view), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "39/10"):
                libero_data.load_view(root)


class LiberoResumeIdentityTests(unittest.TestCase):
    def identity(self) -> dict:
        return {
            "protocol_sha256": "a" * 64,
            "image_digest": "sha256:" + "b" * 64,
            "runtime": {"openpi_src": {"sha256": "c" * 64}},
            "view_sha256": "d" * 64,
            "norm_stats_sha256": "e" * 64,
            "base_checkpoint_tree": {"sha256": "f" * 64},
            "model": {"scope": "s2"},
            "training": {"global_batch_size": 8},
            "fsdp_devices": 4,
            "data_loader_workers": 8,
        }

    def saved(self, identity: dict) -> dict:
        return {"identity": identity, "identity_sha256": canonical_sha(identity)}

    def test_unchanged_identity_is_accepted(self) -> None:
        identity = self.identity()
        train_libero.validate_saved_identity(self.saved(identity), identity)

    def test_source_data_base_batch_and_scope_changes_are_rejected(self) -> None:
        original = self.identity()
        saved = self.saved(original)
        mutations = [
            ("runtime", {"openpi_src": {"sha256": "0" * 64}}),
            ("view_sha256", "0" * 64),
            ("base_checkpoint_tree", {"sha256": "0" * 64}),
            ("training", {"global_batch_size": 4}),
            ("model", {"scope": "full"}),
            ("fsdp_devices", 2),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                current = copy.deepcopy(original)
                current[key] = value
                with self.assertRaisesRegex(ValueError, "resume identity differs"):
                    train_libero.validate_saved_identity(saved, current)

    def test_tampered_saved_identity_is_rejected(self) -> None:
        identity = self.identity()
        saved = self.saved(identity)
        saved["identity"]["view_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity hash"):
            train_libero.validate_saved_identity(saved, identity)

    def test_initial_and_probe_do_not_accept_step_or_save_overrides(self) -> None:
        source = (ROOT / "scripts" / "libero" / "train_libero.py").read_text(encoding="utf-8")
        self.assertNotIn('"--steps"', source)
        self.assertNotIn('"--save-interval"', source)


class LiberoValidationTests(unittest.TestCase):
    def test_validation_source_is_gradient_free_and_uses_train_false(self) -> None:
        source = inspect.getsource(train_libero.validation_sample_losses)
        self.assertIn("train=False", source)
        self.assertNotIn("value_and_grad", source)
        self.assertNotIn("tx.update", source)

    def test_fixed_validation_plan_covers_each_sample_once_and_pads_final_batch(self) -> None:
        plan = libero_data.fixed_validation_sample_plan(895, 8, 42)
        self.assertEqual(plan["batch_count"], 112)
        self.assertEqual(plan["padded_count"], 896)
        self.assertEqual(plan["padding"], 1)
        self.assertEqual(plan["valid_samples_per_batch"][-1], 7)
        self.assertEqual(plan["_sampler_indices"][:895], list(range(895)))
        self.assertEqual(plan["_sampler_indices"][895:], [0])
        self.assertEqual(sum(plan["valid_samples_per_batch"]), 895)
        again = libero_data.fixed_validation_sample_plan(895, 8, 42)
        self.assertEqual(plan, again)

    def test_weighted_validation_loss_counts_every_valid_sample_once(self) -> None:
        result = train_libero.weighted_validation_loss(
            [[1.0] * 8, [5.0] * 7 + [999.0]],
            [8, 7],
        )
        expected = (8.0 + 35.0) / 15.0
        self.assertAlmostEqual(result["validation_loss"], expected)
        self.assertEqual(result["samples_evaluated"], 15)
        self.assertEqual(result["weight_sum"], 15.0)
        with self.assertRaises(FloatingPointError):
            train_libero.weighted_validation_loss([[1.0, float("nan")]], [2])

    def test_collect_validation_loss_requires_every_planned_batch(self) -> None:
        result = train_libero.collect_validation_loss(
            [[0], [1], [2]],
            [1, 1, 1],
            lambda batch, index: [index + 0.5],
        )
        self.assertEqual(result["batches_evaluated"], 3)
        self.assertEqual(result["samples_evaluated"], 3)
        self.assertEqual(result["validation_loss"], 1.5)
        with self.assertRaisesRegex(ValueError, "returned 2 batches"):
            train_libero.collect_validation_loss(
                [[0], [1]], [1, 1, 1], lambda batch, index: [1.0])

    def test_validation_plan_binds_resume_and_rejects_changes(self) -> None:
        plan = {
            "schema_version": 1,
            "seed": 42,
            "sample_count": 895,
            "sample_order_sha256": "a" * 64,
            "_sampler_indices": list(range(895)) + [0],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registered, digest = train_libero.register_validation_plan(
                root, plan, fresh=True)
            same, same_digest = train_libero.register_validation_plan(
                root, plan, fresh=False)
            self.assertEqual(registered, same)
            self.assertEqual(digest, same_digest)
            changed = copy.deepcopy(plan)
            changed["seed"] = 43
            with self.assertRaisesRegex(ValueError, "differs"):
                train_libero.register_validation_plan(root, changed, fresh=False)

    def test_checkpoint_is_complete_before_validation_runs(self) -> None:
        source = inspect.getsource(train_libero.main)
        save = source.index(
            "train._checkpoints.save_state(manager, state, loader, new_step)")
        wait = source.index("manager.wait_until_finished()", save)
        validate = source.index(
            "validate_saved_checkpoint(new_step, checkpoint_info)", save)
        self.assertLess(save, wait)
        self.assertLess(wait, validate)


class LiberoSceneSplitTests(unittest.TestCase):
    def test_validation_selection_is_whole_family_and_deterministic(self) -> None:
        families = [
            list(range(0, 13)),
            list(range(13, 22)),
            [22, 23, 24],
            [25, 26],
            [27, 28],
            [29, 30],
        ] + [[value] for value in range(31, 49)]
        first, train_first = build_libero_scene_split.choose_validation_families(
            families, 10, 42)
        second, train_second = build_libero_scene_split.choose_validation_families(
            families, 10, 42)
        self.assertEqual(first, second)
        self.assertEqual(train_first, train_second)
        self.assertEqual(sum(len(row) for row in first), 10)
        self.assertEqual(sorted(sum(families, [])), sorted(sum(train_first + first, [])))
        self.assertFalse(set(sum(first, [])) & set(sum(train_first, [])))


class LiberoRldsParserTests(unittest.TestCase):
    def test_crc32c_reference_value(self) -> None:
        self.assertEqual(audit_libero_rlds.crc32c(b"123456789"), 0xE3069283)

    def test_tfrecord_framing_checks_both_crcs(self) -> None:
        payload = b"tensorflow-example-fixture"
        header = struct.pack("<Q", len(payload))
        record = (
            header
            + struct.pack("<I", audit_libero_rlds.masked_crc32c(header))
            + payload
            + struct.pack("<I", audit_libero_rlds.masked_crc32c(payload))
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.tfrecord"
            path.write_bytes(record)
            rows = list(audit_libero_rlds.read_tfrecords(path))
            self.assertEqual(rows, [(0, payload)])
            broken = bytearray(record)
            broken[-1] ^= 1
            path.write_bytes(broken)
            with self.assertRaisesRegex(ValueError, "payload CRC"):
                list(audit_libero_rlds.read_tfrecords(path))


class LiberoStateDigestTests(unittest.TestCase):
    def document(self):
        return {
            "train_state_step": 3,
            "ema_decay": 0.999,
            "sections": {
                section: {"tensor": {
                    "shape": [2],
                    "dtype": "float32",
                    "finite": True,
                    "sha256": "a" * 64,
                }}
                for section in ("params", "opt_state", "ema_params")
            },
        }

    def test_exact_complete_state_digest_passes(self) -> None:
        document = self.document()
        result = compare_libero_state_digests.compare_documents(
            document, copy.deepcopy(document))
        self.assertTrue(result["passed"])
        self.assertEqual(result["cross_process_max_abs"], 0.0)

    def test_step_or_optimizer_hash_mismatch_fails(self) -> None:
        reference = self.document()
        restored = copy.deepcopy(reference)
        restored["train_state_step"] = 2
        result = compare_libero_state_digests.compare_documents(reference, restored)
        self.assertFalse(result["passed"])
        restored = copy.deepcopy(reference)
        restored["sections"]["opt_state"]["tensor"]["sha256"] = "b" * 64
        result = compare_libero_state_digests.compare_documents(reference, restored)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
