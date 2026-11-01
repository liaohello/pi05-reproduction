"""Clean100 contracts that hold without numpy or the external runtime."""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "so101"))

import clean60_contracts as clean60
import clean100_checks


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def scene_hash(index: int) -> str:
    return hashlib.sha256(f"scene-{index}".encode()).hexdigest()


def episode_rows() -> list[dict]:
    rows = []
    for index in range(100):
        rows.append({
            "episode_index": index,
            "seed": 12000 + index if index < 60 else 13200 + index - 60,
            "scene_sha256": scene_hash(index),
            "split": "train" if index in TRAIN_IDS else "validation",
            "frames": 100,
            "state_shape": [100, 12],
            "action_shape": [100, 12],
            "image_shape": [480, 640, 3],
            "max_state_error_vs_trace": 0.0,
            "max_action_error_vs_trace": 0.0,
            "category": "clean",
            "subgroup": "clean",
            "perturbation_frame": None,
            "eligible_windows": 91,
            "excluded_cross_event_windows": 0,
            "fixed_windows": list(range(16)),
            "event_windows": list(range(16)),
        })
    return rows


TRAIN_IDS = list(range(48)) + list(range(60, 92))
VALIDATION_IDS = list(range(48, 60)) + list(range(92, 100))


def binding(episodes: int = 100) -> dict:
    return {
        "passed": True,
        "episodes_bound": episodes,
        "checks": {
            "object_counts": [4], "routes": ["direct"], "perturbation_free": True,
            "minimum_pairwise_distance_m": {"episodes_below": 0},
        },
    }


def meta_fixture(rows: list[dict]) -> dict:
    protocol_binding = binding(len(rows))
    return {
        "schema_version": 2,
        "episodes": rows,
        "state_action_dim": 12,
        "feature_names": clean60.FEATURE_NAMES,
        "feature_units": clean60.FEATURE_UNITS,
        "cameras": clean60.CAMERAS,
        "action_semantics": clean60.ACTION_SEMANTICS,
        "clean_profile": {
            "category": "clean", "source": clean60.CLEAN_SOURCE,
            "perturbation_free": True, "episodes_bound": len(rows),
            "protocol_binding_sha256": clean60.sha256_json(protocol_binding),
        },
        "view_sha256": hashlib.sha256(b"view").hexdigest(),
    }


def split_fixture() -> dict:
    return {"train": TRAIN_IDS, "validation": VALIDATION_IDS}


def admission_fixture(meta: dict, split: dict) -> dict:
    return {
        "passed": True,
        "data_gate": {"passed": True, "clean_source": clean60.CLEAN_SOURCE},
        "view_sha256": meta["view_sha256"],
        "split_sha256": clean60.sha256_json(split),
        "train_stats_sha256": hashlib.sha256(b"stats").hexdigest(),
        "totals": {"episodes": 100, "train": 80, "validation": 20,
                   "clean": 100, "recovery": 0},
    }


class Clean100ProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(source("configs/so101/clean100_protocol.json"))
        self.clean60 = json.loads(source("configs/so101/clean60_protocol.json"))

    def test_seed_block_is_disjoint_and_registered(self) -> None:
        start, stop = self.protocol["collection"]["direct_seed_range"]
        self.assertEqual((start, stop), (13200, 13999))
        clean_start, clean_stop = self.clean60["collection"]["direct_seed_range"]
        eval_start, eval_stop = self.protocol["evaluation"]["seed_range"]
        self.assertFalse(set(range(start, stop + 1)) & set(range(clean_start, clean_stop + 1)))
        self.assertFalse(set(range(start, stop + 1)) & set(range(eval_start, eval_stop + 1)))
        self.assertEqual(self.protocol["seed_registry"]["new_block"], [start, stop])

    def test_counts_and_immutable_thresholds(self) -> None:
        collection = self.protocol["collection"]
        self.assertEqual((collection["target_successes"], collection["split"]["train"],
                          collection["split"]["validation"]), (40, 32, 8))
        self.assertEqual(self.protocol["merge_target"]["total_episodes"], 100)
        self.assertEqual((self.protocol["merge_target"]["train"],
                          self.protocol["merge_target"]["validation"]), (80, 20))
        layout = self.protocol["scene_layout"]
        self.assertEqual(layout["minimum_pairwise_distance_m"], 0.08)
        self.assertEqual(layout["object_count"], 4)
        self.assertFalse(layout["perturbations_enabled"])
        self.assertFalse(layout["visual_randomization_enabled"])
        for key in ("collection", "control", "observation", "action", "evaluation"):
            if key == "collection":
                continue
            self.assertEqual(self.protocol[key], self.clean60[key], key)


class Clean100SplitTest(unittest.TestCase):
    def rows(self) -> list[dict]:
        rows = []
        for index in range(60):
            rows.append({"episode_index": index, "collection": "old",
                         "split": "train" if index < 48 else "validation"})
        for index in range(40):
            rows.append({"episode_index": 60 + index, "collection": "new",
                         "split": "train" if index < 32 else "validation"})
        old_split = {index: ("train" if index < 48 else "validation")
                     for index in range(60)}
        return rows, old_split

    def test_singleton_families_keep_the_default_split(self) -> None:
        rows, old_split = self.rows()
        families = [[row["episode_index"]] for row in rows]
        assignment, report = clean100_checks.resolve_split(rows, old_split, families, 32, 8)
        self.assertEqual(report["adjustments"], [])
        self.assertEqual(sum(assignment[index] == "train" for index in range(60, 100)), 32)
        self.assertEqual(sum(assignment[index] == "validation" for index in range(60, 100)), 8)

    def test_family_crossing_default_split_moves_as_one_block(self) -> None:
        rows, old_split = self.rows()
        families = [[row["episode_index"]] for row in rows]
        families = [family for family in families if family != [60] and family != [99]]
        families.append([60, 99])
        assignment, report = clean100_checks.resolve_split(rows, old_split, families, 32, 8)
        self.assertEqual(assignment[60], assignment[99])
        self.assertTrue(report["adjustments"])
        self.assertEqual(sum(assignment[index] == "train" for index in range(60, 100)), 32)

    def test_new_family_touching_old_train_is_forced_train(self) -> None:
        rows, old_split = self.rows()
        families = [[row["episode_index"]] for row in rows]
        families = [family for family in families if family != [0] and family != [92]]
        families.append([0, 92])
        assignment, _ = clean100_checks.resolve_split(rows, old_split, families, 32, 8)
        self.assertEqual(assignment[92], "train")
        self.assertEqual(assignment[0], "train")
        self.assertEqual(sum(assignment[index] == "train" for index in range(60, 100)), 32)

    def test_preserved_family_crossing_is_rejected(self) -> None:
        rows, old_split = self.rows()
        families = [[row["episode_index"]] for row in rows]
        families = [family for family in families if family not in ([0], [48])]
        families.append([0, 48])
        with self.assertRaises(ValueError):
            clean100_checks.resolve_split(rows, old_split, families, 32, 8)

    def test_impossible_target_is_rejected(self) -> None:
        rows, old_split = self.rows()
        families = [[row["episode_index"]] for row in rows]
        families = [family for family in families if family not in ([60], [61])]
        families.append([60, 61])
        # every new episode is forced by an old-side family, so no block can move
        for index in range(60, 100):
            families = [family for family in families if family != [index]]
            families.append([index % 60, index])
        with self.assertRaises(ValueError):
            clean100_checks.resolve_split(rows, old_split, families, 32, 8)


class Clean100ViewContractTest(unittest.TestCase):
    def test_100_80_20_is_accepted(self) -> None:
        rows = episode_rows()
        meta = meta_fixture(rows)
        protocol_binding = binding(100)
        split = split_fixture()
        self.assertEqual(clean60.view_contract_problems(
            meta, protocol_binding=protocol_binding,
            expected_counts=(100, 80, 20), require_clean=True), [])
        self.assertEqual(clean60.admission_contract_problems(
            admission_fixture(meta, split), meta=meta, split=split,
            train_stats_sha256=hashlib.sha256(b"stats").hexdigest()), [])

    def test_duplicate_scene_hash_is_rejected(self) -> None:
        rows = episode_rows()
        rows[1]["scene_sha256"] = rows[0]["scene_sha256"]
        problems = clean60.view_contract_problems(
            meta_fixture(rows), protocol_binding=binding(100),
            expected_counts=(100, 80, 20), require_clean=True)
        self.assertTrue(any("scene hashes" in problem for problem in problems), problems)

    def test_missing_camera_is_rejected(self) -> None:
        meta = meta_fixture(episode_rows())
        meta["cameras"] = meta["cameras"][:-1]
        problems = clean60.view_contract_problems(
            meta, protocol_binding=binding(100),
            expected_counts=(100, 80, 20), require_clean=True)
        self.assertTrue(any("cameras" in problem for problem in problems), problems)

    def test_wrong_counts_are_rejected(self) -> None:
        meta = meta_fixture(episode_rows())
        problems = clean60.view_contract_problems(
            meta, protocol_binding=binding(100),
            expected_counts=(60, 48, 12), require_clean=True)
        self.assertTrue(any("counts" in problem for problem in problems), problems)

    def test_clean60_defaults_still_hold(self) -> None:
        rows = episode_rows()[:60]
        for row in rows:
            row["split"] = "train" if row["episode_index"] < 48 else "validation"
        meta = meta_fixture(rows)
        self.assertEqual(clean60.view_contract_problems(
            meta, protocol_binding=binding(60),
            expected_counts=(60, 48, 12), require_clean=True), [])
        # the untouched default call also keeps requiring a clean view
        self.assertEqual(clean60.view_contract_problems(
            meta, protocol_binding=binding(60), require_clean=True), [])


class Clean100TimeAndFiniteTest(unittest.TestCase):
    def test_non_contiguous_frame_indices_are_rejected(self) -> None:
        self.assertTrue(clean100_checks.frame_index_continuous(list(range(5)), 5))
        self.assertFalse(clean100_checks.frame_index_continuous([0, 1, 3, 4, 5], 5))
        self.assertFalse(clean100_checks.frame_index_continuous([0, 1, 2, 3], 5))
        self.assertFalse(clean100_checks.frame_index_continuous([1, 2, 3, 4, 5], 5))

    def test_timestamp_grid_and_alignment(self) -> None:
        stamps = [index / 50.0 for index in range(5)]
        self.assertTrue(clean100_checks.timestamps_nominal(stamps, 5))
        shifted = [value + 0.02 for value in stamps]
        self.assertFalse(clean100_checks.timestamps_nominal(shifted, 5))
        gaps = [0.0, 0.02, 0.05, 0.06, 0.08]
        self.assertFalse(clean100_checks.timestamps_nominal(gaps, 5))
        self.assertTrue(clean100_checks.control_clock_aligned(stamps, stamps, 5))
        self.assertFalse(clean100_checks.control_clock_aligned(
            [0.0, 0.02, 0.04, 0.06, 0.09], stamps, 5))
        self.assertTrue(clean100_checks.control_clock_regular(stamps, 5))

    def test_non_finite_values_are_rejected(self) -> None:
        self.assertTrue(clean100_checks.all_finite([0.0, 1.0, [[2.0]]]))
        self.assertFalse(clean100_checks.all_finite([0.0, float("nan")]))
        self.assertFalse(clean100_checks.all_finite([float("inf")]))
        self.assertFalse(clean100_checks.timestamps_nominal(
            [0.0, float("nan"), 0.04], 3))


class Clean100AuditContractsTest(unittest.TestCase):
    def test_scene_and_data_contracts_accept_100_80_20(self) -> None:
        digest = hashlib.sha256(b"view").hexdigest()
        summary = {
            "episodes": 100, "train": 80, "validation": 20,
            "unique_scene_hashes": 100,
            "near_duplicate_families_crossing_split": [],
            "all_four_objects_per_episode": True,
            "all_100_scenes_reconstructed_hash_exact": True,
            "all_object_ik_records_complete": True,
            "all_region_keepout_checks_passed": True,
            "all_original_object_predicates_passed": True,
            "all_physical_success_rows": True,
            "all_physical_replay_exact": True,
            "attempts": 300, "successes": 100, "failures": 200,
        }
        scene = {"passed": True, "training_view_sha256": digest, "summary": summary}
        self.assertEqual(clean60.scene_audit_problems(
            scene, view_sha256=digest, expected_counts=(100, 80, 20),
            expected_attempts=(300, 100, 200),
            scene_key="all_100_scenes_reconstructed_hash_exact"), [])
        self.assertTrue(clean60.scene_audit_problems(
            scene, view_sha256=digest, expected_counts=(60, 48, 12),
            expected_attempts=(175, 60, 115)))
        rows = [{"split": "train" if index < 80 else "validation",
                 "state_action_trace_exact": True, "strict_predicate_passed": True}
                for index in range(100)]
        data = {
            "passed": True, "view_sha256": digest,
            "attempts": {"total": 300, "successes": 100, "failures": 200},
            "images_decoded": True, "actions_reapplied": True, "episodes": rows,
        }
        self.assertEqual(clean60.data_audit_problems(
            data, view_sha256=digest, expected_counts=(100, 80, 20),
            expected_attempts=(300, 100, 200)), [])
        data["episodes"][0]["state_action_trace_exact"] = False
        self.assertTrue(clean60.data_audit_problems(
            data, view_sha256=digest, expected_counts=(100, 80, 20),
            expected_attempts=(300, 100, 200)))

    def test_norm_audit_binds_80_20_and_rejects_hash_drift(self) -> None:
        digest = hashlib.sha256(b"x").hexdigest()
        stats = {"mean": [0.0] * 12, "std": [1.0] * 12,
                 "q01": [-1.0] * 12, "q99": [1.0] * 12}
        audit = {
            "schema_version": 3, "passed": True,
            "view_sha256": digest, "norm_stats_sha256": digest,
            "base_checkpoint_sha256": digest, "training_config_sha256": digest,
            "binding": {"train_episodes": 80, "validation_episodes": 20},
            "statistics": {
                "recomputed_from_train_only_matches_view": True,
                "all_values_finite": True,
                "near_constant_dims_std_lt_1e-6": {"state": [], "actions": []},
            },
            "transform": {
                "internal_action_dim": 32, "worst_recovery_error": 0.0,
                "pad_first_12_max_abs_error": 0.0, "padded_tail_max_abs": 0.0,
                "clipping_applied": False,
                "inference_output_execution_order_verified": True,
            },
            "loader_consistency": {
                "episodes_covered": 80,
                "action_max_abs_error_vs_runtime_pipeline": 0.0,
                "state_max_abs_error_vs_runtime_pipeline": 0.0,
            },
            "semantics": {"feature_units": clean60.FEATURE_UNITS,
                          "action_horizon": 10, "action_shape": [10, 12]},
        }
        expected = {"view_sha256": digest, "norm_stats_sha256": digest,
                    "base_checkpoint_sha256": digest, "training_config_sha256": digest,
                    "expected_train": 80, "expected_validation": 20}
        self.assertEqual(clean60.norm_audit_problems(audit, **expected), [])
        wrong = dict(expected, norm_stats_sha256="0" * 64)
        self.assertTrue(clean60.norm_audit_problems(audit, **wrong))
        wrong = dict(expected, expected_train=48, expected_validation=12)
        self.assertTrue(clean60.norm_audit_problems(audit, **wrong))

    def test_stats_recompute_requires_the_registered_train_ids(self) -> None:
        digest = hashlib.sha256(b"stats").hexdigest()
        report = {
            "passed": True, "view_sha256": hashlib.sha256(b"v").hexdigest(),
            "new_norm_stats_sha256": digest,
            "train_episodes": TRAIN_IDS,
            "validation_episodes_excluded": VALIDATION_IDS,
            "near_constant_dimensions_std_lt_1e-6": {"state": [], "actions": []},
        }
        args = {"norm_stats_sha256": digest,
                "view_sha256": report["view_sha256"],
                "expected_train_ids": TRAIN_IDS,
                "expected_validation_ids": VALIDATION_IDS}
        self.assertEqual(clean60.stats_recompute_problems(report, **args), [])
        wrong = dict(args, norm_stats_sha256="0" * 64)
        self.assertTrue(clean60.stats_recompute_problems(report, **wrong))
        wrong = dict(args, expected_validation_ids=list(range(48, 60)))
        self.assertTrue(clean60.stats_recompute_problems(report, **wrong))


if __name__ == "__main__":
    unittest.main()
