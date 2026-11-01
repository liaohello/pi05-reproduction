"""Static contracts for the v4 narrowed feasible family and 24-episode pilot."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_PROTOCOL = ROOT / "configs/so101/scene2_dual_random_protocol.json"
V2_PROTOCOL = ROOT / "configs/so101/scene2_dual_random_protocol_v2.json"
V3_PROTOCOL = ROOT / "configs/so101/scene2_dual_random_protocol_v3.json"
V4_PROTOCOL = ROOT / "configs/so101/scene2_dual_random_protocol_v4.json"
V4_PROFILE = ROOT / "configs/so101/scene2_dual_random_profile_v4.yaml"
V4_MODULE = ROOT / "scripts/so101/scene2_common_v4.py"
V4_REPORTS = sorted((ROOT / "reports").glob("so101-scene2-dual-random-v4*.md"))


class Scene2V4ProtocolContractTest(unittest.TestCase):
    def setUp(self):
        self.v1 = json.loads(V1_PROTOCOL.read_text(encoding="utf-8"))
        self.v2 = json.loads(V2_PROTOCOL.read_text(encoding="utf-8"))
        self.v3 = json.loads(V3_PROTOCOL.read_text(encoding="utf-8"))
        self.v4 = json.loads(V4_PROTOCOL.read_text(encoding="utf-8"))

    def test_prior_records_untouched(self):
        self.assertEqual(self.v1["protocol_id"], "scene2_dual_random_v1")
        self.assertEqual(self.v2["protocol_id"], "scene2_dual_random_v2")
        self.assertEqual(self.v3["protocol_id"], "scene2_dual_random_v3")

    def test_v4_rules(self):
        self.assertEqual(self.v4["protocol_id"], "scene2_dual_random_v4")
        self.assertEqual(self.v4["source_binding"]["profile"],
                         "configs/so101/scene2_dual_random_profile_v4.yaml")
        self.assertEqual(self.v4["source_binding"]["certificate"],
                         "configs/certification/scene2_dual_random_certificate_v4.json")
        layout = self.v4["scene_layout"]
        self.assertEqual(layout["blocks"]["yaw"]["domain_degrees"], [33.0, 67.0])
        self.assertEqual(layout["blocks"]["block_tray_outer_clearance_m"], 0.105)
        self.assertEqual(layout["trays"]["positive_y_sampling"]["y_m"], [0.263, 0.274])
        self.assertEqual(layout["trays"]["negative_y_sampling"]["y_m"], [-0.274, -0.235])
        self.assertEqual(layout["trays"]["positive_y_sampling"]["x_m"], [0.214, 0.270])
        self.assertEqual(layout["trays"]["slot_rule"], "v1 external two-slot rule (slot_0 then slot_1)")
        self.assertTrue(layout["family_basis"]["retrospective"])
        pilot = self.v4["collection"]["physical_pilot"]
        self.assertEqual(pilot["layouts"], 12)
        self.assertEqual(pilot["episodes"], 24)
        self.assertEqual(pilot["episode_attempt_cap"], 24)
        self.assertTrue(pilot["pilot_is_not_formal_validation"])

    def test_success_predicate_unchanged(self):
        for key in ("requires_all_target", "requires_all_distractor", "not_sufficient"):
            self.assertEqual(self.v1["success_predicate"][key],
                             self.v4["success_predicate"][key])
        self.assertEqual(self.v1["success_predicate"]["id"],
                         self.v4["success_predicate"]["id"])

    def test_v4_profile(self):
        import yaml

        profile = yaml.safe_load(V4_PROFILE.read_text(encoding="utf-8"))
        self.assertEqual(profile["sorting"]["yaw_domain_degrees"], [33.0, 67.0])
        self.assertEqual(profile["sorting"]["tray_sampling_positive"]["y_m"], [0.263, 0.274])
        self.assertEqual(profile["sorting"]["tray_sampling_negative"]["y_m"], [-0.274, -0.235])
        self.assertEqual(profile["sorting"]["scene2_block_tray_outer_clearance_m"], 0.105)
        self.assertEqual(profile["sorting"]["scene2_slot_mode"], "external")
        self.assertEqual(profile["grasp"]["open_percent"], 70.0)
        self.assertEqual(profile["sorting"]["distractor_max_displacement_m"], 0.002)
        for key in ("scene2_release_open_candidates", "scene2_vertical_withdraw",
                    "scene2_close_screen", "scene2_release_open_percent"):
            self.assertNotIn(key, profile["grasp"])


class Scene2V4AdapterContractTest(unittest.TestCase):
    def setUp(self):
        self.source = V4_MODULE.read_text(encoding="utf-8")

    def test_wrappers_and_v1_reference(self):
        for name in ("certify", "layout_pilot", "pregenerate", "collect", "audit"):
            self.assertTrue((ROOT / f"scripts/so101/scene2_{name}_v4.py").is_file(), name)
        self.assertTrue((ROOT / "scripts/so101/scene2_settle_baseline_v4.py").is_file())
        self.assertIn("_v2.configure(root, python=python)", self.source)
        self.assertIn("_v1.measure_initial = measure_initial", self.source)
        self.assertIn("parked_arms_clear_of_trays", self.source)
        self.assertIn("parked_arm_tray_clearance_m", self.source)

    def test_v4_layout_baseline_scenes(self):
        layout = ROOT / "experiments/so101/scene2-dual-random-v4-layout-pilot.json"
        baseline = ROOT / "experiments/so101/scene2-dual-random-v4-baseline.json"
        scenes = ROOT / "experiments/so101/scene2-dual-random-v4-pilot-scenes.json"
        self.assertTrue(layout.is_file())
        self.assertTrue(baseline.is_file())
        self.assertTrue(scenes.is_file())
        layout_data = json.loads(layout.read_text(encoding="utf-8"))
        baseline_data = json.loads(baseline.read_text(encoding="utf-8"))
        scenes_data = json.loads(scenes.read_text(encoding="utf-8"))
        self.assertEqual(layout_data["accepted"], 240)
        self.assertEqual(layout_data["unique_physical_hashes"], 240)
        self.assertTrue(layout_data["all_accepted_initial_checks_passed"])
        self.assertEqual(baseline_data["clean_run"]["scenes_measured"], 12)
        self.assertEqual(baseline_data["clean_run"]["threshold_m"], 0.002)
        self.assertEqual(len(scenes_data["layouts"]), 12)
        self.assertEqual(sum(len(value) for value in scenes_data["scenes"].values()), 24)
        for layout_row in scenes_data["layouts"]:
            for block in layout_row["blocks"]:
                self.assertGreaterEqual(block["yaw_degrees"], 33.0)
                self.assertLessEqual(block["yaw_degrees"], 67.0)

    def test_v4_pilot_and_audit(self):
        pilot = json.loads((ROOT / "experiments/so101/scene2-dual-random-v4-pilot.json").read_text(encoding="utf-8"))
        audit = json.loads((ROOT / "experiments/so101/scene2-dual-random-v4-audit.json").read_text(encoding="utf-8"))
        self.assertEqual(pilot["layouts"], 12)
        self.assertEqual(pilot["episodes_executed"], 24)
        self.assertEqual(pilot["successes"], 19)
        self.assertEqual(pilot["layout_successes_both_conditions"], 7)
        self.assertEqual(pilot["pair_initial_fingerprint_matches"], 12)
        self.assertFalse(pilot["gate"]["met"])
        self.assertEqual(audit["episodes"], 24)
        self.assertEqual(audit["initial_pair_matches"], 12)
        self.assertTrue(audit["passed"])

    def test_v4_reports_and_private_paths(self):
        self.assertTrue(V4_REPORTS, "v4 reports are missing")
        for path in V4_REPORTS:
            text = path.read_text(encoding="utf-8")
            self.assertLess(len(text), 1000, path.name)
            for token in ("/" + "home" + "/", "Desk" + "top", "cache/so101", "runs/so101"):
                self.assertNotIn(token, text, path.name)
        paths = [V4_PROTOCOL, V4_PROFILE, V4_MODULE,
                 ROOT / "docs/so101-scene2-dual-random-v4.md"]
        paths += sorted((ROOT / "experiments/so101").glob("scene2-dual-random-v4-*.json"))
        for path in paths:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for token in ("/" + "home" + "/", "Desk" + "top", "cache/so101", "runs/so101"):
                self.assertNotIn(token, text, path.name)


if __name__ == "__main__":
    unittest.main()
