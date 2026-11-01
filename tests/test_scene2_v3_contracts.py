"""Static contracts for the v3 per-scene release/withdraw search."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_PROTOCOL = ROOT / "configs/so101/scene2_dual_random_protocol.json"
V2_PROTOCOL = ROOT / "configs/so101/scene2_dual_random_protocol_v2.json"
V3_PROTOCOL = ROOT / "configs/so101/scene2_dual_random_protocol_v3.json"
V3_PROFILE = ROOT / "configs/so101/scene2_dual_random_profile_v3.yaml"
V3_MODULE = ROOT / "scripts/so101/scene2_common_v3.py"
V3_REPORTS = sorted((ROOT / "reports").glob("so101-scene2-dual-random-v3*.md"))


class Scene2V3ProtocolContractTest(unittest.TestCase):
    def setUp(self):
        self.v1 = json.loads(V1_PROTOCOL.read_text(encoding="utf-8"))
        self.v2 = json.loads(V2_PROTOCOL.read_text(encoding="utf-8"))
        self.v3 = json.loads(V3_PROTOCOL.read_text(encoding="utf-8"))

    def test_v1_v2_untouched(self):
        self.assertEqual(self.v1["protocol_id"], "scene2_dual_random_v1")
        self.assertEqual(self.v2["protocol_id"], "scene2_dual_random_v2")
        self.assertEqual(self.v1["collection"]["physical_pilot"]["layouts"], 4)
        self.assertEqual(self.v2["collection"]["physical_pilot"]["layouts"], 4)

    def test_v3_rules(self):
        self.assertEqual(self.v3["protocol_id"], "scene2_dual_random_v3")
        self.assertEqual(self.v3["source_binding"]["profile"],
                         "configs/so101/scene2_dual_random_profile_v3.yaml")
        self.assertEqual(self.v3["source_binding"]["certificate"],
                         "configs/certification/scene2_dual_random_certificate_v3.json")
        self.assertEqual(self.v3["scene_layout"]["blocks"]["block_tray_outer_clearance_m"], 0.10)
        grasp = self.v3["grasp"]
        self.assertEqual(grasp["release_open_candidates"], [40.0, 50.0, 30.0])
        self.assertEqual(grasp["withdraw_directions"],
                         ["gripper_x_neg", "gripper_x_pos", "world_x_neg", "world_x_pos"])
        self.assertEqual(grasp["withdraw_retreat_m"], 0.06)
        self.assertEqual(self.v3["collection"]["seed_registry"]["layout_pilot"], [23200, 23439])
        self.assertEqual(self.v3["collection"]["seed_registry"]["physical_pilot_layouts"],
                         [23440, 23479])

    def test_success_predicate_unchanged(self):
        for key in ("requires_all_target", "requires_all_distractor", "not_sufficient"):
            self.assertEqual(self.v1["success_predicate"][key],
                             self.v3["success_predicate"][key])
        self.assertEqual(self.v1["success_predicate"]["id"],
                         self.v3["success_predicate"]["id"])

    def test_v3_profile(self):
        import yaml

        profile = yaml.safe_load(V3_PROFILE.read_text(encoding="utf-8"))
        self.assertEqual(profile["sorting"]["scene2_block_tray_outer_clearance_m"], 0.10)
        self.assertEqual(profile["sorting"]["scene2_slot_mode"], "center")
        self.assertEqual(profile["grasp"]["scene2_release_open_candidates"], [40.0, 50.0, 30.0])
        self.assertEqual(profile["grasp"]["scene2_withdraw_retreat_m"], 0.06)
        self.assertEqual(profile["sorting"]["distractor_max_displacement_m"], 0.002)


class Scene2V3AdapterContractTest(unittest.TestCase):
    def setUp(self):
        self.source = V3_MODULE.read_text(encoding="utf-8")

    def test_wrappers_and_overrides(self):
        for name in ("certify", "layout_pilot", "pregenerate", "collect", "audit"):
            self.assertTrue((ROOT / f"scripts/so101/scene2_{name}_v3.py").is_file(), name)
        self.assertTrue((ROOT / "scripts/so101/scene2_settle_baseline_v3.py").is_file())
        self.assertIn("_OPEN_CANDIDATES = (40.0, 50.0, 30.0)", self.source)
        self.assertIn("_DIRECTIONS =", self.source)
        self.assertIn("_v1.place_verdict = place_verdict", self.source)
        self.assertIn("_v1.dual_block_expert_class = dual_block_expert_class", self.source)

    def test_per_scene_search_is_present(self):
        for token in ("release_checks", "chosen_open", "direction_checks",
                      "chosen_direction", "build_release", "build_vertical",
                      "state[\"open\"]", "state[\"direction\"]"):
            self.assertIn(token, self.source)

    def test_v3_layout_baseline_scenes(self):
        layout = ROOT / "experiments/so101/scene2-dual-random-v3-layout-pilot.json"
        baseline = ROOT / "experiments/so101/scene2-dual-random-v3-baseline.json"
        scenes = ROOT / "experiments/so101/scene2-dual-random-v3-pilot-scenes.json"
        self.assertTrue(layout.is_file())
        self.assertTrue(baseline.is_file())
        self.assertTrue(scenes.is_file())
        layout_data = json.loads(layout.read_text(encoding="utf-8"))
        baseline_data = json.loads(baseline.read_text(encoding="utf-8"))
        scenes_data = json.loads(scenes.read_text(encoding="utf-8"))
        self.assertEqual(layout_data["accepted"], 240)
        self.assertEqual(layout_data["unique_physical_hashes"], 240)
        self.assertEqual(baseline_data["clean_run"]["scenes_measured"], 12)
        self.assertEqual(baseline_data["clean_run"]["threshold_m"], 0.002)
        self.assertEqual(len(scenes_data["layouts"]), 4)
        self.assertEqual(sum(len(value) for value in scenes_data["scenes"].values()), 8)

    def test_v3_reports_and_private_paths(self):
        self.assertTrue(V3_REPORTS, "v3 reports are missing")
        for path in V3_REPORTS:
            text = path.read_text(encoding="utf-8")
            self.assertLess(len(text), 1000, path.name)
            for token in ("/" + "home" + "/", "Desk" + "top", "cache/so101", "runs/so101"):
                self.assertNotIn(token, text, path.name)
        paths = [V3_PROTOCOL, V3_PROFILE, V3_MODULE,
                 ROOT / "docs/so101-scene2-dual-random-v3.md"]
        paths += sorted((ROOT / "experiments/so101").glob("scene2-dual-random-v3-*.json"))
        for path in paths:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for token in ("/" + "home" + "/", "Desk" + "top", "cache/so101", "runs/so101"):
                self.assertNotIn(token, text, path.name)


if __name__ == "__main__":
    unittest.main()
