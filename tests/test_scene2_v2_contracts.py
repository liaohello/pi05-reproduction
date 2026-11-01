"""Static contracts for the v2 scene2 clearance probe (no simulator)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_PROTOCOL = ROOT / "configs/so101/scene2_dual_random_protocol.json"
V2_PROTOCOL = ROOT / "configs/so101/scene2_dual_random_protocol_v2.json"
V2_PROFILE = ROOT / "configs/so101/scene2_dual_random_profile_v2.yaml"
V2_MODULE = ROOT / "scripts/so101/scene2_common_v2.py"
V2_REPORTS = sorted((ROOT / "reports").glob("so101-scene2-dual-random-v2*.md"))


class Scene2V2ProtocolContractTest(unittest.TestCase):
    def setUp(self):
        self.v1 = json.loads(V1_PROTOCOL.read_text(encoding="utf-8"))
        self.v2 = json.loads(V2_PROTOCOL.read_text(encoding="utf-8"))

    def test_v1_is_untouched(self):
        self.assertEqual(self.v1["protocol_id"], "scene2_dual_random_v1")
        self.assertEqual(self.v1["status"], "layout_pilot_passed_pending_physical_pilot")
        self.assertEqual(self.v1["collection"]["physical_pilot"]["layouts"], 4)
        self.assertEqual(self.v1["collection"]["physical_pilot"]["episodes"], 8)

    def test_v2_rules(self):
        self.assertEqual(self.v2["protocol_id"], "scene2_dual_random_v2")
        self.assertEqual(self.v2["source_binding"]["profile"],
                         "configs/so101/scene2_dual_random_profile_v2.yaml")
        self.assertEqual(self.v2["source_binding"]["certificate"],
                         "configs/certification/scene2_dual_random_certificate_v2.json")
        self.assertEqual(self.v2["scene_layout"]["blocks"]["block_tray_outer_clearance_m"], 0.07)
        self.assertEqual(self.v2["scene_layout"]["trays"]["slot_rule"].split(";")[0],
                         "one central slot per tray at the tray centre")
        grasp = self.v2["grasp"]
        self.assertEqual(grasp["release_open_percent"], 50.0)
        self.assertTrue(grasp["vertical_withdraw"])
        self.assertTrue(grasp["close_screen"])
        self.assertEqual(self.v2["collection"]["seed_registry"]["layout_pilot"], [22000, 22239])
        self.assertEqual(self.v2["collection"]["seed_registry"]["physical_pilot_layouts"], [22240, 22279])

    def test_success_predicate_is_unchanged(self):
        for key in ("requires_all_target", "requires_all_distractor", "not_sufficient"):
            self.assertEqual(self.v1["success_predicate"][key],
                             self.v2["success_predicate"][key])
        self.assertEqual(self.v1["success_predicate"]["id"],
                         self.v2["success_predicate"]["id"])

    def test_v2_profile(self):
        import yaml

        profile = yaml.safe_load(V2_PROFILE.read_text(encoding="utf-8"))
        self.assertEqual(profile["sorting"]["scene2_slot_mode"], "center")
        self.assertEqual(profile["sorting"]["scene2_block_tray_outer_clearance_m"], 0.07)
        self.assertEqual(profile["grasp"]["scene2_release_open_percent"], 50.0)
        self.assertTrue(profile["grasp"]["scene2_vertical_withdraw"])
        self.assertTrue(profile["grasp"]["scene2_close_screen"])
        self.assertEqual(profile["sorting"]["distractor_max_displacement_m"], 0.002)
        self.assertFalse(profile["domain_randomization"]["enabled"])


class Scene2V2AdapterContractTest(unittest.TestCase):
    def setUp(self):
        self.source = V2_MODULE.read_text(encoding="utf-8")

    def test_wrapper_keeps_v1_files(self):
        for name in ("certify", "layout_pilot", "settle_baseline", "pregenerate",
                     "collect", "audit"):
            self.assertTrue((ROOT / f"scripts/so101/scene2_{name}_v2.py").is_file(), name)
        self.assertIn("_ORIGINAL_BLOCK_POSITION", self.source)
        self.assertIn("_ORIGINAL_PICK_VERDICT", self.source)
        self.assertIn("_ORIGINAL_PLACE_VERDICT", self.source)

    def test_v2_clearance_features(self):
        for token in ("scene2_slot_mode", "tray_outer_clearance",
                      "scene2_close_screen", "scene2_vertical_withdraw",
                      "scene2_release_open_percent", "build_release",
                      "build_vertical", "retreat_xy"):
            self.assertIn(token, self.source)
        self.assertIn("regions.tray_slots = _tray_slots", self.source)
        self.assertIn("_v1.dual_block_expert_class = dual_block_expert_class", self.source)

    def test_v2_layout_and_baseline_summaries(self):
        layout = ROOT / "experiments/so101/scene2-dual-random-v2-layout-pilot.json"
        baseline = ROOT / "experiments/so101/scene2-dual-random-v2-baseline.json"
        scenes = ROOT / "experiments/so101/scene2-dual-random-v2-pilot-scenes.json"
        self.assertTrue(layout.is_file())
        self.assertTrue(baseline.is_file())
        self.assertTrue(scenes.is_file())
        layout_data = json.loads(layout.read_text(encoding="utf-8"))
        baseline_data = json.loads(baseline.read_text(encoding="utf-8"))
        scenes_data = json.loads(scenes.read_text(encoding="utf-8"))
        self.assertEqual(layout_data["accepted"], 240)
        self.assertEqual(layout_data["unique_physical_hashes"], 240)
        self.assertTrue(layout_data["all_accepted_have_both_conditions"])
        self.assertEqual(baseline_data["clean_run"]["threshold_m"], 0.002)
        self.assertEqual(len(scenes_data["layouts"]), 4)
        self.assertEqual(sum(len(value) for value in scenes_data["scenes"].values()), 8)
        self.assertEqual(len({row["physical_sha256"] for row in scenes_data["layouts"]}), 4)

    def test_v2_reports_are_short_and_clean(self):
        self.assertTrue(V2_REPORTS, "v2 reports are missing")
        for path in V2_REPORTS:
            text = path.read_text(encoding="utf-8")
            self.assertLess(len(text), 1000, path.name)
            for token in ("/" + "home" + "/", "Desk" + "top", "cache/so101", "runs/so101"):
                self.assertNotIn(token, text, path.name)
        docs = [ROOT / "docs/so101-scene2-dual-random-v2.md"]
        for path in docs:
            if path.is_file():
                self.assertLess(len(path.read_text(encoding="utf-8")), 1000, path.name)

    def test_v2_tracked_files_have_no_private_paths(self):
        paths = [V2_PROTOCOL, V2_PROFILE, V2_MODULE, ROOT / "docs/so101-scene2-dual-random-v2.md"]
        paths += sorted((ROOT / "experiments/so101").glob("scene2-dual-random-v2-*.json"))
        paths += V2_REPORTS
        for path in paths:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for token in ("/" + "home" + "/", "Desk" + "top", "cache/so101", "runs/so101"):
                self.assertNotIn(token, text, path.name)


if __name__ == "__main__":
    unittest.main()
