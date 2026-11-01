"""Static contracts for workstream C scene2 (no external simulator needed)."""
from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/so101/scene2_dual_random_protocol.json"
PROFILE = ROOT / "configs/so101/scene2_dual_random_profile.yaml"
ADAPTER = ROOT / "scripts/so101/scene2_common.py"
COLLECTOR = ROOT / "scripts/so101/scene2_collect.py"
AUDIT = ROOT / "scripts/so101/scene2_audit.py"
BASELINE = ROOT / "scripts/so101/scene2_settle_baseline.py"
CERTIFY = ROOT / "scripts/so101/scene2_certify.py"


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _class(path: Path, name: str) -> ast.ClassDef:
    return next(node for node in _module(path).body
                if isinstance(node, ast.ClassDef) and node.name == name)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class Scene2ProtocolContractTest(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    def test_task_and_instructions(self):
        task = self.protocol["task"]
        self.assertEqual(task["name"], "dual_block_select_place")
        self.assertEqual(task["objects_per_episode"], 2)
        self.assertEqual(task["instruction_template"], "Put the {color} block into the {color} tray.")
        self.assertEqual(task["instruction_values"]["red"], "Put the red block into the red tray.")
        self.assertEqual(task["instruction_values"]["blue"], "Put the blue block into the blue tray.")
        self.assertFalse(task["transfer_protocol"])
        self.assertFalse(task["handover_between_arms"])

    def test_colour_is_not_bound_to_arm_or_side(self):
        layout = self.protocol["scene_layout"]
        self.assertEqual(set(layout["arrangements"]["values"]),
                         {"red_positive_y", "red_negative_y"})
        decoupling = layout["colour_decoupling"]
        self.assertIn("sampled independently", decoupling["rule"])
        self.assertIn("both arms", decoupling["consequence"])
        self.assertEqual(layout["active_arm_rule"]["selection"],
                         "left if the target tray centre has y > 0, right if y < 0")
        self.assertIn("home park", layout["active_arm_rule"]["other_arm"])

    def test_sampling_domains_and_bounds_are_frozen(self):
        layout = self.protocol["scene_layout"]
        self.assertEqual(layout["blocks"]["yaw"]["domain_degrees"], [30.0, 75.0])
        self.assertEqual(layout["blocks"]["yaw"]["domain_status"], "frozen_after_layout_pilot")
        self.assertEqual(layout["trays"]["candidate_region_status"], "frozen_after_layout_pilot")
        self.assertEqual(layout["blocks"]["minimum_pairwise_distance_m"], 0.06)
        self.assertEqual(layout["trays"]["frozen_sampling"]["positive_y"]["x_m"], [0.21, 0.27])
        completed = self.protocol["collection"]["layout_pilot"]["completed"]
        self.assertEqual(completed["accepted"], 240)
        self.assertEqual(completed["attempted"], 240)
        self.assertEqual(completed["by_arrangement"], {"red_positive_y": 120, "red_negative_y": 120})
        self.assertEqual(completed["by_color_arm"]["red"], {"left": 120, "right": 120})
        self.assertEqual(completed["by_color_arm"]["blue"], {"left": 120, "right": 120})
        self.assertTrue(completed["all_accepted_have_both_conditions"])
        self.assertTrue(completed["all_accepted_initial_checks_passed"])
        self.assertIn("rejection-biased", completed["yaw_bias_note"])

    def test_success_predicate_checks_target_and_distractor(self):
        predicate = self.protocol["success_predicate"]
        target = " ".join(predicate["requires_all_target"])
        distractor = " ".join(predicate["requires_all_distractor"])
        for phrase in ("same-colour tray", "supported", "released", "0.02 m/s",
                       "home park", "0.01 m", "0.5 s"):
            self.assertIn(phrase, target)
        for phrase in ("no contact", "bilateral gripper grasp", "displacement",
                       "out of bounds", "0.5 s"):
            self.assertIn(phrase, distractor)
        baseline = predicate["distractor_baseline"]
        self.assertEqual(baseline["status"], "frozen_before_physical_pilot")
        self.assertEqual(baseline["threshold_m"], 0.002)
        self.assertFalse(baseline["measured_any_robot_contact"])
        self.assertFalse(baseline["measured_any_false_grasp"])
        self.assertIn("expert return code", predicate["not_sufficient"])

    def test_family_rule_and_pilot_caps(self):
        self.assertIn("never cross train/validation", self.protocol["scene_family"]["rule"])
        self.assertIn("counterfactual", self.protocol["scene_family"]["rule"])
        pilot = self.protocol["collection"]["physical_pilot"]
        self.assertEqual(pilot["layouts"], 4)
        self.assertEqual(pilot["episodes"], 8)
        self.assertEqual(pilot["episode_attempt_cap"], 8)
        self.assertIn("each layout is executed once", pilot["rule"])
        self.assertTrue(pilot["pilot_is_not_formal_validation"])
        report = ROOT / "reports/so101-scene2-dual-random-scene.md"
        if report.is_file():
            text = report.read_text(encoding="utf-8").lower()
            self.assertIn("not eight independent scenes", text)

    def test_no_clean60_norm_or_checkpoint(self):
        forbidden = self.protocol["collection"]["formal"]["forbidden"]
        self.assertIn("clean60 norm", forbidden)
        self.assertIn("clean60 checkpoint", forbidden)
        self.assertIn("clean100 norm", forbidden)
        self.assertIn("clean100 checkpoint", forbidden)
        self.assertIn("not computed in this phase", self.protocol["collection"]["formal"]["normalisation"])

    def test_chunk_length_is_not_frozen_by_this_branch(self):
        interface = self.protocol["data_interface"]
        self.assertIn("model line decides", interface["chunk_length"])
        self.assertEqual(len(self.protocol["observation"]["state"]["shape"]), 1)
        self.assertEqual(self.protocol["observation"]["state"]["shape"][0], 12)
        self.assertEqual(self.protocol["action"]["shape"], [12])
        self.assertEqual(len(self.protocol["observation"]["cameras"]), 3)

    def test_record_conflict_is_explicit(self):
        conflict = self.protocol["record_conflicts"][0]
        self.assertIn("Place the red block in the tray.", conflict["recorded_instructions"])
        self.assertIn("Put the red block into the red tray.",
                      json.dumps(self.protocol["task"]["instruction_values"]))
        self.assertIn("governs scene2 data", conflict["resolution"])

    def test_amendments_are_recorded(self):
        ids = {entry["id"] for entry in self.protocol["pilot_amendments"]}
        self.assertIn("baseline_seed_conflict", ids)
        self.assertIn("pre_pilot_open_yaw_scan", ids)
        self.assertEqual(self.protocol["collection"]["seed_registry"]["evaluation_reserved"],
                         [21100, 21199])
        self.assertEqual(self.protocol["collection"]["seed_registry"]["distractor_baseline"],
                         [20800, 20899])


class Scene2ProfileContractTest(unittest.TestCase):
    def setUp(self):
        import yaml

        self.profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))

    def test_two_objects_and_no_randomisation(self):
        sorting = self.profile["sorting"]
        self.assertEqual(len(sorting["initial_positions_xy"]), 2)
        self.assertEqual(set(sorting["objects"]), {"cube_0", "cube_1"})
        self.assertEqual({entry["color"] for entry in sorting["objects"].values()},
                         {"red", "blue"})
        self.assertFalse(self.profile["domain_randomization"]["enabled"])
        self.assertFalse(self.profile["scene"]["transfer"]["marker"])

    def test_frozen_execution_parameters(self):
        sorting = self.profile["sorting"]
        self.assertEqual(sorting["yaw_domain_degrees"], [30.0, 75.0])
        self.assertEqual(sorting["distractor_max_displacement_m"], 0.002)
        self.assertEqual(sorting["block_block_min_distance_m"], 0.06)
        self.assertEqual(self.profile["grasp"]["open_percent"], 70.0)
        self.assertEqual(self.profile["grasp"]["closed_percent"], 0.0)


class Scene2AdapterContractTest(unittest.TestCase):
    def test_task_handles_both_blocks_not_only_cube_zero(self):
        task = _class(ADAPTER, "DualBlockSelectPlaceTask")
        source = ast.get_source_segment(_text(ADAPTER), task)
        self.assertIn('self.names = ["cube_0", "cube_1"]', source)
        self.assertIn("def object_statuses", source)
        self.assertIn("def arm_clearance", source)
        self.assertIn("distractor_max_displacement_m", source)
        self.assertIn("bilateral_grasp", source)

    def test_expert_uses_the_sampled_slot(self):
        source = _text(ADAPTER)
        self.assertIn('self.scene.extra["slot_xy"]', source)
        self.assertIn('self.scene.extra["support_z"]', source)
        self.assertIn("stage.name == \"pick_descend\"", source)
        self.assertIn("allowed_object=_target", source)

    def test_scene_hash_binds_layout_and_condition(self):
        source = _text(ADAPTER)
        self.assertIn('scene.sha256 = layout["layout_sha256"]', source)
        self.assertIn('"condition_sha256": common.sha256_json(condition_payload)', source)
        self.assertIn('"physical_sha256"', source)
        self.assertIn('layout["physical_sha256"] = common.sha256_json(physical)', source)
        self.assertNotIn("SingleBlockPlaceTask(", source)

    def test_sampler_screens_both_targets(self):
        source = _text(ADAPTER)
        self.assertIn("for target in blocks:", source)
        self.assertIn("pick_verdict(env, effective", source)
        self.assertIn("place_verdict(env, effective", source)
        self.assertIn("regions.tray_slots(effective", source)
        self.assertIn("measure_initial(env, config, blocks)", source)
        self.assertIn("SceneGenerationError", source)

    def test_initial_state_is_hashed_after_reset(self):
        source = _text(ADAPTER)
        self.assertIn("task.initial_fingerprint = initial_fingerprint(env, scene)", source)
        self.assertIn("snapshot_sha256", source)
        self.assertIn("set_initial_measured", source)
        self.assertIn("env.last_action = env.robot.apply_action(env.data, env.robot.home())", source)


class Scene2CollectorAuditContractTest(unittest.TestCase):
    def test_collector_writes_three_camera_frames_and_applied_action(self):
        source = _text(COLLECTOR)
        self.assertIn("class CameraSink", source)
        self.assertIn('frames.jsonl', source)
        self.assertIn('"applied_action"', source)
        self.assertIn("observation.images", source)
        self.assertIn("ffmpeg", source)
        self.assertIn("initial_fingerprint", source)
        self.assertIn("pair_initial_fingerprint_matches", source)
        self.assertIn("pilot_is_not_validation", source)
        self.assertIn('"layouts": len(layouts)', source)
        self.assertIn('"episodes_executed": len(rows)', source)
        self.assertIn("protocol_episode_cap", source)
        self.assertIn("adapter_exception", source)

    def test_audit_checks_ticks_videos_and_trajectories(self):
        source = _text(AUDIT)
        for token in ("dt_missing", "dt_duplicate", "dt_nonmonotonic", "substeps",
                      "applied_vs_commanded", "frame_count_matches_trace",
                      "longest_duplicate_run", "initial_state_sha256",
                      "net_displacement_m", "contact_sheet", "consistency_passed"):
            self.assertIn(token, source)

    def test_baseline_freezes_a_floor_and_checks_contacts(self):
        source = _text(BASELINE)
        self.assertIn("max(0.002, maximum + 0.001)", source)
        self.assertIn("any_robot_contact", source)
        self.assertIn("any_false_grasp", source)
        self.assertIn("post-settle", source)

    def test_certificate_reports_domain_yaw_not_only_zero_ninety(self):
        source = _text(CERTIFY)
        self.assertIn("_certify_place_slot_yaws", source)
        self.assertIn("certified_in_domain", source)
        self.assertIn("domain_yaws", source)


class Scene2ReportAndSafetyTest(unittest.TestCase):
    def test_reports_are_short(self):
        reports = sorted((ROOT / "reports").glob("so101-scene2-dual-random*.md"))
        self.assertTrue(reports, "scene2 reports are missing")
        for path in reports:
            self.assertLess(len(path.read_text(encoding="utf-8")), 1000,
                            f"{path.name} exceeds 1000 characters")

    def test_new_files_do_not_contain_private_paths(self):
        paths = [PROTOCOL, PROFILE, ADAPTER, COLLECTOR, AUDIT, BASELINE, CERTIFY,
                 ROOT / "docs/so101-scene2-dual-random.md",
                 ROOT / "docs/so101-scene2-dual-random-interface.md"]
        paths += sorted((ROOT / "reports").glob("so101-scene2-dual-random*.md"))
        paths += sorted((ROOT / "experiments/so101").glob("scene2-dual-random-*.json"))
        forbidden = ["/" + "home" + "/", "Desk" + "top", "cache/so101", "runs/so101"]
        for path in paths:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{path.name} contains a private path token")

    def test_tracked_pilot_scene_summary_has_four_layouts(self):
        path = ROOT / "experiments/so101/scene2-dual-random-pilot-scenes.json"
        if not path.is_file():
            self.skipTest("pilot scene summary has not been written yet")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["layouts_accepted"], 4)
        self.assertEqual(len(data["scenes"]), 4)
        self.assertEqual(sum(len(value) for value in data["scenes"].values()), 8)
        self.assertEqual(len({entry["physical_sha256"] for entry in data["layouts"]}), 4)
        self.assertNotIn("external_root", json.dumps(data))

    def test_no_synthetic_timestamp_claim(self):
        text = json.dumps(json.loads(PROTOCOL.read_text(encoding="utf-8")))
        self.assertIn("no synthetic index is treated as proof of no dropped frames", text)


if __name__ == "__main__":
    unittest.main()
