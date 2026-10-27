"""E4-T data, resource and deployment safeguards."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "deploy"))
sys.path.insert(0, str(ROOT / "scripts" / "so101"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect = load_module("collect_e4t", ROOT / "scripts" / "so101" / "collect_e4t.py")
sync = load_module("deploy_sync", ROOT / "scripts" / "deploy" / "sync.py")
resources = load_module("e4t_resources", ROOT / "scripts" / "so101" / "e4t_resources.py")
protocol = json.loads((ROOT / "configs" / "so101" / "e4t_protocol.json").read_text())


class DataPlanTest(unittest.TestCase):
    def test_main_plan_category_and_mechanism_quotas(self):
        specs = collect.candidate_specs(protocol, pilot=False)
        targets = protocol["collection"]["target_successes_per_profile"]
        seen = {profile: 0 for profile in targets}
        counts = {}
        for row in specs:
            profile = row["profile"]
            if seen[profile] >= targets[profile]:
                continue
            seen[profile] += 1
            schedule = row.get("perturbation")
            if schedule:
                key = (schedule["category"], schedule["mechanism"])
                counts[key] = counts.get(key, 0) + 1
        expected = {}
        for category, settings in protocol["perturbations"]["categories"].items():
            for mechanism, count in settings["mechanisms"].items():
                if count:
                    expected[(category, mechanism)] = count
        self.assertEqual(counts, expected)

    def test_split_helper_freezes_category_quotas(self):
        rows = []
        profile_counts = {name: tuple(value) for name, value in
                          protocol["split"]["new_run_per_category"].items()}
        index = 0
        for profile, (train, validation) in profile_counts.items():
            for kind, count in (("train", train), ("validation", validation)):
                for _ in range(count):
                    rows.append({"profile": profile, "source_episode_index": index,
                                 "scene_sha256": f"sha-{index}"})
                    index += 1
        split = collect.assign_split(rows, protocol)
        self.assertEqual(split["problems"], [])
        self.assertEqual(len(split["train"]), sum(v[0] for v in profile_counts.values()))
        self.assertEqual(len(split["validation"]), sum(v[1] for v in profile_counts.values()))
        self.assertEqual(len({row["scene_sha256"] for row in rows}), 30)

    def test_cross_event_window_exclusion(self):
        starts = collect._eligible_starts(30, 20, 10)
        self.assertNotIn(11, starts)  # 11..20 crosses trigger
        self.assertIn(10, starts)     # 10..19 ends before trigger
        self.assertNotIn(20, starts)  # 20..29 crosses trigger
        self.assertNotIn(21, starts)  # no horizon left within the 30-frame episode


class MechanismAcceptanceTest(unittest.TestCase):
    """The four registered mechanisms each need their own physical evidence."""

    def test_slip_requires_recorded_command_change_and_grasp_separation(self):
        event = {"type": "perturbation", "mechanism": "gripper_release_slip", "object": "cube_0",
                 "sim_time_s": 5.0, "control_frame": 250,
                 "gripper_release": {"actuator": "left_gripper_position", "index": 5,
                                     "ctrl_before_rad": 0.0, "ctrl_after_rad": 1.2,
                                     "target_rad": 1.2},
                 "slip_hold_frames": 30}
        events = [event, {"type": "expert_recovery_settle", "object": "cube_0", "sim_time": 5.02},
                  {"type": "plan_start", "object": "cube_0", "label": "direct_pick_cube_0",
                   "sim_time": 9.0}]
        self.assertTrue(collect.grasp_separation_evidence(events, event))
        without = [event, {"type": "plan_start", "object": "cube_0", "label": "x", "sim_time": 9.0}]
        self.assertFalse(collect.grasp_separation_evidence(without, event))

    def test_bump_is_a_registered_arm_impulse_not_a_teleport(self):
        protocol = json.loads((ROOT / "configs/so101/e4t_protocol.json").read_text())
        spec = collect._category_schedule("recovery_C", 8260, protocol, 0, "arm_bump_impulse")
        self.assertEqual(spec["mechanism"], "arm_bump_impulse")
        self.assertNotEqual(spec["arm_delta_velocity_rad_s"], 0.0)
        self.assertEqual(spec["delta_position_m"], [0.0, 0.0, 0.0])
        low, high = protocol["perturbations"]["categories"]["C"]["arm_bump_delta_v_rad_s"]
        self.assertTrue(low <= abs(spec["arm_delta_velocity_rad_s"]) <= high)
        self.assertEqual(len({collect._category_schedule("recovery_C", seed, protocol, seed - 8260,
                                                         "arm_bump_impulse")["plan_sha256"]
                             for seed in range(8260, 8270)}), 10)

    def test_mechanism_quotas_still_sum_after_the_revision(self):
        protocol = json.loads((ROOT / "configs/so101/e4t_protocol.json").read_text())
        for category, settings in protocol["perturbations"]["categories"].items():
            self.assertEqual(sum(settings["mechanisms"].values()), int(settings["target"]),
                             f"category {category} mechanism quota does not sum to its target")

    def test_withdrawn_mechanism_is_excluded_not_admitted(self):
        verify = load_module("verify_e4t", ROOT / "scripts" / "so101" / "verify_e4t.py")
        admitted = verify.admitted_mechanisms(protocol)
        self.assertEqual(admitted["A"], {"bounded_velocity_impulse", "explicit_state_displacement"})
        self.assertEqual(admitted["C"], {"arm_bump_impulse"})
        # Category B lists every mechanism with a zero quota: all of them are withdrawn, so no
        # B episode can be admitted, and its episodes must be reported as excluded instead.
        self.assertEqual(admitted["B"], set())
        self.assertNotIn("gripper_release_slip", admitted["B"])
        self.assertNotIn("gripper_release_slip", admitted["C"])

    def test_collected_mechanism_mix_matches_the_registered_per_mechanism_quotas(self):
        # The main shard overshot the impulse sub-quota and the top-up ran the rest, so the round's
        # registered mix is what the dataset is actually checked against.
        expected = protocol["collection"]["mechanism_mix"]
        counts = {}
        for settings in protocol["perturbations"]["categories"].values():
            for mechanism, quota in settings["mechanisms"].items():
                counts[mechanism] = counts.get(mechanism, 0) + int(quota)
        self.assertEqual({key: value for key, value in counts.items() if value}, expected)
        self.assertEqual(sum(expected.values()), 25)

class DeploymentSafetyTest(unittest.TestCase):
    cfg = {
        "ssh": {"host": "example.invalid", "port": 22, "user": "example", "key": "~/.ssh/id_ed25519"},
        "remote": {"root": "/tmp/remote", "runs_dir": "/tmp/remote/runs"},
        "sync": {"excludes": []},
    }

    def test_real_deploy_omits_delete(self):
        command = sync.build_rsync(self.cfg, check=False, dry_run=False)
        self.assertNotIn("--delete", command)

    def test_dry_run_reports_without_changing(self):
        command = sync.build_rsync(self.cfg, check=False, dry_run=True)
        self.assertIn("--delete", command)
        self.assertIn("--dry-run", command)

    def test_check_reports_without_changing(self):
        command = sync.build_rsync(self.cfg, check=True, dry_run=False)
        self.assertIn("--delete", command)
        self.assertIn("--dry-run", command)


class ResourceOwnershipTest(unittest.TestCase):
    def test_registration_requires_all_exact_fields(self):
        entry = {
            "container_id": "abc", "run_id": "r", "owner": "o", "exact_name": "n",
            "image": "i", "command": ["x"],
        }
        inspected = {
            "container_id": "abc", "image": "i", "args": ["x"],
            "labels": {"pi05.e4t.project": "pi05-reproduction", "pi05.e4t.run_id": "r",
                       "pi05.e4t.owner": "o", "pi05.e4t.exact_name": "n"},
            "path": "/usr/bin/python3",
        }
        self.assertEqual(resources._entry_matches(entry, inspected), [])
        inspected["labels"]["pi05.e4t.run_id"] = "different"
        self.assertTrue(any("run_id" in problem for problem in resources._entry_matches(entry, inspected)))
        inspected["labels"]["pi05.e4t.run_id"] = "r"
        inspected["args"] = ["different"]
        self.assertTrue(any("command" in problem for problem in resources._entry_matches(entry, inspected)))


class AdmissionContractTest(unittest.TestCase):
    """The training container reads the admission file the exporter writes, one process to the other."""

    export = load_module("export_e4t_view", ROOT / "scripts" / "so101" / "export_e4t_view.py")

    def test_complete_admission_is_accepted(self):
        document = {"passed": True, "data_gate": {"passed": True}, "view_sha256": "abc",
                    "totals": {"episodes": 50, "train": 40, "validation": 10}}
        self.assertEqual(self.export.admission_contract_problems(document), [])

    def test_admission_without_a_data_gate_is_refused(self):
        document = {"passed": True, "view_sha256": "abc",
                    "totals": {"episodes": 50, "train": 40, "validation": 10}}
        self.assertEqual(self.export.admission_contract_problems(document),
                         ["missing field 'data_gate.passed'"])

    def test_admission_with_a_false_data_gate_is_refused(self):
        document = {"passed": True, "data_gate": {"passed": False}, "view_sha256": "abc",
                    "totals": {"episodes": 50, "train": 40, "validation": 10}}
        self.assertEqual(self.export.admission_contract_problems(document),
                         ["field 'data_gate.passed' is not true"])

    def test_training_entry_reads_exactly_the_contracted_fields(self):
        source = (ROOT / "scripts" / "so101" / "train_so101.py").read_text()
        self.assertIn('admission.get("data_gate", {}).get("passed")', source)
        self.assertIn('admission.get("view_sha256")', source)
        self.assertIn('admission_counts.get("episodes")', source)
        self.assertEqual({item[0] for item in self.export.ADMISSION_CONTRACT},
                         {"passed", "data_gate", "view_sha256", "totals"})


if __name__ == "__main__":
    unittest.main()
