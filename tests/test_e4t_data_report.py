from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "so101"))

import e4t_data_report  # noqa: E402


class SplitManifestTests(unittest.TestCase):
    def test_split_ids_and_hash_are_stable_and_scene_bound(self) -> None:
        episodes = [
            {"episode_index": 4, "split": "validation", "category": "C", "scene_sha256": "scene-4"},
            {"episode_index": 1, "split": "train", "category": "clean", "scene_sha256": "scene-1"},
            {"episode_index": 3, "split": "validation", "category": "A", "scene_sha256": "scene-3"},
            {"episode_index": 0, "split": "train", "category": "A", "scene_sha256": "scene-0"},
            {"episode_index": 2, "split": "train", "category": "C", "scene_sha256": "scene-2"},
        ]

        manifest = e4t_data_report.split_manifest(episodes, "frozen-view-sha")

        self.assertEqual(manifest["train_episode_ids"], [0, 1, 2])
        self.assertEqual(manifest["validation_episode_ids"], [3, 4])
        self.assertEqual(manifest["independent_scene_hashes"], 5)
        self.assertEqual(manifest["counts_by_split_category"], {
            "train/A": 1, "train/C": 1, "train/clean": 1,
            "validation/A": 1, "validation/C": 1,
        })
        self.assertEqual([row["episode_index"] for row in manifest["episodes"]], [0, 1, 2, 3, 4])
        canonical = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        expected = hashlib.sha256(json.dumps(
            canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        self.assertEqual(manifest["manifest_sha256"], expected)


if __name__ == "__main__":
    unittest.main()
