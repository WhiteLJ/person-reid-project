from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class TrackerProfileTests(unittest.TestCase):
    def _load(self, name: str) -> dict[str, object]:
        path = Path("config/trackers") / name
        with path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def test_baseline_has_appearance_reid_disabled(self) -> None:
        profile = self._load("botsort_baseline.yaml")
        self.assertFalse(profile["with_reid"])
        self.assertEqual(profile["track_buffer"], 30)
        self.assertEqual(profile["gmc_method"], "sparseOptFlow")

    def test_fixed_camera_profile_disables_gmc_for_atlas_ab(self) -> None:
        profile = self._load("botsort_fixed_camera.yaml")
        self.assertFalse(profile["with_reid"])
        self.assertEqual(profile["gmc_method"], "none")

    def test_crowd_profile_is_explicit_experiment(self) -> None:
        profile = self._load("botsort_crowd.yaml")
        self.assertFalse(profile["with_reid"])
        self.assertEqual(profile["track_buffer"], 60)

    def test_crowd_appearance_reid_profile_is_separate(self) -> None:
        profile = self._load("botsort_crowd_reid.yaml")
        self.assertTrue(profile["with_reid"])
        self.assertEqual(profile["model"], "auto")


if __name__ == "__main__":
    unittest.main()
