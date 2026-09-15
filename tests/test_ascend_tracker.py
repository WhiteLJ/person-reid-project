from __future__ import annotations

import unittest

import numpy as np

from src.ascend_tracker import AscendBotSortTracker, _DetectionResults
from src.models import Detection


class AscendTrackerTests(unittest.TestCase):
    def test_results_adapter_provides_botsort_fields(self) -> None:
        results = _DetectionResults(
            np.asarray(((1, 2, 11, 22),), dtype=np.float32),
            np.asarray((0.9,), dtype=np.float32),
            np.asarray((0,), dtype=np.float32),
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(np.allclose(results.xywh[0], (6, 12, 10, 20)))
        selected = results[np.asarray([True])]
        self.assertEqual(len(selected), 1)

    def test_botsort_adapter_keeps_track_state_across_updates(self) -> None:
        tracker = AscendBotSortTracker(
            "config/trackers/botsort_baseline.yaml",
            persist=True,
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detection = Detection((80, 40, 140, 200), 0.95, 0)

        first = tracker.update([detection], frame)
        second = tracker.update([detection], frame)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(second[0].class_id, 0)

    def test_ascend_adapter_rejects_appearance_reid_profile(self) -> None:
        with self.assertRaises(ValueError):
            AscendBotSortTracker("config/trackers/botsort_crowd_reid.yaml")


if __name__ == "__main__":
    unittest.main()
