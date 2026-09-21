from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_reid_calibration import _format_stats, _pairwise_scores


class VehicleReIDCalibrationTests(unittest.TestCase):
    def test_pairwise_scores_separate_positive_and_negative_pairs(self) -> None:
        embeddings = {
            "car1": [
                ("car1/001.jpg", np.asarray((1.0, 0.0), dtype=np.float32)),
                ("car1/002.jpg", np.asarray((1.0, 0.0), dtype=np.float32)),
            ],
            "car2": [
                ("car2/001.jpg", np.asarray((0.0, 1.0), dtype=np.float32)),
            ],
        }

        positive, negative = _pairwise_scores(embeddings)

        self.assertEqual(len(positive), 1)
        self.assertEqual(len(negative), 2)
        self.assertAlmostEqual(positive[0][0], 1.0)
        self.assertAlmostEqual(negative[0][0], 0.0)

    def test_format_stats_handles_empty_and_nonempty_values(self) -> None:
        self.assertEqual(_format_stats([]), "no_pairs")
        formatted = _format_stats([0.2, 0.8])
        self.assertIn("min=0.200000", formatted)
        self.assertIn("max=0.800000", formatted)


if __name__ == "__main__":
    unittest.main()
