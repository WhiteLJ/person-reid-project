from __future__ import annotations

import unittest

import numpy as np

from src.feature_diversity import filter_diverse_references, reference_similarity_stats


def _vector(first: float, second: float = 0.0) -> np.ndarray:
    value = np.zeros((512,), dtype=np.float32)
    value[0] = first
    value[1] = second
    return value / np.linalg.norm(value)


class FeatureDiversityTests(unittest.TestCase):
    def test_pairwise_stats_report_near_duplicates(self) -> None:
        stats = reference_similarity_stats(
            [_vector(1.0), _vector(1.0), _vector(0.0, 1.0)]
        )
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["pair_count"], 3)
        self.assertEqual(stats["duplicate_pair_count"], 1)
        self.assertAlmostEqual(float(stats["max"]), 1.0)

    def test_filter_keeps_stable_order_and_removes_duplicate_reference(self) -> None:
        first = _vector(1.0)
        duplicate = _vector(1.0)
        different = _vector(0.0, 1.0)

        retained = filter_diverse_references(
            [first, duplicate, different],
            duplicate_threshold=0.98,
        )

        self.assertEqual(len(retained), 2)
        self.assertTrue(np.allclose(retained[0], first))
        self.assertTrue(np.allclose(retained[1], different))
        self.assertIsNot(retained[0], first)


if __name__ == "__main__":
    unittest.main()
