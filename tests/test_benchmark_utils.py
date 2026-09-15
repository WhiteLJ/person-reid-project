from __future__ import annotations

import unittest

from src.benchmark_utils import percentile_stats


class BenchmarkUtilsTests(unittest.TestCase):
    def test_percentile_stats_reports_distribution(self) -> None:
        stats = percentile_stats((1.0, 2.0, 3.0, 4.0, 100.0))

        self.assertEqual(stats["count"], 5)
        self.assertAlmostEqual(stats["average"], 22.0)
        self.assertAlmostEqual(stats["p50"], 3.0)
        self.assertEqual(stats["max"], 100.0)

    def test_empty_percentile_stats_is_well_defined(self) -> None:
        stats = percentile_stats(())

        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["p99"], 0.0)


if __name__ == "__main__":
    unittest.main()
