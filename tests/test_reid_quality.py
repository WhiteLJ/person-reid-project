from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.config import ReIDConfig, ReIDQualityConfig
from src.models import Track
from src.reid_quality import assess_reid_quality


def _reid_config() -> ReIDConfig:
    return ReIDConfig(
        model_name="osnet_x0_25",
        weight=Path("unused.pth"),
        image_height=256,
        image_width=128,
        min_crop_width=20,
        min_crop_height=50,
    )


class ReIDQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((120, 100, 3), dtype=np.uint8)
        self.config = _reid_config()

    def test_good_crop_is_accepted(self) -> None:
        track = Track(1, (10, 0, 50, 110), 0.9, 0)
        result = assess_reid_quality(
            self.frame, track, [track], self.config, ReIDQualityConfig()
        )
        self.assertTrue(result.accepted)
        assert result.crop is not None
        self.assertEqual(result.crop.shape[:2], (110, 40))

    def test_low_confidence_is_rejected(self) -> None:
        track = Track(1, (10, 0, 50, 110), 0.2, 0)
        result = assess_reid_quality(
            self.frame, track, [track], self.config, ReIDQualityConfig()
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "low_confidence")

    def test_small_crop_is_rejected(self) -> None:
        track = Track(1, (10, 10, 20, 50), 0.9, 0)
        result = assess_reid_quality(
            self.frame, track, [track], self.config, ReIDQualityConfig()
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "small_crop")

    def test_severe_edge_truncation_is_rejected(self) -> None:
        track = Track(1, (-20, 0, 20, 110), 0.9, 0)
        result = assess_reid_quality(
            self.frame, track, [track], self.config, ReIDQualityConfig()
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "edge_truncated")
        self.assertAlmostEqual(result.edge_truncation_ratio, 0.5)

    def test_overlap_uses_fraction_of_target_area_not_iou(self) -> None:
        target = Track(1, (10, 0, 50, 110), 0.9, 0)
        other = Track(2, (30, 0, 50, 110), 0.9, 0)
        result = assess_reid_quality(
            self.frame,
            target,
            [target, other],
            self.config,
            ReIDQualityConfig(max_person_overlap_ratio=0.49),
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "person_overlap")
        self.assertAlmostEqual(result.max_person_overlap_ratio, 0.5)

    def test_partial_overlap_below_gate_is_accepted(self) -> None:
        target = Track(1, (10, 0, 50, 110), 0.9, 0)
        other = Track(2, (45, 0, 50, 110), 0.9, 0)
        result = assess_reid_quality(
            self.frame,
            target,
            [target, other],
            self.config,
            ReIDQualityConfig(max_person_overlap_ratio=0.20),
        )
        self.assertTrue(result.accepted)
        self.assertAlmostEqual(result.max_person_overlap_ratio, 0.125)


if __name__ == "__main__":
    unittest.main()
