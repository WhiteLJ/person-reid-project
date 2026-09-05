from __future__ import annotations

import unittest

from src.models import Track
from src.roi_selector import bbox_iou, find_track_by_roi, roi_xywh_to_xyxy


class ROISelectorTests(unittest.TestCase):
    def test_roi_xywh_to_xyxy(self) -> None:
        self.assertEqual(roi_xywh_to_xyxy((10, 20, 30, 40)), (10.0, 20.0, 40.0, 60.0))

    def test_identical_boxes_have_iou_one(self) -> None:
        box = (10, 20, 40, 60)
        self.assertAlmostEqual(bbox_iou(box, box), 1.0)

    def test_non_intersecting_boxes_have_iou_zero(self) -> None:
        self.assertEqual(bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_find_track_returns_track_with_highest_iou(self) -> None:
        tracks = [
            Track(1, (0, 0, 20, 20), 0.9, 0),
            Track(2, (10, 10, 30, 30), 0.9, 0),
        ]

        result = find_track_by_roi((12, 12, 10, 10), tracks, min_iou=0.10)

        self.assertIsNotNone(result)
        self.assertEqual(result.track_id, 2)

    def test_find_track_returns_none_below_iou_threshold(self) -> None:
        tracks = [Track(1, (0, 0, 20, 20), 0.9, 0)]

        result = find_track_by_roi((100, 100, 10, 10), tracks, min_iou=0.20)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
