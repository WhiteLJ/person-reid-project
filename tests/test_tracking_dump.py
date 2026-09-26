from __future__ import annotations

import unittest

from tools.compare_tracking_dumps import bbox_iou, compare_records


class TrackingDumpTests(unittest.TestCase):
    def test_bbox_iou_is_computed_for_geometry_matching(self) -> None:
        self.assertAlmostEqual(
            bbox_iou([0, 0, 100, 100], [10, 0, 110, 100]),
            90.0 / 110.0,
        )

    def test_compare_reports_bbox_geometry_divergence(self) -> None:
        pc = {
            10: {
                "frame_index": 10,
                "detections": [
                    {
                        "class": 0,
                        "bbox": [20, 20, 100, 220],
                        "width": 80,
                        "height": 200,
                        "area": 16000,
                        "center": [60, 120],
                        "confidence": 0.9,
                    }
                ],
                "tracks": {"person": [], "vehicle": []},
            }
        }
        atlas = {
            10: {
                "frame_index": 10,
                "detections": [
                    {
                        "class": 0,
                        "bbox": [20, 20, 140, 220],
                        "width": 120,
                        "height": 200,
                        "area": 24000,
                        "center": [80, 120],
                        "confidence": 0.8,
                    }
                ],
                "tracks": {"person": [], "vehicle": []},
            }
        }

        result = compare_records(pc, atlas)

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0]["matched"][0]["diagnostic"],
            "BBOX_GEOMETRY_DIVERGENCE",
        )
        self.assertIn("area_grew", result[0]["matched"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
