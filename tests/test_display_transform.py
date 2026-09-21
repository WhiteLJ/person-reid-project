from __future__ import annotations

import unittest

import numpy as np

from src.display_transform import DisplayTransform


class DisplayTransformTests(unittest.TestCase):
    def test_1920x1080_fits_to_1280x720_and_maps_roi_back(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        transform = DisplayTransform.from_frame(frame, 1280)

        self.assertEqual(transform.display_shape, (720, 1280))
        display = transform.source_to_display(frame)
        self.assertEqual(display.shape[:2], (720, 1280))
        self.assertEqual(
            transform.display_roi_xywh_to_source((100, 100, 200, 100)),
            (150, 150, 300, 150),
        )

    def test_small_frame_is_not_upscaled(self) -> None:
        transform = DisplayTransform.from_size(640, 480, 1280)

        self.assertEqual(transform.display_shape, (480, 640))
        self.assertEqual(transform.scale, 1.0)

    def test_zero_or_none_max_width_keeps_original_size(self) -> None:
        self.assertEqual(
            DisplayTransform.from_size(1920, 1080, 0).display_shape,
            (1080, 1920),
        )
        self.assertEqual(
            DisplayTransform.from_size(1920, 1080, None).display_shape,
            (1080, 1920),
        )

    def test_bbox_mapping_preserves_source_display_round_trip(self) -> None:
        transform = DisplayTransform.from_size(1920, 1080, 1280)
        source_bbox = (300, 240, 1500, 900)
        display_bbox = transform.source_xyxy_to_display(source_bbox)
        round_trip = transform.display_xyxy_to_source(display_bbox)

        self.assertEqual(round_trip, source_bbox)


if __name__ == "__main__":
    unittest.main()
