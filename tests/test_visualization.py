from __future__ import annotations

import unittest

import numpy as np

from src.models import Detection, Track
from src.visualization import draw_detections, draw_tracks


class VisualizationTests(unittest.TestCase):
    def test_draw_detections_returns_annotated_frame(self) -> None:
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        detections = [Detection((10, 10, 50, 60), 0.9, 0)]

        result = draw_detections(frame, detections, lambda class_id: "person")

        self.assertEqual(result.shape, frame.shape)
        self.assertFalse(np.array_equal(result, frame))
        self.assertTrue(np.any(result[10, 10] != 0))

    def test_draw_tracks_returns_annotated_frame(self) -> None:
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        tracks = [Track(7, (10, 10, 50, 60), 0.9, 0)]

        result = draw_tracks(frame, tracks)

        self.assertEqual(result.shape, frame.shape)
        self.assertFalse(np.array_equal(result, frame))
        self.assertTrue(np.any(result[10, 10] != 0))


if __name__ == "__main__":
    unittest.main()
