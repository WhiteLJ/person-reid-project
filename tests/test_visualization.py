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

        result = draw_tracks(frame, tracks, show_unselected_tracks=True)

        self.assertEqual(result.shape, frame.shape)
        self.assertFalse(np.array_equal(result, frame))
        self.assertTrue(np.any(result[10, 10] != 0))

    def test_draw_tracks_highlights_all_selected_track_ids(self) -> None:
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        tracks = [
            Track(7, (10, 10, 40, 60), 0.9, 0),
            Track(8, (60, 10, 90, 60), 0.8, 0),
        ]

        result = draw_tracks(frame, tracks, selected_track_ids={7, 8})

        self.assertTrue(np.array_equal(result[10, 10], (0, 0, 255)))
        self.assertTrue(np.array_equal(result[10, 60], (0, 0, 255)))

    def test_draw_tracks_hides_unselected_tracks_by_default(self) -> None:
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        tracks = [
            Track(7, (10, 10, 40, 60), 0.9, 0),
            Track(8, (60, 10, 90, 60), 0.8, 0),
        ]

        result = draw_tracks(frame, tracks, selected_track_ids={7})

        self.assertTrue(np.array_equal(result[10, 10], (0, 0, 255)))
        self.assertTrue(np.array_equal(result[10, 60], (0, 0, 0)))

    def test_draw_tracks_can_show_unselected_tracks_for_debugging(self) -> None:
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        tracks = [Track(8, (60, 10, 90, 60), 0.8, 0)]

        result = draw_tracks(frame, tracks, show_unselected_tracks=True)

        self.assertTrue(np.array_equal(result[10, 60], (0, 255, 0)))


if __name__ == "__main__":
    unittest.main()
