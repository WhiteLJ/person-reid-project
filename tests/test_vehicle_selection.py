from __future__ import annotations

import unittest

import numpy as np

from src.models import Track
from src.target_manager import TargetManager
from src.vehicle_selection import VehicleSelectionController, crop_vehicle


class _FakeVehicleReID:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, crop: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.last_crop_shape = crop.shape
        embedding = np.zeros(2048, dtype=np.float32)
        embedding[0] = 3.0
        embedding[1] = 4.0
        return embedding


def _track(track_id: int, class_id: int = 2) -> Track:
    return Track(track_id, (10, 10, 50, 50), 0.9, class_id)


class VehicleSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((80, 100, 3), dtype=np.uint8)
        self.manager = TargetManager()
        self.extractor = _FakeVehicleReID()
        self.controller = VehicleSelectionController(
            self.manager,
            self.extractor,
            min_iou=0.20,
        )

    def test_vehicle_track_creates_2048d_normalized_target(self) -> None:
        target = self.controller.select_from_roi(
            self.frame,
            [_track(12)],
            (10, 10, 40, 40),
            frame_index=3,
        )

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.target_id, 1)
        self.assertEqual(target.current_track_id, 12)
        self.assertEqual(target.centroid.shape, (2048,))
        self.assertEqual(target.centroid.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(target.centroid)), 1.0, places=6)
        self.assertEqual(self.extractor.calls, 1)

    def test_reselecting_same_vehicle_is_idempotent_and_does_not_extract(self) -> None:
        first = self.controller.select_from_roi(
            self.frame, [_track(12)], (10, 10, 40, 40), 0
        )
        second = self.controller.select_from_roi(
            self.frame, [_track(12)], (10, 10, 40, 40), 1
        )

        self.assertIs(first, second)
        self.assertEqual(len(self.manager.targets), 1)
        self.assertEqual(self.extractor.calls, 1)

    def test_two_vehicles_can_be_selected_independently(self) -> None:
        tracks = [_track(12), Track(13, (55, 10, 95, 50), 0.9, 2)]

        self.controller.select_from_roi(self.frame, tracks, (10, 10, 40, 40), 0)
        self.controller.select_from_roi(self.frame, tracks, (55, 10, 40, 40), 0)

        self.assertEqual(len(self.manager.targets), 2)
        self.assertEqual(self.manager.selected_track_ids, {12, 13})
        self.assertEqual(self.extractor.calls, 2)

    def test_remove_one_vehicle_preserves_the_other(self) -> None:
        tracks = [_track(12), Track(13, (55, 10, 95, 50), 0.9, 2)]
        self.controller.select_from_roi(self.frame, tracks, (10, 10, 40, 40), 0)
        self.controller.select_from_roi(self.frame, tracks, (55, 10, 40, 40), 0)

        selected_tracks = self.manager.selected_tracks(tracks)
        self.assertTrue(
            self.controller.remove_from_roi(selected_tracks, (10, 10, 40, 40))
        )
        self.assertEqual(self.manager.selected_track_ids, {13})
        self.assertEqual(len(self.manager.targets), 1)

    def test_clear_only_clears_vehicle_manager(self) -> None:
        vehicle_track = _track(4)
        person_track = _track(4, class_id=0)
        self.controller.select_from_roi(
            self.frame, [vehicle_track], (10, 10, 40, 40), 0
        )
        person_manager = TargetManager()
        person_manager.select(person_track, np.asarray((1.0, 0.0), dtype=np.float32))

        self.manager.clear()

        self.assertEqual(self.manager.targets, {})
        self.assertEqual(len(person_manager.targets), 1)

    def test_roi_is_limited_to_vehicle_tracks(self) -> None:
        person_track = _track(4, class_id=0)

        target = self.controller.select_from_roi(
            self.frame,
            [],
            (10, 10, 40, 40),
            0,
        )

        self.assertIsNone(target)
        self.assertEqual(self.extractor.calls, 0)
        self.assertIsNone(self.manager.target_for_track(person_track.track_id))

    def test_invalid_vehicle_bbox_does_not_infer(self) -> None:
        invalid = Track(12, (60, 60, 60, 60), 0.9, 2)

        target = self.controller.select_from_roi(
            self.frame,
            [invalid],
            (60, 60, 10, 10),
            0,
        )

        self.assertIsNone(target)
        self.assertEqual(self.extractor.calls, 0)

    def test_crop_vehicle_clips_to_frame(self) -> None:
        crop = crop_vehicle(self.frame, (-10, -5, 20, 25))

        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual(crop.shape, (25, 20, 3))


if __name__ == "__main__":
    unittest.main()
