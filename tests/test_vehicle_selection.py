from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.models import Track
from src.target_manager import TargetManager
from src.vehicle_selection import VehicleSelectionController, crop_vehicle
from src.visualization import VEHICLE_COLOR, VEHICLE_SELECTED_COLOR
from tools.vehicle_selection_smoke_test import _draw_vehicle_track, _render_frame
from ui.roi_editor import EditMode, ROIEditSession


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

    def test_add_edit_session_can_select_three_vehicles(self) -> None:
        tracks = [
            _track(12),
            Track(13, (55, 10, 95, 50), 0.9, 2),
            Track(14, (10, 55, 50, 79), 0.9, 2),
        ]

        def on_roi(roi, frozen_tracks, mode) -> None:
            self.assertIs(mode, EditMode.ADD_TARGETS)
            self.controller.select_from_roi(
                self.frame,
                frozen_tracks,
                roi,
                frame_index=0,
            )

        session = ROIEditSession(
            window_name="test-window",
            frame=self.frame,
            tracks=tracks,
            mode=EditMode.ADD_TARGETS,
            wait_key_ms=1,
            on_roi=on_roi,
            render_frame=lambda frame, ignored_tracks: frame.copy(),
        )

        with patch("ui.roi_editor.cv2.imshow"):
            for roi in (
                (10, 10, 40, 40),
                (55, 10, 40, 40),
                (10, 55, 40, 24),
            ):
                x, y, width, height = roi
                session._on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
                session._on_mouse(
                    cv2.EVENT_LBUTTONUP,
                    x + width,
                    y + height,
                    0,
                    None,
                )

        self.assertEqual(self.manager.selected_track_ids, {12, 13, 14})
        self.assertEqual(len(self.manager.targets), 3)
        self.assertEqual(self.extractor.calls, 3)

    def test_remove_edit_session_can_remove_two_vehicles(self) -> None:
        tracks = [
            _track(12),
            Track(13, (55, 10, 95, 50), 0.9, 2),
            Track(14, (10, 55, 50, 79), 0.9, 2),
        ]
        for track in tracks:
            self.manager.select(track, np.ones(2048, dtype=np.float32))

        selected_tracks = self.manager.selected_tracks(tracks)

        def on_roi(roi, frozen_tracks, mode) -> None:
            self.assertIs(mode, EditMode.REMOVE_TARGETS)
            self.controller.remove_from_roi(frozen_tracks, roi)

        session = ROIEditSession(
            window_name="test-window",
            frame=self.frame,
            tracks=selected_tracks,
            mode=EditMode.REMOVE_TARGETS,
            wait_key_ms=1,
            on_roi=on_roi,
            render_frame=lambda frame, ignored_tracks: frame.copy(),
        )

        with patch("ui.roi_editor.cv2.imshow"):
            for roi in ((10, 10, 40, 40), (10, 55, 40, 24)):
                x, y, width, height = roi
                session._on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
                session._on_mouse(
                    cv2.EVENT_LBUTTONUP,
                    x + width,
                    y + height,
                    0,
                    None,
                )

        self.assertEqual(self.manager.selected_track_ids, {13})
        self.assertEqual(len(self.manager.targets), 1)

    def test_unselected_vehicle_visibility_follows_debug_flag(self) -> None:
        track = _track(12)

        hidden = _render_frame(
            self.frame,
            [],
            [track],
            self.manager,
            show_unselected_tracks=False,
        )
        shown = _render_frame(
            self.frame,
            [],
            [track],
            self.manager,
            show_unselected_tracks=True,
        )

        self.assertTrue(np.array_equal(hidden[10, 10], (0, 0, 0)))
        self.assertTrue(np.array_equal(shown[10, 10], VEHICLE_COLOR))

    def test_selected_vehicle_is_visible_and_stays_dark_blue(self) -> None:
        track = _track(12)
        self.manager.select(track, np.ones(2048, dtype=np.float32))

        rendered = _render_frame(
            self.frame,
            [],
            [track],
            self.manager,
            show_unselected_tracks=False,
        )

        self.assertTrue(
            np.array_equal(rendered[10, 10], VEHICLE_SELECTED_COLOR)
        )

    def test_person_and_vehicle_debug_visibility_share_the_same_flag(self) -> None:
        person_track = Track(4, (55, 10, 95, 50), 0.9, 0)
        vehicle_track = _track(12)

        with patch(
            "tools.vehicle_selection_smoke_test._draw_track"
        ) as draw_person, patch(
            "tools.vehicle_selection_smoke_test._draw_vehicle_track"
        ) as draw_vehicle:
            _render_frame(
                self.frame,
                [person_track],
                [vehicle_track],
                self.manager,
                show_unselected_tracks=False,
            )
            draw_person.assert_not_called()
            draw_vehicle.assert_called_once()
            self.assertFalse(
                draw_vehicle.call_args.kwargs["show_unselected_tracks"]
            )

            draw_person.reset_mock()
            draw_vehicle.reset_mock()
            _render_frame(
                self.frame,
                [person_track],
                [vehicle_track],
                self.manager,
                show_unselected_tracks=True,
            )
            draw_person.assert_called_once()
            draw_vehicle.assert_called_once()
            self.assertTrue(
                draw_vehicle.call_args.kwargs["show_unselected_tracks"]
            )

    def test_vehicle_selected_and_unselected_styles_only_change_weight(self) -> None:
        track = _track(12)

        with patch("tools.vehicle_selection_smoke_test.cv2.rectangle") as rectangle:
            _draw_vehicle_track(
                self.frame.copy(),
                track,
                self.manager,
                show_unselected_tracks=True,
            )
            normal_color = rectangle.call_args.args[3]
            normal_thickness = rectangle.call_args.args[4]

            self.manager.select(track, np.ones(2048, dtype=np.float32))
            _draw_vehicle_track(
                self.frame.copy(),
                track,
                self.manager,
                show_unselected_tracks=False,
            )
            selected_color = rectangle.call_args.args[3]
            selected_thickness = rectangle.call_args.args[4]

        self.assertEqual(normal_thickness, 2)
        self.assertEqual(selected_thickness, 4)
        self.assertEqual(normal_color, VEHICLE_COLOR)
        self.assertEqual(selected_color, VEHICLE_SELECTED_COLOR)

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

        self.assertTrue(
            self.controller.remove_from_roi(
                self.manager.selected_tracks(tracks),
                (55, 10, 40, 40),
            )
        )
        self.assertEqual(self.manager.targets, {})

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
