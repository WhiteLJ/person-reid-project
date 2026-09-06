from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.models import Track
from ui.roi_editor import EditMode, ROIEditSession, UIAction, normalize_roi_xyxy


class ROIEditorGeometryTests(unittest.TestCase):
    def test_normalize_roi_supports_reverse_drag_direction(self) -> None:
        result = normalize_roi_xyxy((90, 70), (10, 20), (80, 120, 3))

        self.assertEqual(result, (10, 20, 90, 70))

    def test_normalize_roi_clips_to_frame_boundaries(self) -> None:
        result = normalize_roi_xyxy((-10, -5), (130, 100), (80, 120, 3))

        self.assertEqual(result, (0, 0, 120, 80))

    def test_normalize_roi_ignores_zero_or_too_small_area(self) -> None:
        self.assertIsNone(normalize_roi_xyxy((10, 10), (10, 20), (80, 120, 3)))
        self.assertIsNone(normalize_roi_xyxy((10, 10), (11, 11), (80, 120, 3)))


class ROIEditSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((80, 120, 3), dtype=np.uint8)
        self.tracks = (
            Track(7, (0, 0, 30, 30), 0.9, 0),
            Track(8, (40, 0, 70, 30), 0.9, 0),
        )

    def _session(self, on_roi, mode=EditMode.ADD_TARGETS) -> ROIEditSession:
        return ROIEditSession(
            window_name="test-window",
            frame=self.frame,
            tracks=self.tracks,
            mode=mode,
            wait_key_ms=1,
            on_roi=on_roi,
            render_frame=lambda frame, tracks: frame.copy(),
        )

    @patch("ui.roi_editor.cv2.imshow")
    @patch("ui.roi_editor.cv2.setMouseCallback")
    @patch("ui.roi_editor.cv2.waitKey", return_value=ord("q"))
    def test_q_propagates_quit_and_clears_callback(
        self, wait_key, set_mouse_callback, imshow
    ) -> None:
        session = self._session(lambda roi, tracks, mode: None)

        result = session.run()

        self.assertEqual(result, UIAction.QUIT)
        self.assertEqual(set_mouse_callback.call_count, 2)
        self.assertIsNone(set_mouse_callback.call_args_list[1].args[1])

    @patch("ui.roi_editor.cv2.imshow")
    @patch("ui.roi_editor.cv2.setMouseCallback")
    @patch("ui.roi_editor.cv2.waitKey", return_value=13)
    def test_normal_exit_clears_callback_and_state(
        self, wait_key, set_mouse_callback, imshow
    ) -> None:
        session = self._session(lambda roi, tracks, mode: None)
        session._dragging = True
        session._drag_start = (4, 4)
        session._drag_current = (10, 10)

        result = session.run()

        self.assertEqual(result, UIAction.NONE)
        self.assertEqual(set_mouse_callback.call_count, 2)
        self.assertIsNone(set_mouse_callback.call_args_list[1].args[1])
        self.assertFalse(session._dragging)
        self.assertIsNone(session._drag_start)
        self.assertIsNone(session._drag_current)

    @patch("ui.roi_editor.cv2.imshow")
    @patch("ui.roi_editor.cv2.setMouseCallback")
    def test_exception_clears_callback_and_state(self, set_mouse_callback, imshow) -> None:
        session = self._session(lambda roi, tracks, mode: None)
        imshow.side_effect = RuntimeError("render failed")

        with self.assertRaises(RuntimeError):
            session.run()

        self.assertEqual(set_mouse_callback.call_count, 2)
        self.assertIsNone(set_mouse_callback.call_args_list[1].args[1])
        self.assertFalse(session._dragging)
        self.assertIsNone(session._drag_start)
        self.assertIsNone(session._drag_current)

    @patch("ui.roi_editor.cv2.imshow")
    @patch("ui.roi_editor.cv2.setMouseCallback")
    def test_one_session_can_submit_multiple_rois(self, set_mouse_callback, imshow) -> None:
        received: list[tuple[tuple[int, int, int, int], tuple[Track, ...], EditMode]] = []
        session = self._session(lambda roi, tracks, mode: received.append((roi, tracks, mode)))

        keys = [None, 13]

        def wait_key(_wait_ms: int) -> int:
            if keys.pop(0) is None:
                session._on_mouse(cv2.EVENT_LBUTTONDOWN, 20, 20, 0, None)
                session._on_mouse(cv2.EVENT_LBUTTONUP, 1, 1, 0, None)
                session._on_mouse(cv2.EVENT_LBUTTONDOWN, 60, 20, 0, None)
                session._on_mouse(cv2.EVENT_LBUTTONUP, 41, 1, 0, None)
                return 255
            return 13

        with patch("ui.roi_editor.cv2.waitKey", side_effect=wait_key):
            result = session.run()

        self.assertEqual(result, UIAction.NONE)
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0][0], (1, 1, 19, 19))
        self.assertEqual(received[1][0], (41, 1, 19, 19))
        self.assertIs(received[0][1], session.frozen_tracks)
        self.assertEqual(received[0][2], EditMode.ADD_TARGETS)

    @patch("ui.roi_editor.cv2.imshow")
    @patch("ui.roi_editor.cv2.setMouseCallback")
    def test_gallery_session_can_submit_multiple_rois(
        self, set_mouse_callback, imshow
    ) -> None:
        received: list[tuple[int, EditMode]] = []
        session = self._session(
            lambda roi, tracks, mode: received.append((roi[0], mode)),
            mode=EditMode.ENROLL_GALLERY,
        )

        keys = [None, 13]

        def wait_key(_wait_ms: int) -> int:
            if keys.pop(0) is None:
                session._on_mouse(cv2.EVENT_LBUTTONDOWN, 20, 20, 0, None)
                session._on_mouse(cv2.EVENT_LBUTTONUP, 1, 1, 0, None)
                session._on_mouse(cv2.EVENT_LBUTTONDOWN, 60, 20, 0, None)
                session._on_mouse(cv2.EVENT_LBUTTONUP, 41, 1, 0, None)
                return 255
            return 13

        with patch("ui.roi_editor.cv2.waitKey", side_effect=wait_key):
            result = session.run()

        self.assertEqual(result, UIAction.NONE)
        self.assertEqual(received, [(1, EditMode.ENROLL_GALLERY), (41, EditMode.ENROLL_GALLERY)])


if __name__ == "__main__":
    unittest.main()
