from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.video_source import VideoSource
from src.frame_source import SourceReadStatus


class _FakeCapture:
    def __init__(self) -> None:
        self.released = False
        self.fps = 30.0
        self.frames = [np.zeros((4, 5, 3), dtype=np.uint8)]

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self.frames:
            value = self.frames.pop(0)
            if isinstance(value, tuple) and len(value) == 2:
                return value
            return True, value
        return False, None

    def release(self) -> None:
        self.released = True

    def get(self, _property) -> float:
        return self.fps


class VideoSourceTests(unittest.TestCase):
    @patch("src.video_source.cv2.VideoCapture")
    def test_open_read_and_release(self, video_capture) -> None:
        fake_capture = _FakeCapture()
        video_capture.return_value = fake_capture
        source = VideoSource(0)

        source.open()
        frame = source.read()
        self.assertIsNotNone(frame)
        self.assertEqual(source.nominal_fps, 30.0)
        self.assertIsNone(source.read())
        source.release()

        video_capture.assert_called_once_with(0)
        self.assertTrue(fake_capture.released)

        self.assertEqual(source.read_status, SourceReadStatus.TEMPORARY_FAILURE)

    @patch("src.video_source.cv2.VideoCapture")
    def test_video_file_read_failure_is_eof(self, video_capture) -> None:
        fake_capture = _FakeCapture()
        fake_capture.frames = []
        video_capture.return_value = fake_capture
        source = VideoSource("video.mp4")

        source.open()
        self.assertIsNone(source.read())
        self.assertEqual(source.read_status, SourceReadStatus.EOF)

    @patch("src.video_source.cv2.VideoCapture")
    def test_camera_read_failure_is_temporary_and_can_reconnect(self, video_capture) -> None:
        failed_capture = _FakeCapture()
        failed_capture.frames = [(False, None)]
        recovered_capture = _FakeCapture()
        video_capture.side_effect = [failed_capture, recovered_capture]

        source = VideoSource(0)
        source.open()

        self.assertIsNone(source.read())
        self.assertEqual(source.read_status, SourceReadStatus.TEMPORARY_FAILURE)
        self.assertEqual(source.consecutive_read_failures, 1)
        self.assertTrue(source.reconnect())
        self.assertIsNotNone(source.read())
        self.assertEqual(source.read_status, SourceReadStatus.FRAME)
        self.assertEqual(video_capture.call_count, 2)

    @patch("src.video_source.cv2.VideoCapture")
    def test_invalid_camera_fps_is_exposed_as_unknown(self, video_capture) -> None:
        for fps in (0.0, float("nan")):
            fake_capture = _FakeCapture()
            fake_capture.fps = fps
            video_capture.return_value = fake_capture
            source = VideoSource(0)
            source.open()
            self.assertIsNone(source.nominal_fps)
            source.release()


if __name__ == "__main__":
    unittest.main()
