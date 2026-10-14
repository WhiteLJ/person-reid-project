from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.video_source import VideoSource
from src.frame_source import SourceReadStatus


class _FakeCapture:
    def __init__(self) -> None:
        self.released = False
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


class VideoSourceTests(unittest.TestCase):
    @patch("src.video_source.cv2.VideoCapture")
    def test_open_read_and_release(self, video_capture) -> None:
        fake_capture = _FakeCapture()
        video_capture.return_value = fake_capture
        source = VideoSource(0)

        source.open()
        frame = source.read()
        self.assertIsNotNone(frame)
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


if __name__ == "__main__":
    unittest.main()
