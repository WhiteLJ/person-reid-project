from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.video_source import VideoSource


class _FakeCapture:
    def __init__(self) -> None:
        self.released = False
        self.frames = [np.zeros((4, 5, 3), dtype=np.uint8)]

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
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


if __name__ == "__main__":
    unittest.main()
