from __future__ import annotations

import unittest
from pathlib import Path

from src.frame_source import FrameSource
from src.source_factory import create_frame_source
from src.video_source import VideoSource


class FrameSourceTests(unittest.TestCase):
    def test_local_camera_and_file_use_video_source(self) -> None:
        camera = create_frame_source(0)
        video = create_frame_source(Path("data/demo.mp4"))

        self.assertIsInstance(camera, VideoSource)
        self.assertIsInstance(video, VideoSource)
        self.assertIsInstance(camera, FrameSource)
        self.assertEqual(camera.source_label, "Camera 0")
        self.assertEqual(video.source_label, "demo.mp4")

    def test_network_sources_are_reserved_without_connecting(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "reserved"):
            create_frame_source("rtsp://camera.example/live")


if __name__ == "__main__":
    unittest.main()
