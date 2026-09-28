from __future__ import annotations

import unittest

from src.frame_source import FrameSource
from src.source_factory import create_frame_source
from src.video_source import VideoSource


class FrameSourceTests(unittest.TestCase):
    def test_video_source_implements_reserved_contract(self) -> None:
        source = create_frame_source(0)
        self.assertIsInstance(source, VideoSource)
        self.assertIsInstance(source, FrameSource)
        self.assertEqual(source.source_label, "Camera 0")
        self.assertTrue(all(hasattr(source, name) for name in ("open", "read", "release")))

    def test_protocol_is_importable_without_network_camera(self) -> None:
        self.assertIsNotNone(FrameSource)


if __name__ == "__main__":
    unittest.main()
