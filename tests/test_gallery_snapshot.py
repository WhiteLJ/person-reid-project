from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.gallery_snapshot import crop_track_from_frame, encode_track_snapshot


class GallerySnapshotTests(unittest.TestCase):
    def test_snapshot_uses_track_bbox_and_is_jpeg(self) -> None:
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        frame[20:80, 40:120] = (0, 0, 255)
        crop = crop_track_from_frame(frame, (40, 20, 120, 80))
        self.assertEqual(crop.shape[:2], (60, 80))
        data = encode_track_snapshot(frame, (40, 20, 120, 80))
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape[:2], (60, 80))

    def test_invalid_bbox_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            crop_track_from_frame(np.zeros((10, 10, 3), dtype=np.uint8), (4, 4, 4, 5))


if __name__ == "__main__":
    unittest.main()
