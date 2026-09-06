from __future__ import annotations

import unittest
from pathlib import Path

from src.config import load_config, parse_source, resolve_device


class ConfigTests(unittest.TestCase):
    def test_parse_source_supports_camera_index_and_path(self) -> None:
        self.assertEqual(parse_source(0), 0)
        self.assertEqual(parse_source("2"), 2)
        self.assertEqual(parse_source("videos/demo.mp4"), "videos/demo.mp4")

    def test_load_config_uses_zero_workers_by_default(self) -> None:
        config = load_config(Path("config/config.yaml"))
        self.assertEqual(config.runtime.num_workers, 0)
        self.assertEqual(config.video.source, 0)
        self.assertEqual(config.tracking.tracker, "botsort.yaml")
        self.assertTrue(config.tracking.persist)
        self.assertAlmostEqual(config.selection.min_iou, 0.20)
        self.assertEqual(config.reid.model_name, "osnet_x0_25")
        self.assertEqual(
            config.reid.weight,
            Path("weights/reid/osnet_x0_25_msmt17.pth").resolve(),
        )
        self.assertEqual((config.reid.image_height, config.reid.image_width), (256, 128))
        self.assertEqual((config.reid.min_crop_width, config.reid.min_crop_height), (40, 100))
        self.assertFalse(config.ui.show_unselected_tracks)

    def test_auto_device_is_supported(self) -> None:
        self.assertIn(resolve_device("auto"), {"cpu", "cuda"})


if __name__ == "__main__":
    unittest.main()
