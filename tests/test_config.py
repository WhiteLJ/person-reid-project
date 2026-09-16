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
        self.assertEqual(config.inference.backend, "torch")
        self.assertEqual(config.ascend.device_id, 0)
        self.assertEqual(
            config.ascend.yolo_model,
            Path("weights/atlas/yolov8n.om").resolve(),
        )
        self.assertEqual(
            config.ascend.reid_model,
            Path("weights/atlas/osnet_x0_25.om").resolve(),
        )
        self.assertEqual(config.ascend.reid_dynamic_batches, (1, 2, 4, 8))
        self.assertEqual(
            config.tracking.tracker,
            str(Path("config/trackers/botsort_baseline.yaml").resolve()),
        )
        self.assertTrue(config.tracking.persist)
        self.assertAlmostEqual(config.selection.min_iou, 0.20)
        self.assertEqual(config.reid.model_name, "osnet_x0_25")
        self.assertEqual(
            config.reid.weight,
            Path("weights/reid/osnet_x0_25_msmt17.pth").resolve(),
        )
        self.assertEqual((config.reid.image_height, config.reid.image_width), (256, 128))
        self.assertEqual((config.reid.min_crop_width, config.reid.min_crop_height), (40, 100))
        self.assertEqual(config.reid_recovery.lost_grace_frames, 5)
        self.assertEqual(config.reid_recovery.reference_update_interval_frames, 15)
        self.assertEqual(config.reid_recovery.recovery_interval_frames, 5)
        self.assertEqual(config.reid_recovery.max_reference_embeddings, 8)
        self.assertAlmostEqual(config.reid_recovery.recovery_threshold, 0.80)
        self.assertAlmostEqual(config.reid_recovery.recovery_margin, 0.05)
        self.assertAlmostEqual(config.reid_recovery.reference_update_threshold, 0.80)
        self.assertEqual(config.reid_recovery.recovery_min_track_age_frames, 3)
        self.assertEqual(config.reid_recovery.recovery_confirmation_hits, 2)
        self.assertEqual(config.reid_recovery.recovery_pending_max_age_frames, 60)
        self.assertEqual(config.gallery_enrichment.post_recovery_stable_frames, 30)
        self.assertEqual(config.gallery_enrichment.max_reference_embeddings, 8)
        self.assertAlmostEqual(
            config.gallery_enrichment.duplicate_similarity_threshold, 0.95
        )
        self.assertAlmostEqual(config.reid_quality.min_track_confidence, 0.35)
        self.assertAlmostEqual(config.reid_quality.max_edge_truncation_ratio, 0.30)
        self.assertAlmostEqual(config.reid_quality.max_person_overlap_ratio, 0.60)
        self.assertAlmostEqual(config.reid_quality.min_frame_edge_margin_ratio, 0.01)
        self.assertTrue(config.gallery_recognition.enabled)
        self.assertEqual(config.gallery_recognition.recognition_interval_frames, 10)
        self.assertEqual(config.gallery_recognition.min_track_age_frames, 5)
        self.assertAlmostEqual(config.gallery_recognition.recognition_threshold, 0.80)
        self.assertAlmostEqual(config.gallery_recognition.recognition_margin, 0.05)
        self.assertEqual(config.gallery_recognition.confirmation_hits, 2)
        self.assertEqual(
            config.database.path,
            Path("database/person_reid.db").resolve(),
        )
        self.assertFalse(config.ui.show_unselected_tracks)
        self.assertTrue(config.diagnostics.enabled)
        self.assertEqual(config.diagnostics.log_interval_frames, 300)

    def test_auto_device_is_supported(self) -> None:
        self.assertIn(resolve_device("auto"), {"cpu", "cuda"})

    def test_atlas_config_selects_ascend_without_changing_business_paths(self) -> None:
        config = load_config(Path("config/config_atlas.yaml"))

        self.assertEqual(config.inference.backend, "ascend")
        self.assertEqual(config.model.device, "cpu")
        self.assertEqual(
            config.ascend.yolo_model,
            Path("weights/atlas/yolov8n.om").resolve(),
        )
        self.assertEqual(
            config.ascend.reid_model,
            Path("weights/atlas/osnet_x0_25.om").resolve(),
        )
        self.assertEqual(config.ascend.reid_dynamic_batches, (1, 2, 4, 8))
        self.assertEqual(config.reid_recovery.recovery_threshold, 0.80)
        self.assertEqual(config.gallery_recognition.recognition_threshold, 0.80)
        self.assertAlmostEqual(
            config.reid_quality.min_frame_edge_margin_ratio, 0.01
        )
        self.assertAlmostEqual(
            config.gallery_enrichment.duplicate_similarity_threshold, 0.95
        )


if __name__ == "__main__":
    unittest.main()
