from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.config import load_config
from src.models import Track
from src.target_manager import TargetManager
from src.visualization import (
    VEHICLE_COLOR,
    VEHICLE_SELECTED_COLOR,
    draw_multiclass_tracks,
)
from ui.opencv_ui import OpenCVUI


class PC7IntegrationTests(unittest.TestCase):
    def test_person_and_vehicle_duplicate_track_ids_are_rendered_from_separate_managers(self) -> None:
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        person_track = Track(4, (10, 10, 60, 80), 0.9, 0)
        vehicle_track = Track(4, (100, 10, 180, 80), 0.9, 2)
        person_manager = TargetManager()
        vehicle_manager = TargetManager()
        person_manager.select(person_track, np.asarray([1.0, 0.0], dtype=np.float32))

        result = draw_multiclass_tracks(
            frame,
            [person_track],
            [vehicle_track],
            person_manager,
            vehicle_manager,
            class_name=lambda class_id: {0: "person", 2: "car"}[class_id],
            show_unselected_tracks=True,
        )

        self.assertTrue(np.array_equal(result[10, 10], (0, 0, 255)))
        self.assertTrue(np.array_equal(result[10, 100], VEHICLE_COLOR))
        self.assertIsNone(vehicle_manager.target_for_track(4))

        vehicle_manager.select(vehicle_track, np.asarray([0.0, 1.0], dtype=np.float32))
        selected = draw_multiclass_tracks(
            frame,
            [person_track],
            [vehicle_track],
            person_manager,
            vehicle_manager,
            show_unselected_tracks=True,
        )
        self.assertTrue(np.array_equal(selected[10, 100], VEHICLE_SELECTED_COLOR))

    def test_formal_ui_scales_only_the_display_frame(self) -> None:
        config = load_config("config/config.yaml")
        ui = OpenCVUI(config.ui)
        source = np.zeros((1080, 1920, 3), dtype=np.uint8)

        with (
            patch("ui.opencv_ui.cv2.namedWindow"),
            patch("ui.opencv_ui.cv2.imshow") as imshow,
            patch("ui.opencv_ui.cv2.waitKey", return_value=-1),
        ):
            ui.show(source)

        displayed = imshow.call_args.args[1]
        self.assertEqual(displayed.shape[:2], (720, 1280))
        self.assertEqual(source.shape[:2], (1080, 1920))

    def test_pc_config_keeps_vehicle_classes_and_frozen_budgets(self) -> None:
        config = load_config("config/config.yaml")
        self.assertEqual(config.multiclass_tracking.vehicle_class_ids, (2, 5, 7))
        self.assertEqual(config.reid_recovery.recovery_candidates_per_frame, 3)
        self.assertEqual(config.gallery_recognition.recognition_candidates_per_frame, 3)
        self.assertEqual(config.vehicle_recovery.recovery_candidates_per_frame, 1)
        self.assertEqual(
            config.vehicle_gallery_recognition.recognition_candidates_per_frame,
            1,
        )

    def test_atlas_config_exposes_multiclass_vehicle_contract(self) -> None:
        config = load_config("config/config_atlas.yaml")
        self.assertEqual(config.inference.backend, "ascend")
        self.assertEqual(config.multiclass_tracking.vehicle_class_ids, (2, 5, 7))
        self.assertEqual(config.ascend.vehicle_reid_dynamic_batches, (1, 2, 4, 8))
        self.assertTrue(config.ui.show_unselected_tracks)


if __name__ == "__main__":
    unittest.main()
