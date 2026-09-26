from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class AtlasVehicleToolTests(unittest.TestCase):
    def test_atlas_smoke_parser_has_requested_defaults(self) -> None:
        from tools.atlas_vehicle_reid_smoke_test import build_parser

        args = build_parser().parse_args(["--image", "car.jpg"])
        self.assertEqual(args.batches, [1, 2, 4, 8])
        self.assertEqual(args.warmup, 3)
        self.assertEqual(args.repeat, 20)

    def test_pc_reference_generation_saves_embeddings_and_names(self) -> None:
        from tools.prepare_vehicle_atlas_reference import create_reference

        images = [
            np.zeros((20, 30, 3), dtype=np.uint8),
            np.ones((25, 35, 3), dtype=np.uint8),
        ]
        embeddings = np.zeros((2, 2048), dtype=np.float32)
        embeddings[:, 0] = 1.0
        with patch(
            "tools.prepare_vehicle_atlas_reference.cv2.imread",
            side_effect=images,
        ), patch(
            "tools.prepare_vehicle_atlas_reference.VehicleReIDExtractor"
        ) as extractor_class, patch(
            "tools.prepare_vehicle_atlas_reference.np.savez"
        ) as savez:
            extractor_class.return_value.extract_batch.return_value = embeddings
            result = create_reference(
                "config/config.yaml",
                ["data/car_a.jpg", "other/car_b.jpg"],
                Path("vehicle_reference.npz"),
            )

        self.assertEqual(result, Path("vehicle_reference.npz"))
        saved = savez.call_args.kwargs
        np.testing.assert_array_equal(saved["embeddings"], embeddings)
        self.assertEqual(saved["image_names"].tolist(), ["car_a.jpg", "car_b.jpg"])
        np.testing.assert_array_equal(
            saved["image_shapes"],
            np.asarray([[20, 30, 3], [25, 35, 3]], dtype=np.int32),
        )


if __name__ == "__main__":
    unittest.main()
