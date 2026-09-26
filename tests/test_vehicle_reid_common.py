from __future__ import annotations

import inspect
import unittest

import numpy as np

from src.vehicle_reid import VehicleReIDExtractor
from src.vehicle_reid_common import (
    normalize_vehicle_features,
    prepare_vehicle_batch,
    preprocess_vehicle_crop,
)


class VehicleReIDCommonTests(unittest.TestCase):
    def test_preprocessing_is_contiguous_float32_and_normalized(self) -> None:
        crop = np.zeros((23, 31, 3), dtype=np.uint8)
        crop[:, :, 0] = 255
        result = preprocess_vehicle_crop(crop)
        self.assertEqual(result.shape, (3, 256, 256))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        self.assertTrue(np.isfinite(result).all())

    def test_batch_preserves_order_and_shape(self) -> None:
        crops = [
            np.full((20, 30, 3), value, dtype=np.uint8)
            for value in (10, 20, 30)
        ]
        result = prepare_vehicle_batch(crops)
        self.assertEqual(result.shape, (3, 3, 256, 256))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        self.assertLess(float(result[0].mean()), float(result[1].mean()))

    def test_empty_batch_is_explicit(self) -> None:
        result = prepare_vehicle_batch([])
        self.assertEqual(result.shape, (0, 3, 256, 256))
        self.assertEqual(result.dtype, np.float32)

    def test_pc_private_preprocess_uses_the_same_formula(self) -> None:
        from pathlib import Path
        from src.config import VehicleReIDConfig

        config = VehicleReIDConfig(
            enabled=True,
            model_name="test",
            weight=Path("unused.pth"),
            config=Path("unused.yml"),
            device="cpu",
            image_height=256,
            image_width=256,
        )
        extractor = VehicleReIDExtractor(config, model=object())
        crop = np.full((18, 27, 3), 90, dtype=np.uint8)
        np.testing.assert_allclose(
            extractor._preprocess(crop).numpy(),
            preprocess_vehicle_crop(crop),
            rtol=0.0,
            atol=0.0,
        )

    def test_normalization_enforces_optional_vehicle_dimension(self) -> None:
        result = normalize_vehicle_features(
            np.ones((2, 2048), dtype=np.float32),
            expected_dim=2048,
        )
        self.assertEqual(result.shape, (2, 2048))
        self.assertTrue(np.allclose(np.linalg.norm(result, axis=1), 1.0))
        with self.assertRaises(ValueError):
            normalize_vehicle_features(np.ones((1, 512), dtype=np.float32), expected_dim=2048)

    def test_torch_free_modules_do_not_reference_torch(self) -> None:
        import src.ascend_vehicle_reid as ascend_vehicle_reid
        import src.vehicle_reid_common as common

        self.assertNotIn("import torch", inspect.getsource(common))
        self.assertNotIn("import torch", inspect.getsource(ascend_vehicle_reid))
        self.assertNotIn("src.vehicle_reid", inspect.getsource(ascend_vehicle_reid))


if __name__ == "__main__":
    unittest.main()
