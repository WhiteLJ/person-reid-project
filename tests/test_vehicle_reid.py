from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from src.config import VehicleReIDConfig
from src.vehicle_reid import VehicleReIDExtractor


class _FakeVehicleModel(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # Produce a deterministic four-dimensional feature without assuming
        # the production model's real embedding dimension.
        means = batch.mean(dim=(2, 3))
        return torch.cat(
            (means[:, :3], means[:, :1].abs() + 1.0),
            dim=1,
        )


def _config() -> VehicleReIDConfig:
    return VehicleReIDConfig(
        enabled=True,
        model_name="fake_vehicle_model",
        weight=Path("unused.pth"),
        config=Path("unused.yml"),
        device="cpu",
        image_height=16,
        image_width=8,
    )


class VehicleReIDExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = VehicleReIDExtractor(
            _config(),
            model=_FakeVehicleModel(),
        )
        self.crop_a = np.full((40, 30, 3), 80, dtype=np.uint8)
        self.crop_b = np.full((24, 50, 3), 120, dtype=np.uint8)

    def test_invalid_crop_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.extractor.extract(np.zeros((20, 20), dtype=np.uint8))
        with self.assertRaises(ValueError):
            self.extractor.extract(np.zeros((0, 20, 3), dtype=np.uint8))

    def test_empty_batch_has_explicit_empty_shape(self) -> None:
        empty = self.extractor.extract_batch([])
        self.assertEqual(empty.shape, (0, 0))
        self.assertEqual(empty.dtype, np.float32)

    def test_embedding_is_float32_finite_and_normalized(self) -> None:
        embedding = self.extractor.extract(self.crop_a)
        self.assertEqual(embedding.shape, (4,))
        self.assertEqual(embedding.dtype, np.float32)
        self.assertTrue(np.isfinite(embedding).all())
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)

    def test_single_and_batch_preserve_order_and_match(self) -> None:
        single_a = self.extractor.extract(self.crop_a)
        batch = self.extractor.extract_batch([self.crop_a, self.crop_b])
        np.testing.assert_allclose(single_a, batch[0], rtol=1e-5, atol=1e-6)
        self.assertEqual(batch.shape, (2, 4))

    def test_vehicle_output_dimension_is_not_person_512_contract(self) -> None:
        embeddings = self.extractor.extract_batch([self.crop_a, self.crop_b])
        self.assertEqual(embeddings.shape[1], 4)
        self.assertNotEqual(embeddings.shape[1], 512)


if __name__ == "__main__":
    unittest.main()
