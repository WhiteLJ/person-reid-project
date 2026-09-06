from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.config import ReIDConfig
from src.reid import (
    ReIDExtractor,
    cosine_similarity,
    crop_person,
    normalize_embedding,
)


class _FakeFeatureExtractor:
    def __init__(self) -> None:
        self.calls: list[list[np.ndarray]] = []

    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        self.calls.append(images)
        features = np.zeros((len(images), 512), dtype=np.float64)
        for index, image in enumerate(images):
            features[index, :3] = image[0, 0].astype(np.float64)
            features[index, 3] = index + 1
        return features


def _config(weight: Path | None = None) -> ReIDConfig:
    return ReIDConfig(
        model_name="osnet_x0_25",
        weight=weight or Path("missing-reid-checkpoint.pth"),
        image_height=256,
        image_width=128,
        min_crop_width=40,
        min_crop_height=100,
    )


class CropTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((120, 100, 3), dtype=np.uint8)

    def test_crop_person_returns_clamped_crop(self) -> None:
        crop = crop_person(
            self.frame,
            (-10, -5, 50, 130),
            min_crop_width=40,
            min_crop_height=100,
        )

        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual(crop.shape, (120, 50, 3))

    def test_crop_person_rejects_invalid_or_outside_bbox(self) -> None:
        self.assertIsNone(
            crop_person(
                self.frame,
                (10, 10, 10, 80),
                min_crop_width=1,
                min_crop_height=1,
            )
        )
        self.assertIsNone(
            crop_person(
                self.frame,
                (110, 10, 130, 80),
                min_crop_width=1,
                min_crop_height=1,
            )
        )

    def test_crop_person_filters_small_crop(self) -> None:
        self.assertIsNone(
            crop_person(
                self.frame,
                (0, 0, 39, 99),
                min_crop_width=40,
                min_crop_height=100,
            )
        )


class EmbeddingTests(unittest.TestCase):
    def test_normalize_and_cosine_similarity(self) -> None:
        normalized = normalize_embedding(np.array([3.0, 4.0]))

        self.assertEqual(normalized.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0, places=6)
        self.assertAlmostEqual(cosine_similarity(normalized, normalized), 1.0, places=6)

    def test_extract_converts_bgr_to_rgb_and_returns_single_embedding(self) -> None:
        fake = _FakeFeatureExtractor()
        extractor = ReIDExtractor(_config(), "cpu", feature_extractor=fake)
        crop = np.zeros((100, 40, 3), dtype=np.uint8)
        crop[0, 0] = (1, 2, 3)  # BGR

        embedding = extractor.extract(crop)

        self.assertEqual(embedding.shape, (512,))
        self.assertEqual(embedding.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=6)
        np.testing.assert_array_equal(fake.calls[0][0][0, 0], (3, 2, 1))

    def test_extract_batch_preserves_order_and_shape(self) -> None:
        fake = _FakeFeatureExtractor()
        extractor = ReIDExtractor(_config(), "cpu", feature_extractor=fake)
        first = np.zeros((100, 40, 3), dtype=np.uint8)
        second = np.zeros((100, 40, 3), dtype=np.uint8)
        first[0, 0] = (0, 0, 10)
        second[0, 0] = (0, 20, 0)

        embeddings = extractor.extract_batch([first, second])

        self.assertEqual(embeddings.shape, (2, 512))
        self.assertEqual(embeddings.dtype, np.float32)
        self.assertTrue(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0))
        np.testing.assert_array_equal(fake.calls[0][0][0, 0], (10, 0, 0))
        np.testing.assert_array_equal(fake.calls[0][1][0, 0], (0, 20, 0))
        self.assertGreater(embeddings[0, 0], embeddings[1, 0])

    def test_empty_batch_has_fixed_embedding_dimension(self) -> None:
        extractor = ReIDExtractor(_config(), "cpu", feature_extractor=_FakeFeatureExtractor())

        embeddings = extractor.extract_batch([])

        self.assertEqual(embeddings.shape, (0, 512))
        self.assertEqual(embeddings.dtype, np.float32)

    def test_feature_extractor_is_created_once(self) -> None:
        fake = _FakeFeatureExtractor()
        with patch.object(Path, "is_file", return_value=True), patch.object(
            ReIDExtractor,
            "_create_official_extractor",
            return_value=fake,
        ) as create_extractor, patch.object(
            ReIDExtractor,
            "_audit_loaded_checkpoint",
        ):
            extractor = ReIDExtractor(_config(), "cpu")
            crop = np.ones((100, 40, 3), dtype=np.uint8)
            extractor.extract(crop)
            extractor.extract(crop)

        create_extractor.assert_called_once()

    def test_missing_real_checkpoint_fails_before_model_creation(self) -> None:
        with patch.object(ReIDExtractor, "_create_official_extractor") as create_extractor:
            with self.assertRaises(FileNotFoundError):
                ReIDExtractor(_config(), "cpu")

        create_extractor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
