from __future__ import annotations

import unittest

import numpy as np

from src.ascend_reid import AscendReIDExtractor, dynamic_batch_chunks, preprocess_reid_crop
from src.config import ReIDConfig


class _FakeModel:
    pass


class _FakeRuntime:
    def __init__(self) -> None:
        self.load_calls = 0
        self.execute_batches: list[int] = []

    def load_model(self, _path, *, dynamic_batch_sizes):
        self.load_calls += 1
        self.dynamic_batch_sizes = tuple(dynamic_batch_sizes)
        return _FakeModel()

    def execute(self, _model, inputs, *, dynamic_batch):
        batch = inputs[0]
        self.execute_batches.append(dynamic_batch)
        # Encode the first input pixel into an otherwise deterministic vector
        # so output order can be checked after chunking and concatenation.
        rows = []
        for sample in batch:
            row = np.zeros((512,), dtype=np.float32)
            row[0] = float(sample[0, 0, 0]) + 2.0
            row[1] = 1.0
            rows.append(row)
        return (np.asarray(rows, dtype=np.float32).reshape(-1),)


def _config() -> ReIDConfig:
    from pathlib import Path

    return ReIDConfig("osnet_x0_25", Path("unused.om"), 256, 128, 1, 1)


class AscendReIDTests(unittest.TestCase):
    def test_preprocess_converts_bgr_to_rgb_and_normalizes(self) -> None:
        crop = np.zeros((10, 20, 3), dtype=np.uint8)
        crop[:, :, 0] = 255  # BGR blue becomes RGB blue channel.

        tensor = preprocess_reid_crop(crop)

        self.assertEqual(tensor.shape, (3, 256, 128))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isclose(tensor[0, 0, 0], (0.0 - 0.485) / 0.229, atol=1e-5))
        self.assertTrue(np.isclose(tensor[2, 0, 0], (1.0 - 0.406) / 0.225, atol=1e-5))

    def test_dynamic_batch_chunks_do_not_use_fake_samples(self) -> None:
        self.assertEqual(dynamic_batch_chunks(13), (8, 4, 1))
        self.assertEqual(dynamic_batch_chunks(7), (4, 2, 1))
        self.assertEqual(dynamic_batch_chunks(0), ())

    def test_batch_extraction_preserves_order_and_loads_once(self) -> None:
        runtime = _FakeRuntime()
        extractor = AscendReIDExtractor(
            _config(),
            "unused.om",
            (1, 2, 4, 8),
            runtime,  # type: ignore[arg-type]
        )
        crops = []
        for value in (10, 20, 30, 40, 50):
            crop = np.zeros((20, 20, 3), dtype=np.uint8)
            crop[:, :, :] = value
            crops.append(crop)

        embeddings = extractor.extract_batch(crops)
        first_values = embeddings[:, 0]

        self.assertEqual(embeddings.shape, (5, 512))
        self.assertEqual(embeddings.dtype, np.float32)
        self.assertTrue(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5))
        self.assertEqual(runtime.load_calls, 1)
        self.assertEqual(runtime.execute_batches, [4, 1])
        self.assertEqual(extractor.inference_count, 2)
        self.assertLess(first_values[0], first_values[1])
        self.assertLess(first_values[1], first_values[2])
        self.assertLess(first_values[2], first_values[3])


if __name__ == "__main__":
    unittest.main()
