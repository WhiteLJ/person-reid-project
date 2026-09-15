from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tools.reid_backend_parity import (
    load_reference,
    pairwise_similarity_matrix,
    save_reference,
)


class ReIDBackendParityTests(unittest.TestCase):
    def test_reference_round_trip_does_not_use_object_arrays(self) -> None:
        crops = (
            np.zeros((10, 8, 3), dtype=np.uint8),
            np.full((12, 9, 3), 7, dtype=np.uint8),
        )
        torch_values = np.zeros((2, 512), dtype=np.float32)
        onnx_values = np.zeros((2, 512), dtype=np.float32)
        torch_values[:, 0] = 1.0
        onnx_values[:, 0] = 1.0

        path = Path(".test_tmp") / "reid_backend_parity_reference.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            save_reference(path, ("A1", "B1"), crops, torch_values, onnx_values)
            names, loaded_crops, loaded_torch, loaded_onnx = load_reference(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(names, ("A1", "B1"))
        self.assertEqual([crop.shape for crop in loaded_crops], [(10, 8, 3), (12, 9, 3)])
        self.assertEqual(loaded_torch.shape, (2, 512))
        self.assertEqual(loaded_onnx.shape, (2, 512))

    def test_pairwise_similarity_matrix_is_normalized(self) -> None:
        values = np.zeros((2, 512), dtype=np.float32)
        values[0, 0] = 2.0
        values[1, 1] = 3.0

        matrix = pairwise_similarity_matrix(values)

        self.assertTrue(np.allclose(np.diag(matrix), 1.0))
        self.assertAlmostEqual(float(matrix[0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
