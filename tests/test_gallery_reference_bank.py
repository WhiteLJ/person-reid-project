from __future__ import annotations

import unittest

import numpy as np

from src.gallery_reference_bank import (
    normalized_centroid,
    pairwise_cosine_matrix,
    update_persistent_reference_bank,
)


def _basis(index: int) -> np.ndarray:
    value = np.zeros((512,), dtype=np.float32)
    value[index] = 1.0
    return value


def _near(first: int, second: int, weight: float = 0.05) -> np.ndarray:
    value = _basis(first) + weight * _basis(second)
    return value / np.linalg.norm(value)


class PersistentReferenceBankTests(unittest.TestCase):
    def test_duplicate_late_pose_does_not_displace_anchor_or_diversity(self) -> None:
        bank = [_basis(index) for index in range(8)]
        late_pose = _near(7, 8, weight=0.02)

        updated, changed = update_persistent_reference_bank(
            bank,
            late_pose,
            max_reference_embeddings=8,
            duplicate_similarity_threshold=0.97,
        )

        self.assertFalse(changed)
        self.assertEqual(len(updated), 8)
        self.assertTrue(np.allclose(updated[0], _basis(0)))
        for index in range(8):
            self.assertTrue(np.allclose(updated[index], bank[index]))

    def test_full_bank_replaces_most_redundant_non_anchor(self) -> None:
        bank = [
            _basis(0),
            _near(1, 2),
            _near(1, 2, weight=0.06),
            _basis(3),
            _basis(4),
            _basis(5),
            _basis(6),
            _basis(7),
        ]
        candidate = _basis(8)

        updated, changed = update_persistent_reference_bank(
            bank,
            candidate,
            max_reference_embeddings=8,
            duplicate_similarity_threshold=0.97,
        )

        self.assertTrue(changed)
        self.assertEqual(len(updated), 8)
        self.assertTrue(np.allclose(updated[0], bank[0]))
        self.assertTrue(any(np.allclose(value, candidate) for value in updated))

    def test_multiple_late_samples_from_same_pose_do_not_fill_bank(self) -> None:
        bank = [_basis(index) for index in range(7)]
        first_new_pose = _basis(7)
        bank, changed = update_persistent_reference_bank(
            bank,
            first_new_pose,
            max_reference_embeddings=8,
        )
        self.assertTrue(changed)

        for _ in range(20):
            bank, changed = update_persistent_reference_bank(
                bank,
                _near(7, 9, weight=0.01),
                max_reference_embeddings=8,
            )
            self.assertFalse(changed)

        self.assertEqual(len(bank), 8)
        self.assertTrue(np.allclose(bank[0], _basis(0)))
        for index in range(7):
            self.assertTrue(np.allclose(bank[index], _basis(index)))

    def test_centroid_and_pairwise_matrix_are_normalized(self) -> None:
        references = [_basis(0), _basis(1), _basis(2)]
        centroid = normalized_centroid(references)
        matrix = pairwise_cosine_matrix(references)

        self.assertEqual(centroid.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=5)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertTrue(np.allclose(np.diag(matrix), 1.0))
        self.assertTrue(np.allclose(matrix, matrix.T))


if __name__ == "__main__":
    unittest.main()
