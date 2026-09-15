from __future__ import annotations

import unittest

import numpy as np

from src.gallery import GalleryPerson
from src.gallery_diagnostics import (
    person_feature_diagnostics,
    rank_candidate_against_gallery,
)


def _unit(index: int) -> np.ndarray:
    value = np.zeros((512,), dtype=np.float32)
    value[index] = 1.0
    return value


class GalleryDiagnosticsTests(unittest.TestCase):
    def test_reference_bank_statistics_and_centroid_statistics(self) -> None:
        centroid = (_unit(0) + _unit(1)) / np.sqrt(2.0)
        person = GalleryPerson(
            person_id=1,
            label="Target P001",
            reference_embeddings=[_unit(0), _unit(1)],
            centroid=np.asarray(centroid, dtype=np.float32),
        )

        result = person_feature_diagnostics(person)

        self.assertEqual(result["reference_count"], 2)
        self.assertAlmostEqual(float(result["centroid_norm"]), 1.0, places=5)
        pairwise = result["reference_pairwise"]
        assert isinstance(pairwise, dict)
        self.assertEqual(pairwise["duplicate_pair_count"], 0)

    def test_ranking_uses_centroid_and_is_deterministic(self) -> None:
        people = [
            GalleryPerson(2, "Target P002", [_unit(1)], _unit(1)),
            GalleryPerson(1, "Target P001", [_unit(0)], _unit(0)),
        ]

        ranking = rank_candidate_against_gallery(people, _unit(0))

        self.assertEqual([person_id for person_id, _score in ranking], [1, 2])


if __name__ == "__main__":
    unittest.main()
