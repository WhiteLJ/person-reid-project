"""Read-only diagnostics for persistent Gallery ReID features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .feature_diversity import reference_similarity_stats
from .gallery import GalleryPerson
from .reid import cosine_similarity


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=np.float32)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def person_feature_diagnostics(
    person: GalleryPerson,
    *,
    duplicate_threshold: float = 0.98,
) -> dict[str, object]:
    """Return feature-bank statistics without changing the Gallery."""

    reference_stats = reference_similarity_stats(
        person.reference_embeddings,
        duplicate_threshold=duplicate_threshold,
    )
    reference_to_centroid = [
        cosine_similarity(reference, person.centroid)
        for reference in person.reference_embeddings
    ]
    return {
        "person_id": person.person_id,
        "label": person.label,
        "reference_count": len(person.reference_embeddings),
        "centroid_norm": float(np.linalg.norm(person.centroid)),
        "reference_pairwise": reference_stats,
        "reference_to_centroid": _summary(reference_to_centroid),
    }


def rank_candidate_against_gallery(
    people: Sequence[GalleryPerson],
    embedding: np.ndarray,
) -> list[tuple[int, float]]:
    """Return centroid ranking used by production recognition, read-only."""

    ranking = [
        (person.person_id, cosine_similarity(person.centroid, embedding))
        for person in people
    ]
    return sorted(ranking, key=lambda item: (-item[1], item[0]))
