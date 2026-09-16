"""Bounded representative reference-bank operations for persistent Gallery data.

The SessionTarget reference bank is intentionally a separate runtime data
structure.  This module only manages the persistent Gallery bank used by
enrichment and therefore never applies FIFO eviction.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .reid import cosine_similarity, normalize_embedding


def normalized_centroid(references: Iterable[np.ndarray]) -> np.ndarray:
    """Return the normalized centroid of a non-empty reference sequence."""

    values = [
        normalize_embedding(reference).astype(np.float32, copy=True)
        for reference in references
    ]
    if not values:
        raise ValueError("reference bank must contain at least one embedding")
    shape = values[0].shape
    if any(value.shape != shape for value in values[1:]):
        raise ValueError("reference embeddings must have matching dimensions")
    return normalize_embedding(np.mean(np.stack(values, axis=0), axis=0)).astype(
        np.float32,
        copy=True,
    )


def pairwise_cosine_matrix(references: Iterable[np.ndarray]) -> np.ndarray:
    """Return an NxN cosine matrix for a reference bank."""

    values = [normalize_embedding(reference) for reference in references]
    if not values:
        return np.empty((0, 0), dtype=np.float32)
    shape = values[0].shape
    if any(value.ndim != 1 or value.shape != shape for value in values):
        raise ValueError("reference embeddings must be matching 1-D vectors")
    matrix = np.stack(values, axis=0)
    return np.matmul(matrix, matrix.T).astype(np.float32, copy=False)


def reference_redundancies(references: Iterable[np.ndarray]) -> tuple[float, ...]:
    """Return each reference's maximum similarity to a different reference."""

    matrix = pairwise_cosine_matrix(references)
    if matrix.shape[0] == 1:
        return (0.0,)
    values: list[float] = []
    for index in range(matrix.shape[0]):
        others = np.delete(matrix[index], index)
        values.append(float(np.max(others)))
    return tuple(values)


def update_persistent_reference_bank(
    references: Iterable[np.ndarray],
    candidate: np.ndarray,
    *,
    max_reference_embeddings: int = 8,
    duplicate_similarity_threshold: float = 0.97,
) -> tuple[list[np.ndarray], bool]:
    """Add or replace one representative persistent reference.

    Index zero is the enrollment-time anchor and is never evicted.  A
    candidate that is already too similar to any current reference is ignored.
    If the bank is full, the most redundant non-anchor reference is replaced
    only when the candidate has strictly lower redundancy than that victim.
    Every returned array is a detached float32 normalized copy.
    """

    if max_reference_embeddings < 1:
        raise ValueError("max_reference_embeddings must be positive")
    if not 0.0 < duplicate_similarity_threshold <= 1.0:
        raise ValueError("duplicate_similarity_threshold must be in (0, 1]")

    bank = [
        normalize_embedding(reference).astype(np.float32, copy=True)
        for reference in references
    ]
    if len(bank) > max_reference_embeddings:
        raise ValueError(
            "persistent reference bank exceeds max_reference_embeddings"
        )

    normalized_candidate = normalize_embedding(candidate).astype(
        np.float32,
        copy=True,
    )
    if bank and any(reference.shape != normalized_candidate.shape for reference in bank):
        raise ValueError("candidate and reference embeddings must match dimensions")

    if not bank:
        return [normalized_candidate], True

    similarities = np.asarray(
        [cosine_similarity(normalized_candidate, reference) for reference in bank],
        dtype=np.float32,
    )
    if float(np.max(similarities)) >= duplicate_similarity_threshold:
        return bank, False

    if len(bank) < max_reference_embeddings:
        bank.append(normalized_candidate)
        return bank, True

    # A full one-element bank has only its protected anchor and cannot replace
    # anything without losing the original enrollment feature.
    if len(bank) == 1:
        return bank, False

    redundancies = reference_redundancies(bank)
    # Stable tie-breaking keeps behavior deterministic: retain the earlier
    # reference when redundancy is tied.
    victim_index = max(
        range(1, len(bank)),
        key=lambda index: (redundancies[index], -index),
    )
    candidate_redundancy = float(np.max(similarities))
    victim_redundancy = redundancies[victim_index]
    if candidate_redundancy >= victim_redundancy:
        return bank, False

    bank[victim_index] = normalized_candidate
    return bank, True

