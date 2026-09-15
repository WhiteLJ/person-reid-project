"""Small, deterministic utilities for Gallery reference-bank diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize feature vectors without importing the neural backend module."""

    value = np.asarray(embedding, dtype=np.float32)
    if value.ndim not in (1, 2) or value.size == 0 or not np.isfinite(value).all():
        raise ValueError("embedding must be a finite non-empty vector or matrix")
    axis = 0 if value.ndim == 1 else 1
    norms = np.linalg.norm(value, axis=axis, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("embedding must not contain a zero vector")
    return (value / norms).astype(np.float32, copy=False)


def _normalized_references(
    references: Iterable[np.ndarray],
) -> list[np.ndarray]:
    normalized: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for index, reference in enumerate(references):
        try:
            value = _normalize_embedding(reference)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid reference embedding at index {index}") from exc
        if value.ndim != 1:
            raise ValueError("reference embeddings must have shape (D,)")
        if expected_shape is None:
            expected_shape = value.shape
        elif value.shape != expected_shape:
            raise ValueError("reference embeddings must have matching dimensions")
        normalized.append(value.astype(np.float32, copy=True))
    return normalized


def reference_similarity_stats(
    references: Iterable[np.ndarray],
    *,
    duplicate_threshold: float = 0.98,
) -> dict[str, int | float | None]:
    """Return pairwise cosine and near-duplicate statistics for one bank."""

    if not 0.0 < duplicate_threshold <= 1.0:
        raise ValueError("duplicate_threshold must be in (0, 1]")
    values = _normalized_references(references)
    similarities = [
        float(np.dot(values[left], values[right]))
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    return {
        "count": len(values),
        "pair_count": len(similarities),
        "min": min(similarities) if similarities else None,
        "mean": float(np.mean(similarities)) if similarities else None,
        "max": max(similarities) if similarities else None,
        "duplicate_pair_count": sum(
            similarity >= duplicate_threshold for similarity in similarities
        ),
    }


def filter_diverse_references(
    references: Iterable[np.ndarray],
    *,
    duplicate_threshold: float,
) -> list[np.ndarray]:
    """Greedily retain high-quality references that are not near duplicates.

    The first reference is always retained. Subsequent references are compared
    with already-retained references in stable input order. This function is
    intended for persistent enrichment only; it must not be used to alter the
    runtime SessionTarget bank used by Recovery.
    """

    if not 0.0 < duplicate_threshold <= 1.0:
        raise ValueError("duplicate_threshold must be in (0, 1]")
    values = _normalized_references(references)
    retained: list[np.ndarray] = []
    for value in values:
        if not retained:
            retained.append(value.copy())
            continue
        maximum_similarity = max(float(np.dot(value, existing)) for existing in retained)
        if maximum_similarity < duplicate_threshold:
            retained.append(value.copy())
    return retained


def normalized_centroid(references: Iterable[np.ndarray]) -> np.ndarray:
    """Compute a normalized centroid from a non-empty reference bank."""

    values = _normalized_references(references)
    if not values:
        raise ValueError("at least one reference is required")
    return _normalize_embedding(np.mean(np.stack(values, axis=0), axis=0)).astype(
        np.float32,
        copy=True,
    )
