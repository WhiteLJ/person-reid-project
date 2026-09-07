"""Strictly single-frame ReID embedding cache."""

from __future__ import annotations

import numpy as np


class ReIDFrameCache:
    """Share embeddings within one frame without permitting cross-frame reuse."""

    def __init__(self) -> None:
        self._frame_index: int | None = None
        self._embeddings: dict[int, np.ndarray] = {}

    @property
    def frame_index(self) -> int | None:
        return self._frame_index

    def begin_frame(self, frame_index: int) -> None:
        """Start a frame; changing frame index invalidates every old embedding."""

        if self._frame_index != frame_index:
            self._frame_index = frame_index
            self._embeddings.clear()

    def get(self, track_id: int, frame_index: int) -> np.ndarray | None:
        """Return a private copy only when the requested frame is current."""

        if self._frame_index != frame_index:
            return None
        embedding = self._embeddings.get(track_id)
        return None if embedding is None else embedding.copy()

    def put(self, track_id: int, embedding: np.ndarray, frame_index: int) -> None:
        """Store a private float32 copy for the current frame only."""

        if self._frame_index != frame_index:
            raise ValueError(
                "cannot store an embedding for a frame that is not current"
            )
        array = np.asarray(embedding, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError("cached embedding must be one-dimensional")
        self._embeddings[track_id] = array.copy()

    def clear(self) -> None:
        """Clear the cache and its frame identity."""

        self._frame_index = None
        self._embeddings.clear()
