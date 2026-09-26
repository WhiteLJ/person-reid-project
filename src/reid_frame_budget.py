"""Synchronous per-domain budget for new ReID embeddings."""

from __future__ import annotations


class ReIDFrameBudget:
    """Limit newly computed embeddings for one domain in one video frame.

    The object is intentionally single-threaded.  ``begin_frame`` is
    idempotent for the current frame so Recovery and Gallery Recognition can
    share one budget without resetting each other.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("ReID frame budget capacity must be positive")
        self.capacity = int(capacity)
        self.frame_index: int | None = None
        self.used = 0

    @property
    def remaining(self) -> int:
        return self.capacity - self.used

    def begin_frame(self, frame_index: int) -> None:
        if self.frame_index != frame_index:
            self.frame_index = frame_index
            self.used = 0

    def can_consume(self, count: int = 1) -> bool:
        if count < 0:
            raise ValueError("budget count must not be negative")
        return self.used + count <= self.capacity

    def consume(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("budget count must not be negative")
        if not self.can_consume(count):
            raise RuntimeError("ReID frame budget exhausted")
        self.used += count
