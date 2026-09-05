"""Data models shared by the MVP-1 and MVP-2 modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """A single person detection in ``xyxy`` image coordinates."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int


@dataclass(frozen=True)
class Track:
    """A temporary BoT-SORT track; it is not a persistent Person ID."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
