"""Data models shared by MVP-1 modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """A single person detection in ``xyxy`` image coordinates."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
