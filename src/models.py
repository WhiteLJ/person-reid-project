"""Shared detection, tracking, and in-session target data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detector output in ``xyxy`` image coordinates."""

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


class TargetState(Enum):
    """State of a selected target during the current application run."""

    ACTIVE = auto()
    LOST = auto()


def _empty_embedding() -> np.ndarray:
    return np.empty((0,), dtype=np.float32)


@dataclass
class SessionTarget:
    """A temporary in-memory identity for one user-selected target.

    ``target_id`` is scoped to one application session.  It is deliberately
    not a persistent Person ID and is never stored in a database in MVP-5.
    ``last_track_id`` remains populated while the target is LOST so recovery
    logs can describe the old tracker binding.
    """

    target_id: int
    current_track_id: int | None
    last_track_id: int | None
    state: TargetState
    reference_embeddings: list[np.ndarray] = field(default_factory=list)
    centroid: np.ndarray = field(default_factory=_empty_embedding)
    missing_frames: int = 0
    last_reference_frame: int | None = None
    last_recovery_frame: int | None = None
