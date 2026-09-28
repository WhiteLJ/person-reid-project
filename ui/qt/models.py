"""Qt-facing immutable data transfer objects.

The GUI receives these snapshots and never receives TargetManager, Gallery,
Repository, or AscendRuntime instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeFrame:
    frame_index: int
    annotated_bgr: Any
    source_width: int
    source_height: int
    person_tracks: tuple[Any, ...]
    vehicle_tracks: tuple[Any, ...]
    person_active_count: int = 0
    person_lost_count: int = 0
    vehicle_active_count: int = 0
    vehicle_lost_count: int = 0
    person_gallery_count: int = 0
    vehicle_gallery_count: int = 0
    fps: float = 0.0
    source_label: str = "Local Video"
    backend: str = "TORCH"


@dataclass(frozen=True)
class GalleryRecordDTO:
    domain: str
    identity_id: int
    display_id: str
    snapshot_jpeg: bytes | None = None


@dataclass(frozen=True)
class SelectionResultDTO:
    domain: str
    target_id: int
    track_id: int
    bbox: tuple[float, float, float, float]
    display_id: str | None = None
