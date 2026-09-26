"""Pure data passed from the application to the OpenCV Dashboard."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DashboardState:
    """Display-only runtime state; it owns no business objects."""

    backend: str = "TORCH"
    source_label: str = "Local Video"
    fps: float = 0.0
    paused: bool = False
    mode: str = "RUNNING"
    person_active_count: int = 0
    person_lost_count: int = 0
    vehicle_active_count: int = 0
    vehicle_lost_count: int = 0
    person_gallery_count: int = 0
    vehicle_gallery_count: int = 0
    frame_index: int = -1
    status_message: str = ""
