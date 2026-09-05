"""Temporary multi-target selection state for MVP-3."""

from __future__ import annotations

from .models import Track


class TargetManager:
    """Maintain the user-selected set of temporary BoT-SORT Track IDs."""

    def __init__(self) -> None:
        self.selected_track_ids: set[int] = set()

    def select(self, track: Track) -> None:
        """Add a Track ID to the selected set, idempotently."""

        self.selected_track_ids.add(track.track_id)

    def deselect(self, track: Track) -> None:
        """Remove one Track ID if it is selected."""

        self.selected_track_ids.discard(track.track_id)

    def clear(self) -> None:
        """Clear all selected Track IDs."""

        self.selected_track_ids.clear()

    def is_selected(self, track_id: int) -> bool:
        return track_id in self.selected_track_ids

    def selected_tracks(self, tracks: list[Track]) -> list[Track]:
        """Return only currently visible Tracks whose IDs are selected."""

        return [track for track in tracks if self.is_selected(track.track_id)]
