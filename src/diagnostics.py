"""Small runtime counters for crowded-scene regression runs."""

from __future__ import annotations

from collections.abc import Sequence
from logging import getLogger

from .models import Track


LOGGER = getLogger(__name__)


class RuntimeDiagnostics:
    """Collect cheap Track lifecycle and frame-rate statistics.

    The counters intentionally do not label a Track ID change as a confirmed
    fragmentation or identity error.  Without ground truth, the application
    can safely report Track creation/end events and target recovery events;
    false recovery remains a human benchmark annotation.
    """

    def __init__(self, *, enabled: bool = True, log_interval_frames: int = 300) -> None:
        if log_interval_frames < 1:
            raise ValueError("log_interval_frames must be positive")
        self.enabled = enabled
        self.log_interval_frames = log_interval_frames
        self.processed_frames = 0
        self.unique_track_ids: set[int] = set()
        self.track_created_count = 0
        self.track_ended_count = 0
        self._previous_track_ids: set[int] = set()
        self._elapsed_seconds = 0.0

    def observe_tracks(self, tracks: Sequence[Track], frame_index: int) -> None:
        """Record Track IDs entering/leaving the current result set."""

        if not self.enabled:
            return
        current_ids = {track.track_id for track in tracks}
        created = sorted(current_ids - self._previous_track_ids)
        ended = sorted(self._previous_track_ids - current_ids)
        for track_id in created:
            self.track_created_count += 1
            self.unique_track_ids.add(track_id)
            LOGGER.debug("TRACK_CREATED track=%d frame=%d", track_id, frame_index)
        for track_id in ended:
            self.track_ended_count += 1
            LOGGER.debug("TRACK_ENDED track=%d frame=%d", track_id, frame_index)
        self.unique_track_ids.update(current_ids)
        self._previous_track_ids = current_ids

    def record_frame(self, frame_seconds: float, frame_index: int) -> None:
        if not self.enabled:
            return
        self.processed_frames += 1
        self._elapsed_seconds += max(0.0, frame_seconds)
        if self.processed_frames % self.log_interval_frames == 0:
            LOGGER.info("CROWD_STATS %s", self.summary())

    def summary(self, **counters: int) -> str:
        """Return a compact, log-friendly summary with optional coordinator counts."""

        average_fps = (
            self.processed_frames / self._elapsed_seconds
            if self._elapsed_seconds > 0.0
            else 0.0
        )
        values = {
            "frames": self.processed_frames,
            "unique_tracks": len(self.unique_track_ids),
            "track_created": self.track_created_count,
            "track_ended": self.track_ended_count,
            "average_fps": f"{average_fps:.2f}",
        }
        values.update(counters)
        return " ".join(f"{key}={value}" for key, value in values.items())

    def log_summary(self, **counters: int) -> None:
        if self.enabled:
            LOGGER.info("CROWD_STATS %s", self.summary(**counters))
