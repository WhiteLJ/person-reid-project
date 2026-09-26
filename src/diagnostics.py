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
        self._frame_total_seconds = 0.0
        self._tracking_seconds = 0.0
        self._recovery_seconds = 0.0
        self._gallery_seconds = 0.0
        self._identity_guard_seconds = 0.0
        self._render_seconds = 0.0
        self._ui_seconds = 0.0
        self._normal_frame_count = 0
        self._normal_frame_seconds = 0.0
        self._recovery_frame_count = 0
        self._recovery_frame_seconds = 0.0
        self._recovery_frame_max_seconds = 0.0
        self._recovery_candidate_count = 0
        self._recovery_quality_valid_count = 0
        self._recovery_reid_batch_count = 0
        self._recovery_reid_seconds = 0.0
        self._recovery_sweep_started_count = 0
        self._recovery_sweep_completed_count = 0
        self._recovery_sweep_candidate_count = 0
        self._recovery_sweep_processed_count = 0
        self._recovery_sweep_frame_count = 0
        self._recovery_sweep_reid_seconds = 0.0
        self._recognition_sweep_started_count = 0
        self._recognition_sweep_completed_count = 0
        self._recognition_sweep_candidate_count = 0
        self._recognition_sweep_processed_count = 0
        self._recognition_sweep_frame_count = 0
        self._recognition_sweep_reid_seconds = 0.0
        self._recognition_reid_batch_count = 0
        self._recognition_reid_seconds = 0.0
        self._recognition_retry_skipped_count = 0
        self._source_timing_totals: dict[str, float] = {}

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

    def record_frame(
        self,
        frame_seconds: float,
        frame_index: int,
        *,
        tracking_seconds: float = 0.0,
        recovery_seconds: float = 0.0,
        gallery_seconds: float = 0.0,
        identity_guard_seconds: float = 0.0,
        render_seconds: float = 0.0,
        ui_seconds: float = 0.0,
        recovery_due: bool = False,
        recovery_candidate_count: int = 0,
        recovery_quality_valid_count: int = 0,
        recovery_reid_batch_count: int = 0,
        recovery_reid_seconds: float = 0.0,
        recovery_sweep_started: bool = False,
        recovery_sweep_completed: bool = False,
        recovery_sweep_candidate_total: int = 0,
        recovery_sweep_processed_this_frame: int = 0,
        recovery_sweep_frames: int = 0,
        recovery_sweep_reid_ms: float = 0.0,
        recognition_reid_batch_count: int = 0,
        recognition_reid_seconds: float = 0.0,
        recognition_sweep_started: bool = False,
        recognition_sweep_completed: bool = False,
        recognition_sweep_candidate_total: int = 0,
        recognition_sweep_processed_this_frame: int = 0,
        recognition_sweep_frames: int = 0,
        recognition_sweep_reid_ms: float = 0.0,
        recognition_retry_skipped: int = 0,
        person_recovery_candidate_reid_ms: float = 0.0,
        person_reference_update_reid_ms: float = 0.0,
        person_guard_reid_ms: float = 0.0,
        person_recognition_reid_ms: float = 0.0,
        vehicle_recovery_candidate_reid_ms: float = 0.0,
        vehicle_reference_update_reid_ms: float = 0.0,
        vehicle_recognition_reid_ms: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        self.processed_frames += 1
        frame_seconds = max(0.0, frame_seconds)
        tracking_seconds = max(0.0, tracking_seconds)
        recovery_seconds = max(0.0, recovery_seconds)
        gallery_seconds = max(0.0, gallery_seconds)
        identity_guard_seconds = max(0.0, identity_guard_seconds)
        render_seconds = max(0.0, render_seconds)
        ui_seconds = max(0.0, ui_seconds)
        recovery_reid_seconds = max(0.0, recovery_reid_seconds)
        recovery_sweep_reid_ms = max(0.0, recovery_sweep_reid_ms)
        recognition_reid_seconds = max(0.0, recognition_reid_seconds)
        recognition_sweep_reid_ms = max(0.0, recognition_sweep_reid_ms)
        self._elapsed_seconds += frame_seconds
        self._frame_total_seconds += frame_seconds
        self._tracking_seconds += tracking_seconds
        self._recovery_seconds += recovery_seconds
        self._gallery_seconds += gallery_seconds
        self._identity_guard_seconds += identity_guard_seconds
        for name, value in {
            "person_recovery_candidate_reid_ms": person_recovery_candidate_reid_ms,
            "person_reference_update_reid_ms": person_reference_update_reid_ms,
            "person_guard_reid_ms": person_guard_reid_ms,
            "person_recognition_reid_ms": person_recognition_reid_ms,
            "vehicle_recovery_candidate_reid_ms": vehicle_recovery_candidate_reid_ms,
            "vehicle_reference_update_reid_ms": vehicle_reference_update_reid_ms,
            "vehicle_recognition_reid_ms": vehicle_recognition_reid_ms,
        }.items():
            self._source_timing_totals[name] = (
                self._source_timing_totals.get(name, 0.0) + max(0.0, value)
            )
        self._render_seconds += render_seconds
        self._ui_seconds += ui_seconds
        if recovery_due:
            self._recovery_frame_count += 1
            self._recovery_frame_seconds += frame_seconds
            self._recovery_frame_max_seconds = max(
                self._recovery_frame_max_seconds,
                frame_seconds,
            )
            self._recovery_candidate_count += max(0, recovery_candidate_count)
            self._recovery_quality_valid_count += max(
                0,
                recovery_quality_valid_count,
            )
            self._recovery_reid_batch_count += max(0, recovery_reid_batch_count)
            self._recovery_reid_seconds += recovery_reid_seconds
        else:
            self._normal_frame_count += 1
            self._normal_frame_seconds += frame_seconds
        if recovery_sweep_started:
            self._recovery_sweep_started_count += 1
            self._recovery_sweep_candidate_count += max(
                0, recovery_sweep_candidate_total
            )
        if recovery_sweep_completed:
            self._recovery_sweep_completed_count += 1
            self._recovery_sweep_frame_count += max(0, recovery_sweep_frames)
        self._recovery_sweep_processed_count += max(
            0, recovery_sweep_processed_this_frame
        )
        self._recovery_sweep_reid_seconds += recovery_sweep_reid_ms / 1000.0 if recovery_sweep_completed else 0.0
        self._recognition_reid_batch_count += max(0, recognition_reid_batch_count)
        self._recognition_reid_seconds += recognition_reid_seconds
        self._recognition_retry_skipped_count += max(0, recognition_retry_skipped)
        if recognition_sweep_started:
            self._recognition_sweep_started_count += 1
            self._recognition_sweep_candidate_count += max(
                0, recognition_sweep_candidate_total
            )
        if recognition_sweep_completed:
            self._recognition_sweep_completed_count += 1
            self._recognition_sweep_frame_count += max(0, recognition_sweep_frames)
        self._recognition_sweep_processed_count += max(
            0, recognition_sweep_processed_this_frame
        )
        self._recognition_sweep_reid_seconds += (
            recognition_sweep_reid_ms / 1000.0
            if recognition_sweep_completed
            else 0.0
        )
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
            "frame_total_ms": f"{self._mean_ms(self._frame_total_seconds, self.processed_frames):.2f}",
            "tracking_ms": f"{self._mean_ms(self._tracking_seconds, self.processed_frames):.2f}",
            "recovery_ms": f"{self._mean_ms(self._recovery_seconds, self.processed_frames):.2f}",
            "gallery_ms": f"{self._mean_ms(self._gallery_seconds, self.processed_frames):.2f}",
            "identity_guard_ms": f"{self._mean_ms(self._identity_guard_seconds, self.processed_frames):.2f}",
            "render_ms": f"{self._mean_ms(self._render_seconds, self.processed_frames):.2f}",
            "ui_ms": f"{self._mean_ms(self._ui_seconds, self.processed_frames):.2f}",
            "normal_frame_ms": f"{self._mean_ms(self._normal_frame_seconds, self._normal_frame_count):.2f}",
            "recovery_frame_ms": f"{self._mean_ms(self._recovery_frame_seconds, self._recovery_frame_count):.2f}",
            "recovery_frame_max_ms": f"{self._recovery_frame_max_seconds * 1000.0:.2f}",
            "recovery_candidate_count": self._recovery_candidate_count,
            "recovery_quality_valid_count": self._recovery_quality_valid_count,
            "recovery_reid_batch_count": self._recovery_reid_batch_count,
            "recovery_reid_ms": f"{self._mean_ms(self._recovery_reid_seconds, self._recovery_frame_count):.2f}",
            "recovery_sweep_started": self._recovery_sweep_started_count,
            "recovery_sweep_completed": self._recovery_sweep_completed_count,
            "recovery_sweep_candidate_total": self._recovery_sweep_candidate_count,
            "recovery_sweep_processed_this_frame": self._recovery_sweep_processed_count,
            "recovery_sweep_frames": self._recovery_sweep_frame_count,
            "recovery_sweep_reid_ms": f"{self._recovery_sweep_reid_seconds * 1000.0:.2f}",
            "recognition_reid_batches": self._recognition_reid_batch_count,
            "recognition_reid_ms": f"{self._recognition_reid_seconds * 1000.0:.2f}",
            "recognition_sweep_started": self._recognition_sweep_started_count,
            "recognition_sweep_completed": self._recognition_sweep_completed_count,
            "recognition_sweep_candidate_total": self._recognition_sweep_candidate_count,
            "recognition_sweep_processed_this_frame": self._recognition_sweep_processed_count,
            "recognition_sweep_frames": self._recognition_sweep_frame_count,
            "recognition_sweep_reid_ms": f"{self._recognition_sweep_reid_seconds * 1000.0:.2f}",
            "recognition_retry_skipped": self._recognition_retry_skipped_count,
        }
        for name, total_ms in self._source_timing_totals.items():
            values[name] = f"{total_ms / self.processed_frames:.2f}" if self.processed_frames else "0.00"
        values.update(counters)
        return " ".join(f"{key}={value}" for key, value in values.items())

    def log_summary(self, **counters: int) -> None:
        if self.enabled:
            LOGGER.info("CROWD_STATS %s", self.summary(**counters))

    @staticmethod
    def _mean_ms(total_seconds: float, count: int) -> float:
        return total_seconds * 1000.0 / count if count else 0.0
