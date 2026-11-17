"""Bounded background recorder for rolling raw-camera replay segments."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from queue import Empty, Full, Queue
import re
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)

_MANAGED_FILE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    r"(?:_(?P<suffix>\d+))?\.mp4$"
)


def _default_wall_clock() -> datetime:
    return datetime.now().astimezone()


def _default_writer(path: Path, fourcc: int, fps: float, size: tuple[int, int]):
    return cv2.VideoWriter(str(path), fourcc, fps, size)


class RollingReplayRecorder:
    """Write raw frames on a bounded worker queue and rotate by wall time.

    The recorder never opens a camera.  The caller submits copies of frames
    already read by the application source.  Queue pressure drops the oldest
    pending frame so the runtime thread never waits for disk I/O.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        segment_minutes: float = 10.0,
        retention_segments: int = 4,
        codec: str = "mp4v",
        fallback_fps: float = 25.0,
        queue_size: int = 120,
        monotonic_clock: Callable[[], float] = monotonic,
        wall_clock: Callable[[], datetime] = _default_wall_clock,
        writer_factory: Callable[[Path, int, float, tuple[int, int]], object]
        | None = None,
    ) -> None:
        if segment_minutes <= 0.0:
            raise ValueError("segment_minutes must be positive")
        if retention_segments < 1:
            raise ValueError("retention_segments must be positive")
        if not codec:
            raise ValueError("codec cannot be empty")
        if not np.isfinite(fallback_fps) or fallback_fps <= 0.0:
            raise ValueError("fallback_fps must be positive")
        if queue_size < 1:
            raise ValueError("queue_size must be positive")

        self.directory = Path(directory)
        self.segment_seconds = float(segment_minutes) * 60.0
        self.retention_segments = int(retention_segments)
        self.codec = codec
        self.fallback_fps = float(fallback_fps)
        self.queue: Queue[np.ndarray] = Queue(maxsize=int(queue_size))
        self._monotonic = monotonic_clock
        self._wall_clock = wall_clock
        self._writer_factory = writer_factory or _default_writer
        self._stop = Event()
        self._thread: Thread | None = None
        self._writer = None
        self._current_path: Path | None = None
        self._current_size: tuple[int, int] | None = None
        self._segment_started_at: float | None = None
        self._segment_frame_count = 0
        self._started = False
        self._closing = False
        self._disabled = False
        self._allocated_paths: set[Path] = set()
        self._stats_lock = Lock()
        self._dropped_frames = 0
        self._written_frames = 0

    @property
    def dropped_frames(self) -> int:
        with self._stats_lock:
            return self._dropped_frames

    @property
    def written_frames(self) -> int:
        with self._stats_lock:
            return self._written_frames

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def disabled(self) -> bool:
        return self._disabled

    def start(self, nominal_fps: float | None = None) -> bool:
        if self._started:
            return not self._disabled
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._cleanup_to_limit(self.retention_segments)
        except Exception:
            LOGGER.exception("REPLAY_START_FAILED directory=%s", self.directory)
            self._disabled = True
            return False

        self._fps = self._select_fps(nominal_fps)
        self._started = True
        self._thread = Thread(
            target=self._run,
            name="rolling-replay-recorder",
            daemon=True,
        )
        self._thread.start()
        LOGGER.info(
            "REPLAY_STARTED directory=%s segment_seconds=%g retention=%d",
            self.directory,
            self.segment_seconds,
            self.retention_segments,
        )
        return True

    def submit(self, frame: np.ndarray) -> bool:
        """Queue a raw BGR frame without waiting for the writer thread."""

        if not self._started or self._closing or self._disabled:
            return False
        try:
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3:
                LOGGER.warning("REPLAY_FRAME_REJECTED shape=%s", array.shape)
                return False
            pending = np.array(array, copy=True)
        except Exception:
            LOGGER.exception("REPLAY_FRAME_COPY_FAILED")
            return False

        try:
            self.queue.put_nowait(pending)
            return True
        except Full:
            # Keep the newest source frame and discard one older pending frame.
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except Empty:
                pass
            with self._stats_lock:
                self._dropped_frames += 1
                dropped = self._dropped_frames
            try:
                self.queue.put_nowait(pending)
            except Full:
                with self._stats_lock:
                    self._dropped_frames += 1
                    dropped = self._dropped_frames
                LOGGER.warning("REPLAY_FRAMES_DROPPED total=%d", dropped)
                return False
            if dropped == 1 or dropped % 100 == 0:
                LOGGER.warning("REPLAY_FRAMES_DROPPED total=%d", dropped)
            return True

    def close(self) -> None:
        if not self._started:
            return
        if self._closing:
            return
        self._closing = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._thread = None
        self._close_writer()
        LOGGER.info("REPLAY_CLOSED dropped_frames=%d", self.dropped_frames)

    def _run(self) -> None:
        while not self._stop.is_set() or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                self._write_frame(frame)
            except Exception:
                # Replay is optional; a codec or disk failure must not escape
                # into RuntimeWorker and stop tracking.
                LOGGER.exception("REPLAY_WRITE_FAILED path=%s", self._current_path)
                self._disabled = True
                self._close_writer()
            finally:
                self.queue.task_done()
        self._close_writer()

    def _write_frame(self, frame: np.ndarray) -> None:
        if self._disabled:
            return
        height, width = frame.shape[:2]
        size = (int(width), int(height))
        now = self._monotonic()
        needs_new_segment = (
            self._writer is None
            or self._current_size != size
            or self._segment_started_at is None
            or now - self._segment_started_at >= self.segment_seconds
        )
        if needs_new_segment:
            self._close_writer()
            self._start_segment(size)
        if self._writer is None:
            return
        self._writer.write(frame)
        self._segment_frame_count += 1
        with self._stats_lock:
            self._written_frames += 1

    def _start_segment(self, size: tuple[int, int]) -> None:
        self._cleanup_before_new_segment()
        path = self._unique_path(self._wall_clock())
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        try:
            writer = self._writer_factory(path, fourcc, self._fps, size)
            is_opened = writer.isOpened() if hasattr(writer, "isOpened") else True
            if not is_opened:
                if hasattr(writer, "release"):
                    writer.release()
                LOGGER.error("REPLAY_WRITER_OPEN_FAILED path=%s", path)
                self._disabled = True
                return
        except Exception:
            LOGGER.exception("REPLAY_WRITER_OPEN_FAILED path=%s", path)
            self._disabled = True
            return

        self._writer = writer
        self._current_path = path
        self._current_size = size
        self._segment_started_at = self._monotonic()
        self._segment_frame_count = 0
        LOGGER.info("REPLAY_SEGMENT_STARTED path=%s", path)

    def _close_writer(self) -> None:
        if self._writer is None:
            return
        path = self._current_path
        frames = self._segment_frame_count
        try:
            self._writer.release()
        except Exception:
            LOGGER.exception("REPLAY_WRITER_RELEASE_FAILED path=%s", path)
        if path is not None:
            LOGGER.info("REPLAY_SEGMENT_CLOSED path=%s frames=%d", path, frames)
        self._writer = None
        self._current_path = None
        self._current_size = None
        self._segment_started_at = None
        self._segment_frame_count = 0

    def _select_fps(self, nominal_fps: float | None) -> float:
        if nominal_fps is not None:
            try:
                value = float(nominal_fps)
                if np.isfinite(value) and 0.0 < value <= 240.0:
                    return value
            except (TypeError, ValueError):
                pass
        return self.fallback_fps

    def _unique_path(self, start_time: datetime) -> Path:
        stem = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        candidate = self.directory / f"{stem}.mp4"
        suffix = 0
        while candidate.exists() or candidate in self._allocated_paths:
            suffix += 1
            candidate = self.directory / f"{stem}_{suffix}.mp4"
        self._allocated_paths.add(candidate)
        return candidate

    def _managed_files(self) -> list[tuple[datetime, int, Path]]:
        files: list[tuple[datetime, int, Path]] = []
        try:
            candidates = self.directory.glob("*.mp4")
        except OSError:
            return files
        for path in candidates:
            match = _MANAGED_FILE_RE.fullmatch(path.name)
            if match is None:
                continue
            try:
                timestamp = datetime.strptime(
                    match.group("timestamp"), "%Y-%m-%d_%H-%M-%S"
                )
            except ValueError:
                continue
            suffix = int(match.group("suffix") or 0)
            files.append((timestamp, suffix, path))
        files.sort(key=lambda item: (item[0], item[1], item[2].name))
        return files

    def _cleanup_to_limit(self, limit: int) -> None:
        files = self._managed_files()
        while len(files) > limit:
            self._delete_managed_file(files.pop(0)[2])

    def _cleanup_before_new_segment(self) -> None:
        files = self._managed_files()
        while len(files) >= self.retention_segments:
            self._delete_managed_file(files.pop(0)[2])

    def _delete_managed_file(self, path: Path) -> None:
        if _MANAGED_FILE_RE.fullmatch(path.name) is None:
            return
        try:
            path.unlink()
            self._allocated_paths.discard(path)
            LOGGER.info("REPLAY_SEGMENT_DELETED path=%s", path)
        except FileNotFoundError:
            self._allocated_paths.discard(path)
        except Exception:
            LOGGER.exception("REPLAY_SEGMENT_DELETE_FAILED path=%s", path)
