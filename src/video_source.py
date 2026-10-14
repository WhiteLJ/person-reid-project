"""OpenCV video and camera input handling."""

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from .config import parse_source
from .frame_source import SourceReadStatus


LOGGER = getLogger(__name__)


class VideoSource:
    """Read frames from a camera index or local video file."""

    def __init__(self, source: int | str | Path) -> None:
        self.source = parse_source(str(source) if isinstance(source, Path) else source)
        self._capture: cv2.VideoCapture | None = None
        self._read_status = SourceReadStatus.NOT_OPEN
        self._consecutive_read_failures = 0
        self._reconnect_pending = False
        self._last_failure_log = 0.0

    @property
    def source_label(self) -> str:
        if isinstance(self.source, int):
            return f"Camera {self.source}"
        return Path(self.source).name or str(self.source)

    @property
    def source_kind(self) -> str:
        return "camera" if isinstance(self.source, int) else "video"

    @property
    def is_camera(self) -> bool:
        return isinstance(self.source, int)

    @property
    def read_status(self) -> str:
        return self._read_status

    @property
    def consecutive_read_failures(self) -> int:
        return self._consecutive_read_failures

    def open(self) -> None:
        if self._capture is not None:
            return
        self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            self.release()
            LOGGER.error("SOURCE_OPEN_FAILED type=%s source=%s", self.source_kind, self.source_label)
            raise OSError(f"unable to open video source: {self.source}")
        self._read_status = SourceReadStatus.NOT_OPEN
        LOGGER.info("SOURCE_OPENED type=%s source=%s", self.source_kind, self.source_label)

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            raise RuntimeError("video source is not open")
        ok, frame = self._capture.read()
        if ok and frame is not None:
            if self._consecutive_read_failures or self._reconnect_pending:
                LOGGER.info(
                    "SOURCE_RECOVERED type=%s source=%s failures=%d",
                    self.source_kind,
                    self.source_label,
                    self._consecutive_read_failures,
                )
            self._consecutive_read_failures = 0
            self._reconnect_pending = False
            self._read_status = SourceReadStatus.FRAME
            return frame

        if self.is_camera:
            self._read_status = SourceReadStatus.TEMPORARY_FAILURE
            self._consecutive_read_failures += 1
            now = monotonic()
            if self._consecutive_read_failures == 1 or now - self._last_failure_log >= 1.0:
                LOGGER.warning(
                    "SOURCE_READ_FAILED type=%s source=%s consecutive_failures=%d",
                    self.source_kind,
                    self.source_label,
                    self._consecutive_read_failures,
                )
                self._last_failure_log = now
        else:
            self._read_status = SourceReadStatus.EOF
            LOGGER.info("SOURCE_EOF type=%s source=%s", self.source_kind, self.source_label)
        return None

    def reconnect(self) -> bool:
        """Reopen a camera after a temporary read failure."""

        if not self.is_camera:
            return False
        LOGGER.info("SOURCE_RECONNECT_ATTEMPT type=%s source=%s", self.source_kind, self.source_label)
        self.release()
        try:
            self._capture = cv2.VideoCapture(self.source)
            if not self._capture.isOpened():
                self.release()
                LOGGER.warning("SOURCE_RECONNECT_FAILED type=%s source=%s", self.source_kind, self.source_label)
                return False
        except Exception:
            self.release()
            LOGGER.exception("SOURCE_RECONNECT_FAILED type=%s source=%s", self.source_kind, self.source_label)
            return False
        self._read_status = SourceReadStatus.TEMPORARY_FAILURE
        self._reconnect_pending = True
        LOGGER.info("SOURCE_RECONNECTED type=%s source=%s", self.source_kind, self.source_label)
        return True

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
