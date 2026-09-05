"""OpenCV video and camera input handling."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import parse_source


class VideoSource:
    """Read frames from a camera index or local video file."""

    def __init__(self, source: int | str | Path) -> None:
        self.source = parse_source(str(source) if isinstance(source, Path) else source)
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self._capture is not None:
            return
        self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            self.release()
            raise OSError(f"unable to open video source: {self.source}")

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            raise RuntimeError("video source is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
