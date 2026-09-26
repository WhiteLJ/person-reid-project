"""Minimal frame-input contract shared by local video and future sources."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameSource(Protocol):
    """Source of OpenCV BGR frames for the business pipeline."""

    @property
    def source_label(self) -> str:
        """Short human-readable label for the UI."""

    def open(self) -> None:
        """Open the source."""

    def read(self) -> np.ndarray | None:
        """Read the next BGR frame, or return ``None`` at end of source."""

    def release(self) -> None:
        """Release source resources."""
