"""Frame input contract used by the application runtime.

The current implementation is a local ``VideoSource``.  Camera/RTSP
implementations can be added later without changing the tracking or UI layer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameSource(Protocol):
    """Minimal BGR frame source contract."""

    @property
    def source_label(self) -> str:
        ...

    def open(self) -> None:
        ...

    def read(self) -> np.ndarray | None:
        ...

    def release(self) -> None:
        ...
