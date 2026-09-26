"""Factory for the current local frame source and future camera sources."""

from __future__ import annotations

from pathlib import Path

from .frame_source import FrameSource
from .video_source import VideoSource


def create_frame_source(source: int | str | Path) -> FrameSource:
    """Create the configured source without coupling the app to its class.

    Local camera indexes and video files continue to use ``VideoSource``.  URL
    sources are deliberately reserved until the physical camera stage.
    """

    if isinstance(source, str):
        normalized = source.strip().lower()
        if normalized.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            raise NotImplementedError(
                "Network camera source is reserved for a later stage"
            )
    return VideoSource(source)
