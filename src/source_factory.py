"""Factory for the currently supported local FrameSource implementation."""

from __future__ import annotations

from typing import Any

from .video_source import VideoSource


def create_frame_source(config_or_source: Any) -> VideoSource:
    """Create the local ``VideoSource`` from an AppConfig or source value.

    Network camera and Hikvision branches are intentionally reserved.  They
    are not opened or probed by MVP-10.
    """

    source = getattr(getattr(config_or_source, "video", None), "source", config_or_source)
    if isinstance(source, dict) and source.get("type") in {"rtsp", "network_camera", "hikvision"}:
        raise NotImplementedError("Network camera source is reserved for a later stage")
    return VideoSource(source)
