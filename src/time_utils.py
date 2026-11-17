"""Small time helpers shared by persistence-facing domain objects."""

from __future__ import annotations

from datetime import datetime


def current_local_timestamp() -> str:
    """Return a sortable local wall-clock timestamp for new Gallery records."""

    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
