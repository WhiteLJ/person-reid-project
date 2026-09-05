"""Logging setup shared by the MVP-1 entry point."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure concise application logging once at process startup."""

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"unknown logging level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="[%(levelname)s] %(message)s",
        force=True,
    )
