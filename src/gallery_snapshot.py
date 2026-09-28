"""Snapshot capture for explicitly saved Gallery identities."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def crop_track_from_frame(
    frame: np.ndarray,
    bbox: Sequence[float],
) -> np.ndarray:
    """Return a clipped BGR crop from a source frame, never from a display copy."""

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be an HxWx3 BGR ndarray")
    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox does not produce a non-empty crop")
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("bbox crop is empty")
    return crop.copy()


def encode_track_snapshot(
    frame: np.ndarray,
    bbox: Sequence[float],
    *,
    max_side: int = 720,
    quality: int = 90,
) -> bytes:
    """Encode one clean track crop as a bounded JPEG snapshot."""

    crop = crop_track_from_frame(frame, bbox)
    if max_side > 0 and max(crop.shape[:2]) > max_side:
        scale = max_side / float(max(crop.shape[:2]))
        crop = cv2.resize(
            crop,
            (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(
        ".jpg",
        crop,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, int(quality)))],
    )
    if not ok or encoded.size == 0:
        raise ValueError("failed to encode Gallery snapshot")
    return bytes(encoded.tobytes())
