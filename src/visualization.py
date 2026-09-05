"""Detection and temporary Track ID drawing for the OpenCV view."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from .models import Detection, Track


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    class_name: Callable[[int], str],
    show_class_name: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """Return a copy of ``frame`` annotated with person detections."""

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_parts: list[str] = []
        if show_class_name:
            label_parts.append(class_name(detection.class_id))
        if show_confidence:
            label_parts.append(f"{detection.confidence:.2f}")
        if label_parts:
            label = " ".join(label_parts)
            text_y = y1 - 8 if y1 > 24 else min(height - 4, y2 + 20)
            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    return annotated


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    show_track_id: bool = True,
    show_confidence: bool = True,
    class_name: Callable[[int], str] | None = None,
    show_class_name: bool = True,
) -> np.ndarray:
    """Return a copy of ``frame`` annotated with temporary Track IDs."""

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for track in tracks:
        x1, y1, x2, y2 = (int(round(value)) for value in track.bbox)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_parts: list[str] = []
        if show_class_name and class_name is not None:
            label_parts.append(class_name(track.class_id))
        if show_track_id:
            label_parts.append(f"ID {track.track_id}")
        if show_confidence:
            label_parts.append(f"conf={track.confidence:.2f}")
        if label_parts:
            label = " ".join(label_parts)
            text_y = y1 - 8 if y1 > 24 else min(height - 4, y2 + 20)
            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    return annotated
