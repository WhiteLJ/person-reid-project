"""Detection and temporary Track ID drawing for the OpenCV view."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence

import cv2
import numpy as np

from .models import Detection, Track


VEHICLE_COLOR = (139, 0, 0)
VEHICLE_SELECTED_COLOR = (0, 0, 255)


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
    tracks: Sequence[Track],
    show_track_id: bool = True,
    show_confidence: bool = True,
    class_name: Callable[[int], str] | None = None,
    show_class_name: bool = True,
    selected_track_ids: Collection[int] | None = None,
    show_unselected_tracks: bool = False,
    gallery_labels_by_track: Mapping[int, str] | None = None,
) -> np.ndarray:
    """Draw selected Tracks and optionally draw unselected Tracks for debugging."""

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    selected_ids = selected_track_ids or set()
    for track in tracks:
        x1, y1, x2, y2 = (int(round(value)) for value in track.bbox)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        selected = track.track_id in selected_ids
        if not selected and not show_unselected_tracks:
            continue
        color = (0, 0, 255) if selected else (0, 255, 0)
        thickness = 4 if selected else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label_parts: list[str] = []
        if selected:
            gallery_label = (gallery_labels_by_track or {}).get(track.track_id)
            if gallery_label is None:
                label_parts.append(f"TARGET | ID {track.track_id}")
            else:
                label_parts.append(
                    f"TARGET {gallery_label} | ID {track.track_id}"
                )
        else:
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
                color,
                thickness,
                cv2.LINE_AA,
            )
    return annotated
