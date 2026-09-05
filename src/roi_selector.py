"""Pure ROI-to-Track matching helpers for MVP-3."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from .models import Track


def roi_xywh_to_xyxy(roi: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert an OpenCV ``(x, y, width, height)`` ROI to ``xyxy`` coordinates."""

    if len(roi) != 4:
        raise ValueError("ROI must contain exactly four values: x, y, width, height")

    x, y, width, height = (float(value) for value in roi)
    if not all(isfinite(value) for value in (x, y, width, height)):
        raise ValueError("ROI coordinates must be finite")

    x2 = x + width
    y2 = y + height
    return min(x, x2), min(y, y2), max(x, x2), max(y, y2)


def bbox_iou(
    box_a: Sequence[float], box_b: Sequence[float]
) -> float:
    """Return intersection-over-union for two ``xyxy`` bounding boxes."""

    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("bounding boxes must contain exactly four values")

    ax1, ay1, ax2, ay2 = (float(value) for value in box_a)
    bx1, by1, bx2, by2 = (float(value) for value in box_b)
    if not all(
        isfinite(value) for value in (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
    ):
        return 0.0

    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - intersection_area
    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area


def find_track_by_roi(
    roi: Sequence[float], tracks: Sequence[Track], min_iou: float
) -> Track | None:
    """Return the Track with the greatest IoU with ``roi`` if it passes the threshold."""

    if not 0.0 < min_iou <= 1.0:
        raise ValueError("min_iou must be greater than 0 and at most 1")

    roi_xyxy = roi_xywh_to_xyxy(roi)
    best_track: Track | None = None
    best_iou = 0.0
    for track in tracks:
        current_iou = bbox_iou(roi_xyxy, track.bbox)
        if best_track is None or current_iou > best_iou:
            best_track = track
            best_iou = current_iou

    if best_track is None or best_iou < min_iou:
        return None
    return best_track
