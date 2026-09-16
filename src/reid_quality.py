"""Quality gates for ReID decisions in crowded scenes.

The tracker may continue to use every valid Track.  This module only decides
whether a crop is reliable enough to influence a ReID reference, recovery, or
Gallery recognition decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import ReIDConfig, ReIDQualityConfig
from .models import Track
from .reid import crop_person


@dataclass(frozen=True)
class ReIDQualityResult:
    """Result of evaluating one Track crop for identity decisions."""

    accepted: bool
    crop: np.ndarray | None
    reason: str | None = None
    edge_truncation_ratio: float = 0.0
    max_person_overlap_ratio: float = 0.0


def assess_reid_quality(
    frame: np.ndarray,
    track: Track,
    tracks: Sequence[Track],
    reid_config: ReIDConfig,
    quality_config: ReIDQualityConfig,
    *,
    person_class_id: int = 0,
) -> ReIDQualityResult:
    """Return a crop only when it is suitable for a ReID decision.

    Overlap is deliberately not IoU.  For the target Track, the metric is the
    fraction of its visible bbox covered by another person bbox:

    ``intersection(target, other) / area(target)``.

    This is useful for rejecting a target whose crop is substantially mixed
    with another person, even when the other person's bbox is much larger or
    smaller.
    """

    if track.class_id != person_class_id:
        return ReIDQualityResult(False, None, "not_person")
    if not np.isfinite(track.confidence) or (
        track.confidence < quality_config.min_track_confidence
    ):
        return ReIDQualityResult(False, None, "low_confidence")

    height, width = frame.shape[:2]
    raw_box = _box(track.bbox)
    raw_area = _area(raw_box)
    if raw_area <= 0.0:
        return ReIDQualityResult(False, None, "invalid_bbox")

    clipped_box = _clip_box(raw_box, width, height)
    clipped_area = _area(clipped_box)
    if clipped_area <= 0.0:
        return ReIDQualityResult(False, None, "outside_frame")
    edge_truncation_ratio = max(0.0, 1.0 - clipped_area / raw_area)
    if edge_truncation_ratio > quality_config.max_edge_truncation_ratio:
        return ReIDQualityResult(
            False,
            None,
            "edge_truncated",
            edge_truncation_ratio=edge_truncation_ratio,
        )

    # Trackers/detectors may already clip a box to the frame.  The legacy
    # truncation ratio cannot detect that case, so reject boxes inside a small
    # configurable safety margin around every image edge.
    margin_x = width * quality_config.min_frame_edge_margin_ratio
    margin_y = height * quality_config.min_frame_edge_margin_ratio
    x1, y1, x2, y2 = clipped_box
    if (
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= width - margin_x
        or y2 >= height - margin_y
    ):
        return ReIDQualityResult(
            False,
            None,
            "frame_edge",
            edge_truncation_ratio=edge_truncation_ratio,
        )

    max_overlap_ratio = 0.0
    for other in tracks:
        if other.track_id == track.track_id or other.class_id != person_class_id:
            continue
        other_box = _clip_box(_box(other.bbox), width, height)
        overlap = _intersection_area(clipped_box, other_box)
        max_overlap_ratio = max(max_overlap_ratio, overlap / clipped_area)
    if max_overlap_ratio > quality_config.max_person_overlap_ratio:
        return ReIDQualityResult(
            False,
            None,
            "person_overlap",
            edge_truncation_ratio=edge_truncation_ratio,
            max_person_overlap_ratio=max_overlap_ratio,
        )

    crop = crop_person(
        frame,
        track.bbox,
        min_crop_width=reid_config.min_crop_width,
        min_crop_height=reid_config.min_crop_height,
    )
    if crop is None:
        return ReIDQualityResult(
            False,
            None,
            "small_crop",
            edge_truncation_ratio=edge_truncation_ratio,
            max_person_overlap_ratio=max_overlap_ratio,
        )
    return ReIDQualityResult(
        True,
        crop,
        edge_truncation_ratio=edge_truncation_ratio,
        max_person_overlap_ratio=max_overlap_ratio,
    )


def _box(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    box = tuple(float(value) for value in values)
    if not np.isfinite(box).all():
        return (0.0, 0.0, 0.0, 0.0)
    return box


def _clip_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def _area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return _area((x1, y1, x2, y2))
