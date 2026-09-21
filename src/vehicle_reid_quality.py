"""Vehicle-specific ReID quality gates for PC4A recovery evidence."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from .config import VehicleReIDQualityConfig
from .models import Track
from .reid_quality import _area, _box, _clip_box, _intersection_area
from .vehicle_selection import crop_vehicle


@dataclass(frozen=True)
class VehicleReIDQualityResult:
    """A vehicle crop and the reason it is or is not identity evidence."""

    accepted: bool
    crop: np.ndarray | None
    reason: str | None = None
    edge_truncation_ratio: float = 0.0
    max_vehicle_overlap_ratio: float = 0.0


def assess_vehicle_reid_quality(
    frame: np.ndarray,
    track: Track,
    tracks: Sequence[Track],
    quality_config: VehicleReIDQualityConfig,
    *,
    vehicle_class_ids: Collection[int] = (2,),
) -> VehicleReIDQualityResult:
    """Accept only sufficiently complete, sufficiently isolated vehicle crops.

    Overlap is intersection divided by the target vehicle area, matching the
    Person quality-gate semantics while considering Vehicle boxes only.
    """

    allowed_classes = {int(class_id) for class_id in vehicle_class_ids}
    if track.class_id not in allowed_classes:
        return VehicleReIDQualityResult(False, None, "not_vehicle")
    if not np.isfinite(track.confidence) or (
        track.confidence < quality_config.min_track_confidence
    ):
        return VehicleReIDQualityResult(False, None, "low_confidence")

    height, width = frame.shape[:2]
    raw_box = _box(track.bbox)
    raw_area = _area(raw_box)
    if raw_area <= 0.0:
        return VehicleReIDQualityResult(False, None, "invalid_bbox")

    clipped_box = _clip_box(raw_box, width, height)
    clipped_area = _area(clipped_box)
    if clipped_area <= 0.0:
        return VehicleReIDQualityResult(False, None, "outside_frame")
    edge_truncation_ratio = max(0.0, 1.0 - clipped_area / raw_area)
    if edge_truncation_ratio > quality_config.max_edge_truncation_ratio:
        return VehicleReIDQualityResult(
            False,
            None,
            "edge_truncated",
            edge_truncation_ratio=edge_truncation_ratio,
        )

    margin_x = width * quality_config.min_frame_edge_margin_ratio
    margin_y = height * quality_config.min_frame_edge_margin_ratio
    x1, y1, x2, y2 = clipped_box
    if (
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= width - margin_x
        or y2 >= height - margin_y
    ):
        return VehicleReIDQualityResult(
            False,
            None,
            "frame_edge",
            edge_truncation_ratio=edge_truncation_ratio,
        )

    max_overlap_ratio = 0.0
    for other in tracks:
        if other.track_id == track.track_id or other.class_id not in allowed_classes:
            continue
        other_box = _clip_box(_box(other.bbox), width, height)
        overlap = _intersection_area(clipped_box, other_box)
        max_overlap_ratio = max(max_overlap_ratio, overlap / clipped_area)
    if max_overlap_ratio > quality_config.max_vehicle_overlap_ratio:
        return VehicleReIDQualityResult(
            False,
            None,
            "vehicle_overlap",
            edge_truncation_ratio=edge_truncation_ratio,
            max_vehicle_overlap_ratio=max_overlap_ratio,
        )

    crop = crop_vehicle(frame, track.bbox)
    if crop is None:
        return VehicleReIDQualityResult(
            False,
            None,
            "small_crop",
            edge_truncation_ratio=edge_truncation_ratio,
            max_vehicle_overlap_ratio=max_overlap_ratio,
        )
    crop_height, crop_width = crop.shape[:2]
    if (
        crop_width < quality_config.min_crop_width
        or crop_height < quality_config.min_crop_height
    ):
        return VehicleReIDQualityResult(
            False,
            None,
            "small_crop",
            edge_truncation_ratio=edge_truncation_ratio,
            max_vehicle_overlap_ratio=max_overlap_ratio,
        )
    return VehicleReIDQualityResult(
        True,
        crop,
        edge_truncation_ratio=edge_truncation_ratio,
        max_vehicle_overlap_ratio=max_overlap_ratio,
    )
