"""PC3 Vehicle manual selection and in-session target binding."""

from __future__ import annotations

from collections.abc import Sequence
from logging import getLogger
from math import ceil, floor

import numpy as np

from .models import SessionTarget, Track
from .roi_selector import find_track_by_roi
from .target_manager import TargetManager


LOGGER = getLogger(__name__)


def crop_vehicle(
    frame: np.ndarray,
    bbox: Sequence[float],
) -> np.ndarray | None:
    """Clamp a vehicle Track bbox and return a copied BGR crop."""

    if not isinstance(frame, np.ndarray) or frame.ndim < 2:
        raise ValueError("frame must be an image NumPy array")
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four values")

    coordinates = tuple(float(value) for value in bbox)
    if not np.isfinite(coordinates).all():
        return None

    height, width = frame.shape[:2]
    raw_x1, raw_y1, raw_x2, raw_y2 = coordinates
    x1 = max(0, min(width, floor(raw_x1)))
    y1 = max(0, min(height, floor(raw_y1)))
    x2 = max(0, min(width, ceil(raw_x2)))
    y2 = max(0, min(height, ceil(raw_y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop.copy()


class VehicleSelectionController:
    """Bind manually selected Vehicle Tracks to a separate TargetManager."""

    def __init__(
        self,
        target_manager: TargetManager,
        reid_extractor: object,
        *,
        min_iou: float,
    ) -> None:
        self.target_manager = target_manager
        self.reid_extractor = reid_extractor
        self.min_iou = min_iou

    def select_from_roi(
        self,
        frame: np.ndarray,
        vehicle_tracks: Sequence[Track],
        roi: Sequence[float],
        frame_index: int,
    ) -> SessionTarget | None:
        """Select one Vehicle Track and extract ReID only for a new target."""

        track = find_track_by_roi(roi, vehicle_tracks, self.min_iou)
        if track is None:
            LOGGER.info(
                "VEHICLE_ROI_FAILED roi=%s min_iou=%.3f",
                roi,
                self.min_iou,
            )
            return None

        existing = self.target_manager.target_for_track(track.track_id)
        if existing is not None:
            LOGGER.info(
                "VEHICLE_TARGET_SELECTION_IGNORED track=%d target=%d reason=already_selected",
                track.track_id,
                existing.target_id,
            )
            return existing

        crop = crop_vehicle(frame, track.bbox)
        if crop is None:
            LOGGER.info(
                "VEHICLE_TARGET_SELECTION_REJECTED track=%d reason=invalid_crop",
                track.track_id,
            )
            return None

        extract = getattr(self.reid_extractor, "extract", None)
        if not callable(extract):
            raise TypeError("reid_extractor must provide extract(crop)")
        embedding = extract(crop)
        target = self.target_manager.select(track, embedding, frame_index)
        LOGGER.info(
            "VEHICLE_TARGET_SELECTED target=%d track=%d frame=%d embedding_dim=%d",
            target.target_id,
            track.track_id,
            frame_index,
            target.centroid.shape[0],
        )
        return target

    def remove_from_roi(
        self,
        vehicle_tracks: Sequence[Track],
        roi: Sequence[float],
    ) -> bool:
        """Remove only the selected Vehicle Track matched by the ROI."""

        track = find_track_by_roi(roi, vehicle_tracks, self.min_iou)
        if track is None:
            LOGGER.info(
                "VEHICLE_REMOVE_ROI_FAILED roi=%s min_iou=%.3f",
                roi,
                self.min_iou,
            )
            return False

        target = self.target_manager.target_for_track(track.track_id)
        if target is None:
            LOGGER.info(
                "VEHICLE_REMOVE_IGNORED track=%d reason=not_selected",
                track.track_id,
            )
            return False

        removed = self.target_manager.remove_by_track_id(track.track_id)
        if removed:
            LOGGER.info(
                "VEHICLE_TARGET_REMOVED target=%d track=%d",
                target.target_id,
                track.track_id,
            )
        return removed

