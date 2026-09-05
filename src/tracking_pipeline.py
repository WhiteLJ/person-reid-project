"""Ultralytics YOLOv8 + BoT-SORT tracking pipeline for MVP-2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from ultralytics import YOLO

from .config import ModelConfig, RuntimeConfig, TrackingConfig
from .models import Track


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def parse_tracks(result: Any, person_class_id: int = 0) -> list[Track]:
    """Convert one Ultralytics tracking result into person-only ``Track`` objects."""

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    track_ids_value = getattr(boxes, "id", None)
    if track_ids_value is None:
        return []

    xyxy = _to_numpy(getattr(boxes, "xyxy", np.empty((0, 4))))
    confidences = _to_numpy(getattr(boxes, "conf", np.empty((0,))))
    class_ids = _to_numpy(getattr(boxes, "cls", np.empty((0,))))
    track_ids = _to_numpy(track_ids_value)
    if xyxy.size == 0 or track_ids.size == 0:
        return []

    xyxy = np.atleast_2d(xyxy)
    confidences = np.asarray(confidences).reshape(-1)
    class_ids = np.asarray(class_ids).reshape(-1)
    track_ids = np.asarray(track_ids).reshape(-1)
    count = min(len(xyxy), len(confidences), len(class_ids), len(track_ids))

    tracks: list[Track] = []
    for bbox, confidence, class_id, track_id in zip(
        xyxy[:count], confidences[:count], class_ids[:count], track_ids[:count]
    ):
        if int(class_id) != person_class_id:
            continue

        coordinates = tuple(float(value) for value in bbox[:4])
        track_id_float = float(track_id)
        confidence_float = float(confidence)
        if (
            len(coordinates) != 4
            or not np.isfinite(coordinates).all()
            or not np.isfinite(track_id_float)
            or not np.isfinite(confidence_float)
        ):
            continue

        tracks.append(
            Track(
                track_id=int(track_id_float),
                bbox=coordinates,
                confidence=confidence_float,
                class_id=int(class_id),
            )
        )
    return tracks


class TrackingPipeline:
    """Load one YOLO model and reuse its persistent BoT-SORT state across frames."""

    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: RuntimeConfig,
        tracking_config: TrackingConfig,
        model: Any | None = None,
    ) -> None:
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.tracking_config = tracking_config

        if model is None:
            if not model_config.yolo_weight.is_file():
                raise FileNotFoundError(
                    f"YOLO weight file not found: {model_config.yolo_weight}"
                )
            model = YOLO(str(model_config.yolo_weight))

        self.model = model
        self.names = getattr(model, "names", {})
        if hasattr(self.model, "overrides"):
            self.model.overrides["verbose"] = False

    def process(self, frame: np.ndarray) -> list[Track]:
        """Run exactly one YOLO+BoT-SORT pass for a frame."""

        results = self.model.track(
            source=frame,
            persist=self.tracking_config.persist,
            tracker=self.tracking_config.tracker,
            classes=[self.model_config.person_class_id],
            conf=self.model_config.conf_threshold,
            iou=self.model_config.iou_threshold,
            imgsz=self.model_config.image_size,
            device=self.model_config.device,
            verbose=False,
        )
        if not results:
            return []
        return parse_tracks(results[0], self.model_config.person_class_id)

    def class_name(self, class_id: int) -> str:
        if isinstance(self.names, Mapping):
            return str(self.names.get(class_id, class_id))
        if isinstance(self.names, Sequence) and not isinstance(self.names, str):
            if 0 <= class_id < len(self.names):
                return str(self.names[class_id])
        return str(class_id)
