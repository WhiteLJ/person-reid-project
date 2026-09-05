"""YOLOv8 person-only detector for MVP-1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from ultralytics import YOLO

from .config import ModelConfig, RuntimeConfig
from .models import Detection


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def parse_detections(result: Any, person_class_id: int = 0) -> list[Detection]:
    """Convert one Ultralytics result into person-only application detections."""

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = _to_numpy(getattr(boxes, "xyxy", np.empty((0, 4))))
    confidences = _to_numpy(getattr(boxes, "conf", np.empty((0,))))
    class_ids = _to_numpy(getattr(boxes, "cls", np.empty((0,))))
    if xyxy.size == 0:
        return []

    xyxy = np.atleast_2d(xyxy)
    confidences = np.asarray(confidences).reshape(-1)
    class_ids = np.asarray(class_ids).reshape(-1)
    count = min(len(xyxy), len(confidences), len(class_ids))

    detections: list[Detection] = []
    for bbox, confidence, class_id in zip(
        xyxy[:count], confidences[:count], class_ids[:count]
    ):
        if int(class_id) != person_class_id:
            continue
        coordinates = tuple(float(value) for value in bbox[:4])
        if len(coordinates) != 4 or not np.isfinite(coordinates).all():
            continue
        detections.append(
            Detection(
                bbox=coordinates,
                confidence=float(confidence),
                class_id=int(class_id),
            )
        )
    return detections


class PersonDetector:
    """Load YOLOv8 once and run person-only detection on individual frames."""

    def __init__(self, model_config: ModelConfig, runtime_config: RuntimeConfig) -> None:
        if not model_config.yolo_weight.is_file():
            raise FileNotFoundError(
                f"YOLO weight file not found: {model_config.yolo_weight}"
            )

        self.config = model_config
        self.runtime_config = runtime_config
        self.model = YOLO(str(model_config.yolo_weight))
        self.names = self.model.names
        self.model.overrides["verbose"] = False

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run one detection pass; tracking is intentionally not used in MVP-1."""

        results = self.model.predict(
            source=frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            classes=[self.config.person_class_id],
            device=self.config.device,
            workers=self.runtime_config.num_workers,
            verbose=False,
        )
        if not results:
            return []
        return parse_detections(results[0], self.config.person_class_id)

    def class_name(self, class_id: int) -> str:
        if isinstance(self.names, Mapping):
            return str(self.names.get(class_id, class_id))
        if isinstance(self.names, Sequence) and not isinstance(self.names, str):
            if 0 <= class_id < len(self.names):
                return str(self.names[class_id])
        return str(class_id)
