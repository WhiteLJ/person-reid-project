"""PC-only Person + Vehicle detection and independent BoT-SORT tracking.

This module is intentionally separate from the production person pipeline.  It
performs one Ultralytics detection pass per frame, splits the detections by
class, and feeds the two class groups into independent CPU BoT-SORT instances.
It does not create SessionTargets, run ReID, or touch Gallery state.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from .config import AppConfig
from .models import Detection, Track


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def parse_multiclass_detections(
    result: Any,
    class_ids: Collection[int],
) -> list[Detection]:
    """Convert one YOLO result into detections for the requested classes."""

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = _to_numpy(getattr(boxes, "xyxy", np.empty((0, 4))))
    confidences = _to_numpy(getattr(boxes, "conf", np.empty((0,))))
    detected_classes = _to_numpy(getattr(boxes, "cls", np.empty((0,))))
    if xyxy.size == 0:
        return []

    xyxy = np.atleast_2d(xyxy)
    confidences = np.asarray(confidences).reshape(-1)
    detected_classes = np.asarray(detected_classes).reshape(-1)
    allowed = {int(class_id) for class_id in class_ids}
    count = min(len(xyxy), len(confidences), len(detected_classes))

    detections: list[Detection] = []
    for bbox, confidence, class_id in zip(
        xyxy[:count], confidences[:count], detected_classes[:count]
    ):
        class_id_float = float(class_id)
        confidence_float = float(confidence)
        coordinates = tuple(float(value) for value in bbox[:4])
        if (
            len(coordinates) != 4
            or not np.isfinite(np.asarray(coordinates)).all()
            or not np.isfinite(class_id_float)
            or not np.isfinite(confidence_float)
            or int(class_id_float) not in allowed
        ):
            continue
        detections.append(
            Detection(
                bbox=coordinates,
                confidence=confidence_float,
                class_id=int(class_id_float),
            )
        )
    return detections


class _DetectionResults:
    """Minimal results adapter required by Ultralytics BOTSORT.update()."""

    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)
        if not (len(self.xyxy) == len(self.conf) == len(self.cls)):
            raise ValueError("detection adapter arrays must have matching lengths")

    @property
    def xywh(self) -> np.ndarray:
        result = np.empty_like(self.xyxy)
        result[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        result[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
        result[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        result[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return result

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, index: Any) -> "_DetectionResults":
        return _DetectionResults(self.xyxy[index], self.conf[index], self.cls[index])


def _resolve_tracker_file(value: str | Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    try:
        from ultralytics.utils.checks import check_yaml

        return Path(check_yaml(str(value)))
    except Exception as exc:
        raise FileNotFoundError(f"BoT-SORT profile not found: {value}") from exc


def _load_botsort_args(path: str | Path) -> dict[str, Any]:
    profile_path = _resolve_tracker_file(path)
    with profile_path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError(f"BoT-SORT profile must be a mapping: {profile_path}")
    if values.get("tracker_type", "botsort") != "botsort":
        raise ValueError("PC2 requires a BoT-SORT tracker profile")
    values.setdefault("track_high_thresh", 0.25)
    values.setdefault("track_low_thresh", 0.1)
    values.setdefault("new_track_thresh", 0.25)
    values.setdefault("track_buffer", 30)
    values.setdefault("match_thresh", 0.8)
    values.setdefault("fuse_score", True)
    values.setdefault("gmc_method", "sparseOptFlow")
    values.setdefault("proximity_thresh", 0.5)
    values.setdefault("appearance_thresh", 0.8)
    values.setdefault("with_reid", False)
    values.setdefault("model", "auto")
    if bool(values["with_reid"]):
        raise ValueError("PC2 requires with_reid: false")
    values["device"] = "cpu"
    return values


class BoTSORTDetectionTracker:
    """One independent CPU BoT-SORT state for one or more class IDs."""

    def __init__(
        self,
        tracker_config: str | Path,
        class_ids: Collection[int],
        *,
        persist: bool = True,
    ) -> None:
        try:
            from ultralytics.trackers.bot_sort import BOTSORT
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics 8.4.138 is required for the PC2 BoT-SORT adapter"
            ) from exc

        self.class_ids = frozenset(int(class_id) for class_id in class_ids)
        if not self.class_ids:
            raise ValueError("BoTSORTDetectionTracker requires at least one class ID")
        values = _load_botsort_args(tracker_config)
        self.tracker = BOTSORT(SimpleNamespace(**values))
        self.persist = persist

    def update(
        self,
        detections: Sequence[Detection],
        frame: np.ndarray,
    ) -> list[Track]:
        if not self.persist:
            self.tracker.reset()
        selected = [
            detection
            for detection in detections
            if detection.class_id in self.class_ids
        ]
        results = _DetectionResults(
            np.asarray([detection.bbox for detection in selected], dtype=np.float32),
            np.asarray(
                [detection.confidence for detection in selected], dtype=np.float32
            ),
            np.asarray([detection.class_id for detection in selected], dtype=np.float32),
        )
        tracked = np.asarray(self.tracker.update(results, img=frame), dtype=np.float32)
        if tracked.size == 0:
            return []

        tracks: list[Track] = []
        for row in np.atleast_2d(tracked):
            if len(row) < 7:
                continue
            bbox = tuple(float(value) for value in row[:4])
            track_id = float(row[4])
            confidence = float(row[5])
            class_id = float(row[6])
            values = (*bbox, track_id, confidence, class_id)
            if not np.isfinite(np.asarray(values)).all():
                continue
            if int(class_id) not in self.class_ids:
                continue
            tracks.append(
                Track(
                    track_id=int(track_id),
                    bbox=bbox,
                    confidence=confidence,
                    class_id=int(class_id),
                )
            )
        return tracks

    def reset(self) -> None:
        self.tracker.reset()


@dataclass(frozen=True)
class MultiClassTrackingResult:
    """Separate temporary Track collections for Person and Vehicle objects."""

    person_tracks: list[Track]
    vehicle_tracks: list[Track]


@dataclass(frozen=True)
class MultiClassTrackingStats:
    frames: int
    yolo_inference_count: int
    unique_person_tracks: int
    unique_vehicle_tracks: int
    yolo_ms_total: float = 0.0
    person_tracker_ms_total: float = 0.0
    vehicle_tracker_ms_total: float = 0.0


class MultiClassTrackingPipeline:
    """Run one YOLO prediction and two independent BoT-SORT updates per frame."""

    def __init__(
        self,
        config: AppConfig,
        *,
        model: Any | None = None,
        person_tracker: Any | None = None,
        vehicle_tracker: Any | None = None,
    ) -> None:
        if config.inference.backend != "torch":
            raise ValueError("MVP-8.3-PC2 is PC-only and requires inference.backend=torch")
        self.config = config
        self.person_class_id = config.model.person_class_id
        self.vehicle_class_ids = tuple(config.multiclass_tracking.vehicle_class_ids)
        self.detected_class_ids = tuple(
            dict.fromkeys((self.person_class_id, *self.vehicle_class_ids))
        )

        if model is None:
            if not config.model.yolo_weight.is_file():
                raise FileNotFoundError(
                    f"YOLO weight file not found: {config.model.yolo_weight}"
                )
            from ultralytics import YOLO

            model = YOLO(str(config.model.yolo_weight))
        self.model = model
        if hasattr(self.model, "overrides"):
            self.model.overrides["verbose"] = False
        self.names = getattr(self.model, "names", {})

        self.person_tracker = person_tracker or BoTSORTDetectionTracker(
            config.tracking.tracker,
            (self.person_class_id,),
            persist=config.tracking.persist,
        )
        self.vehicle_tracker = vehicle_tracker or BoTSORTDetectionTracker(
            config.tracking.tracker,
            self.vehicle_class_ids,
            persist=config.tracking.persist,
        )
        self.frame_count = 0
        self.yolo_inference_count = 0
        self._person_track_ids: set[int] = set()
        self._vehicle_track_ids: set[int] = set()
        self.last_detections: list[Detection] = []
        self.yolo_ms_total = 0.0
        self.person_tracker_ms_total = 0.0
        self.vehicle_tracker_ms_total = 0.0

    def process(self, frame: np.ndarray) -> MultiClassTrackingResult:
        """Detect both classes once, then update each independent tracker once."""

        yolo_started = perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.config.model.conf_threshold,
            iou=self.config.model.iou_threshold,
            imgsz=self.config.model.image_size,
            classes=list(self.detected_class_ids),
            device=self.config.model.device,
            workers=self.config.runtime.num_workers,
            verbose=False,
        )
        self.yolo_ms_total += (perf_counter() - yolo_started) * 1000.0
        self.frame_count += 1
        self.yolo_inference_count += 1
        result = results[0] if results else None
        detections = (
            parse_multiclass_detections(result, self.detected_class_ids)
            if result is not None
            else []
        )
        self.last_detections = list(detections)
        person_detections = [
            detection
            for detection in detections
            if detection.class_id == self.person_class_id
        ]
        vehicle_detections = [
            detection
            for detection in detections
            if detection.class_id in self.vehicle_class_ids
        ]
        person_tracker_started = perf_counter()
        person_tracks = self.person_tracker.update(person_detections, frame)
        self.person_tracker_ms_total += (
            perf_counter() - person_tracker_started
        ) * 1000.0
        vehicle_tracker_started = perf_counter()
        vehicle_tracks = self.vehicle_tracker.update(vehicle_detections, frame)
        self.vehicle_tracker_ms_total += (
            perf_counter() - vehicle_tracker_started
        ) * 1000.0
        self._person_track_ids.update(track.track_id for track in person_tracks)
        self._vehicle_track_ids.update(track.track_id for track in vehicle_tracks)
        return MultiClassTrackingResult(
            person_tracks=person_tracks,
            vehicle_tracks=vehicle_tracks,
        )

    def stats(self) -> MultiClassTrackingStats:
        return MultiClassTrackingStats(
            frames=self.frame_count,
            yolo_inference_count=self.yolo_inference_count,
            unique_person_tracks=len(self._person_track_ids),
            unique_vehicle_tracks=len(self._vehicle_track_ids),
            yolo_ms_total=self.yolo_ms_total,
            person_tracker_ms_total=self.person_tracker_ms_total,
            vehicle_tracker_ms_total=self.vehicle_tracker_ms_total,
        )

    def class_name(self, class_id: int) -> str:
        if isinstance(self.names, Mapping):
            return str(self.names.get(class_id, class_id))
        if isinstance(self.names, Sequence) and not isinstance(self.names, str):
            if 0 <= class_id < len(self.names):
                return str(self.names[class_id])
        return str(class_id)
