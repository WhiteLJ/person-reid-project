"""Atlas Person + Vehicle detection and independent CPU BoT-SORT tracking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import numpy as np

from .ascend_detector import AscendMultiClassDetector
from .ascend_runtime import AscendRuntime
from .ascend_tracker import AscendBotSortTracker
from .config import AppConfig
from .models import Detection, Track
from .pc_multiclass_tracking import MultiClassTrackingResult, MultiClassTrackingStats


_DEFAULT_CLASS_NAMES = {
    0: "person",
    2: "car",
    5: "bus",
    7: "truck",
}


class AscendMultiClassTrackingPipeline:
    """Run one YOLO OM inference and two independent CPU BoT-SORT states.

    The detector and both trackers are owned by this object, while the ACL
    lifecycle is owned by the caller and shared with the Person and Vehicle
    ReID extractors.
    """

    def __init__(
        self,
        config: AppConfig,
        runtime: AscendRuntime,
        *,
        detector: Any | None = None,
        person_tracker: Any | None = None,
        vehicle_tracker: Any | None = None,
    ) -> None:
        if config.inference.backend != "ascend":
            raise ValueError(
                "AscendMultiClassTrackingPipeline requires inference.backend=ascend"
            )
        self.config = config
        self.runtime = runtime
        self.person_class_id = config.model.person_class_id
        self.vehicle_class_ids = tuple(
            sorted({int(value) for value in config.multiclass_tracking.vehicle_class_ids})
        )
        self.detected_class_ids = tuple(
            dict.fromkeys((self.person_class_id, *self.vehicle_class_ids))
        )
        self.detector = detector or AscendMultiClassDetector(
            config.ascend.yolo_model,
            runtime,
            image_size=config.model.image_size,
            confidence_threshold=config.model.conf_threshold,
            iou_threshold=config.model.iou_threshold,
            class_ids=self.detected_class_ids,
        )
        self.person_tracker = person_tracker or AscendBotSortTracker(
            config.tracking.tracker,
            persist=config.tracking.persist,
            class_ids=(self.person_class_id,),
        )
        self.vehicle_tracker = vehicle_tracker or AscendBotSortTracker(
            config.tracking.tracker,
            persist=config.tracking.persist,
            class_ids=self.vehicle_class_ids,
        )
        self.names: Mapping[int, str] = {
            class_id: _DEFAULT_CLASS_NAMES.get(class_id, str(class_id))
            for class_id in self.detected_class_ids
        }
        self.frame_count = 0
        self.yolo_inference_count = 0
        self._person_track_ids: set[int] = set()
        self._vehicle_track_ids: set[int] = set()
        self.yolo_ms_total = 0.0
        self.person_tracker_ms_total = 0.0
        self.vehicle_tracker_ms_total = 0.0

    def process(self, frame: np.ndarray) -> MultiClassTrackingResult:
        yolo_started = perf_counter()
        detections = self.detector.detect(frame)
        self.yolo_ms_total += (perf_counter() - yolo_started) * 1000.0
        self.frame_count += 1
        self.yolo_inference_count += 1

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

        person_started = perf_counter()
        person_tracks = self.person_tracker.update(person_detections, frame)
        self.person_tracker_ms_total += (perf_counter() - person_started) * 1000.0

        vehicle_started = perf_counter()
        vehicle_tracks = self.vehicle_tracker.update(vehicle_detections, frame)
        self.vehicle_tracker_ms_total += (perf_counter() - vehicle_started) * 1000.0

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
        return str(self.names.get(class_id, class_id))

