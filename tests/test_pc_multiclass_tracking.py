from __future__ import annotations

import unittest

import numpy as np

from src.config import load_config
from src.models import Track
from src.pc_multiclass_tracking import (
    MultiClassTrackingPipeline,
    parse_multiclass_detections,
)


class _FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = np.asarray(
            (
                (10, 10, 50, 100),
                (60, 20, 180, 120),
                (2, 3, 9, 10),
            ),
            dtype=np.float32,
        )
        self.conf = np.asarray((0.91, 0.88, 0.99), dtype=np.float32)
        self.cls = np.asarray((0, 2, 1), dtype=np.float32)


class _FakeResult:
    boxes = _FakeBoxes()


class _FakeYOLO:
    names = {0: "person", 1: "bicycle", 2: "car"}

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.overrides: dict[str, object] = {}

    def predict(self, **kwargs: object) -> list[_FakeResult]:
        self.calls.append(kwargs)
        return [_FakeResult()]


class _FakeTracker:
    def __init__(self, track_id: int) -> None:
        self.track_id = track_id
        self.received: list[list[object]] = []

    def update(self, detections: list[object], frame: np.ndarray) -> list[Track]:
        del frame
        self.received.append(list(detections))
        if not detections:
            return []
        detection = detections[0]
        return [
            Track(
                track_id=self.track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_id=detection.class_id,
            )
        ]


class MultiClassTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config("config/config.yaml")

    def test_detection_parser_keeps_only_requested_classes(self) -> None:
        detections = parse_multiclass_detections(_FakeResult(), (0, 2))

        self.assertEqual([detection.class_id for detection in detections], [0, 2])

    def test_one_yolo_prediction_feeds_two_independent_trackers(self) -> None:
        model = _FakeYOLO()
        person_tracker = _FakeTracker(track_id=4)
        vehicle_tracker = _FakeTracker(track_id=4)
        pipeline = MultiClassTrackingPipeline(
            self.config,
            model=model,
            person_tracker=person_tracker,
            vehicle_tracker=vehicle_tracker,
        )
        frame = np.zeros((120, 200, 3), dtype=np.uint8)

        result = pipeline.process(frame)
        stats = pipeline.stats()

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["classes"], [0, 2])
        self.assertEqual(len(person_tracker.received), 1)
        self.assertEqual(len(vehicle_tracker.received), 1)
        self.assertEqual(
            [detection.class_id for detection in person_tracker.received[0]], [0]
        )
        self.assertEqual(
            [detection.class_id for detection in vehicle_tracker.received[0]], [2]
        )
        self.assertEqual([track.track_id for track in result.person_tracks], [4])
        self.assertEqual([track.track_id for track in result.vehicle_tracks], [4])
        self.assertEqual(stats.frames, 1)
        self.assertEqual(stats.yolo_inference_count, 1)
        self.assertEqual(stats.unique_person_tracks, 1)
        self.assertEqual(stats.unique_vehicle_tracks, 1)

    def test_repeated_frames_still_have_one_prediction_per_frame(self) -> None:
        model = _FakeYOLO()
        pipeline = MultiClassTrackingPipeline(
            self.config,
            model=model,
            person_tracker=_FakeTracker(track_id=1),
            vehicle_tracker=_FakeTracker(track_id=1),
        )
        frame = np.zeros((120, 200, 3), dtype=np.uint8)

        pipeline.process(frame)
        pipeline.process(frame)

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(pipeline.stats().yolo_inference_count, 2)


if __name__ == "__main__":
    unittest.main()
