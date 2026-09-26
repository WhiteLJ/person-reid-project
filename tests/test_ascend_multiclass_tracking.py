from __future__ import annotations

import unittest

import numpy as np

from src.ascend_multiclass_tracking import AscendMultiClassTrackingPipeline
from src.config import load_config
from src.models import Detection, Track


class _FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, _frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        return [
            Detection((10, 10, 40, 80), 0.9, 0),
            Detection((50, 10, 100, 80), 0.9, 2),
            Detection((110, 10, 160, 80), 0.9, 5),
            Detection((170, 10, 220, 80), 0.9, 7),
        ]


class _FakeTracker:
    def __init__(self, track_id: int) -> None:
        self.track_id = track_id
        self.calls: list[list[Detection]] = []

    def update(self, detections, _frame):
        self.calls.append(list(detections))
        return [
            Track(self.track_id + index, detection.bbox, detection.confidence, detection.class_id)
            for index, detection in enumerate(detections)
        ]


class AscendMulticlassTrackingTests(unittest.TestCase):
    def test_one_detector_call_and_independent_class_split(self) -> None:
        config = load_config("config/config_atlas.yaml")
        detector = _FakeDetector()
        person_tracker = _FakeTracker(1)
        vehicle_tracker = _FakeTracker(10)
        pipeline = AscendMultiClassTrackingPipeline(
            config,
            runtime=object(),  # type: ignore[arg-type]
            detector=detector,
            person_tracker=person_tracker,
            vehicle_tracker=vehicle_tracker,
        )

        result = pipeline.process(np.zeros((100, 240, 3), dtype=np.uint8))

        self.assertEqual(detector.calls, 1)
        self.assertEqual([d.class_id for d in person_tracker.calls[0]], [0])
        self.assertEqual([d.class_id for d in vehicle_tracker.calls[0]], [2, 5, 7])
        self.assertEqual([track.class_id for track in result.person_tracks], [0])
        self.assertEqual([track.class_id for track in result.vehicle_tracks], [2, 5, 7])
        self.assertEqual(pipeline.stats().yolo_inference_count, 1)


if __name__ == "__main__":
    unittest.main()

