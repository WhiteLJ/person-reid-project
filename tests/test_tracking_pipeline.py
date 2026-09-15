from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.config import InferenceConfig, ModelConfig, RuntimeConfig, TrackingConfig
from src.models import Track
from src.tracking_pipeline import TrackingPipeline, parse_tracks


class _FakeBoxes:
    def __init__(self, ids) -> None:
        self.xyxy = np.array([[1, 2, 20, 40], [3, 4, 30, 50], [5, 6, 10, 12]])
        self.conf = np.array([0.91, 0.88, 0.52])
        self.cls = np.array([0, 1, 0])
        self.id = ids


class _FakeResult:
    def __init__(self, ids) -> None:
        self.boxes = _FakeBoxes(ids)


class _FakeModel:
    names = {0: "person", 1: "car"}

    def __init__(self) -> None:
        self.overrides = {}
        self.track_calls: list[dict] = []

    def track(self, **kwargs):
        self.track_calls.append(kwargs)
        return [_FakeResult(np.array([7, 99, 8]))]


class _FakeAscendDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return []


class _FakeAscendTracker:
    def __init__(self) -> None:
        self.calls = 0

    def update(self, detections, frame):
        self.calls += 1
        self.last = (detections, frame)
        return []


def _configs() -> tuple[ModelConfig, RuntimeConfig, TrackingConfig]:
    return (
        ModelConfig(Path("weights/yolo/yolov8n.pt"), "cpu", 0, 0.35, 0.50, 640),
        RuntimeConfig(0, "INFO"),
        TrackingConfig("botsort.yaml", True, True),
    )


class TrackingPipelineTests(unittest.TestCase):
    def test_parse_tracks_keeps_persons_and_track_ids(self) -> None:
        tracks = parse_tracks(_FakeResult(np.array([7, 99, 8])))

        self.assertEqual(
            tracks,
            [
                Track(7, (1.0, 2.0, 20.0, 40.0), 0.91, 0),
                Track(8, (5.0, 6.0, 10.0, 12.0), 0.52, 0),
            ],
        )

    def test_parse_tracks_handles_missing_ids_or_boxes(self) -> None:
        self.assertEqual(parse_tracks(_FakeResult(None)), [])
        self.assertEqual(parse_tracks(type("NoBoxes", (), {"boxes": None})()), [])

    def test_track_call_uses_persistent_botsort_without_workers(self) -> None:
        model_config, runtime_config, tracking_config = _configs()
        fake_model = _FakeModel()
        pipeline = TrackingPipeline(
            model_config,
            runtime_config,
            tracking_config,
            model=fake_model,
        )
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        first_tracks = pipeline.process(frame)
        second_tracks = pipeline.process(frame)

        self.assertEqual(len(first_tracks), 2)
        self.assertEqual(len(second_tracks), 2)
        self.assertEqual(len(fake_model.track_calls), 2)
        call = fake_model.track_calls[0]
        self.assertTrue(call["persist"])
        self.assertEqual(call["tracker"], "botsort.yaml")
        self.assertEqual(call["classes"], [0])
        self.assertNotIn("workers", call)

    def test_ascend_backend_uses_detector_then_cpu_tracker_once_per_frame(self) -> None:
        model_config, runtime_config, tracking_config = _configs()
        detector = _FakeAscendDetector()
        tracker = _FakeAscendTracker()
        unused_torch_model = _FakeModel()
        pipeline = TrackingPipeline(
            model_config,
            runtime_config,
            tracking_config,
            model=unused_torch_model,
            inference_config=InferenceConfig("ascend"),
            ascend_runtime=object(),  # type: ignore[arg-type]
            ascend_detector=detector,
            ascend_tracker=tracker,
        )
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        self.assertEqual(pipeline.process(frame), [])
        self.assertEqual(pipeline.process(frame), [])

        self.assertEqual(detector.calls, 2)
        self.assertEqual(tracker.calls, 2)
        self.assertIsNone(pipeline.model)
        self.assertEqual(unused_torch_model.track_calls, [])


if __name__ == "__main__":
    unittest.main()
