from __future__ import annotations

import unittest

import numpy as np

from src.ascend_detector import (
    AscendPersonDetector,
    LetterboxTransform,
    decode_yolo_output,
    letterbox_bgr,
    nms_xyxy,
)
from src.models import Detection


class _FakeModel:
    output_shapes = ((1, 6, 1),)


class _FakeRuntime:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.inputs: list[np.ndarray] = []

    def execute(self, _model, inputs):
        self.inputs.extend(inputs)
        return (self.output.reshape(-1).copy(),)


class AscendDetectorTests(unittest.TestCase):
    def test_letterbox_preserves_aspect_ratio_and_rgb_nchw(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # BGR blue

        tensor, transform = letterbox_bgr(frame, 640)

        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isclose(tensor[0, 2, 200, 10], 1.0))  # RGB red channel
        self.assertTrue(np.isclose(tensor[0, 0, 200, 10], 0.0))
        self.assertAlmostEqual(transform.scale, 3.2)
        self.assertAlmostEqual(transform.pad_y, 160.0)

    def test_decode_supports_objectness_output_and_maps_bbox_back(self) -> None:
        transform = LetterboxTransform(3.2, 0.0, 160.0, 640, 640)
        output = np.zeros((1, 6, 2), dtype=np.float32)
        # xywh/objectness/person-score for an original-frame box (20,10,60,90)
        output[0, :, 0] = (128, 320, 128, 256, 0.9, 0.9)
        output[0, :, 1] = (400, 300, 40, 40, 0.1, 0.9)

        detections = decode_yolo_output(
            output,
            transform,
            (100, 200, 3),
            confidence_threshold=0.5,
            iou_threshold=0.5,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_id, 0)
        self.assertTrue(np.allclose(detections[0].bbox, (20, 10, 60, 90)))
        self.assertAlmostEqual(detections[0].confidence, 0.81, places=5)

    def test_decode_supports_channel_first_yolo_output_and_person_filter(self) -> None:
        transform = LetterboxTransform(1.0, 0.0, 0.0, 640, 640)
        output = np.zeros((1, 84, 2), dtype=np.float32)
        output[0, :4, 0] = (50, 50, 20, 20)
        output[0, 4, 0] = 0.8
        output[0, 5, 0] = 0.95
        output[0, :4, 1] = (100, 100, 20, 20)
        output[0, 4, 1] = 0.1
        output[0, 5, 1] = 0.9

        detections = decode_yolo_output(
            output,
            transform,
            (640, 640, 3),
            confidence_threshold=0.5,
            iou_threshold=0.5,
        )

        self.assertEqual(len(detections), 1)
        self.assertTrue(np.allclose(detections[0].bbox, (40, 40, 60, 60)))
        self.assertAlmostEqual(detections[0].confidence, 0.8, places=5)
        self.assertEqual(detections[0].class_id, 0)

    def test_nms_keeps_highest_overlapping_box(self) -> None:
        kept = nms_xyxy(
            np.asarray(((0, 0, 10, 10), (1, 1, 9, 9), (30, 30, 40, 40)), dtype=np.float32),
            np.asarray((0.8, 0.9, 0.7), dtype=np.float32),
            0.5,
        )
        self.assertEqual(kept.tolist(), [1, 2])

    def test_detector_calls_runtime_once_and_returns_detections(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        output = np.zeros((1, 6, 1), dtype=np.float32)
        output[0, :, 0] = (320, 320, 100, 200, 0.9, 0.9)
        runtime = _FakeRuntime(output)
        detector = AscendPersonDetector(
            "unused.om",
            runtime,  # type: ignore[arg-type]
            image_size=640,
            confidence_threshold=0.5,
            iou_threshold=0.5,
            model=_FakeModel(),
        )

        detections = detector.detect(frame)

        self.assertEqual(detector.inference_count, 1)
        self.assertEqual(len(runtime.inputs), 1)
        self.assertEqual(len(detections), 1)


if __name__ == "__main__":
    unittest.main()
