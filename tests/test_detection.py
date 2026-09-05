from __future__ import annotations

import unittest

import numpy as np

from src.detector import parse_detections


class _FakeBoxes:
    xyxy = np.array([[1, 2, 20, 40], [3, 4, 30, 50], [5, 6, 10, 12]])
    conf = np.array([0.91, 0.88, 0.52])
    cls = np.array([0, 1, 0])


class _FakeResult:
    boxes = _FakeBoxes()


class DetectionTests(unittest.TestCase):
    def test_parser_keeps_only_person_class(self) -> None:
        detections = parse_detections(_FakeResult(), person_class_id=0)
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].class_id, 0)
        self.assertAlmostEqual(detections[0].confidence, 0.91)
        self.assertEqual(detections[1].bbox, (5.0, 6.0, 10.0, 12.0))

    def test_parser_handles_missing_boxes(self) -> None:
        result = type("ResultWithoutBoxes", (), {"boxes": None})()
        self.assertEqual(parse_detections(result), [])


if __name__ == "__main__":
    unittest.main()
