from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from src.models import Track
from src.vehicle_plate import (
    PlateObservation,
    VehiclePlateCoordinator,
    _adapt_provider_result,
    normalize_plate_number,
)


def _config(**overrides):
    values = dict(
        enabled=True,
        min_confidence=0.85,
        min_plate_width=20,
        min_plate_height=10,
        min_sharpness=10.0,
        confirmation_hits=2,
        pending_max_age_frames=30,
        min_track_age_frames=3,
        max_new_plate_inferences_per_frame=1,
        inference_interval_frames=5,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _track(track_id: int = 1) -> Track:
    return Track(track_id, (10, 10, 190, 190), 0.95, 2)


class _Provider:
    def __init__(self, observations):
        self.observations = list(observations)
        self.calls = 0

    def recognize(self, crop):
        del crop
        self.calls += 1
        return self.observations


@unittest.skipUnless(
    __import__("src.vehicle_plate", fromlist=["cv2"]).cv2 is not None,
    "OpenCV is required for plate quality tests",
)
class VehiclePlateCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        # A high-frequency pattern makes the Laplacian sharpness gate deterministic.
        self.frame = np.zeros((220, 220, 3), dtype=np.uint8)
        for x in range(30, 140):
            self.frame[50:75, x, :] = 255 if x % 2 else 0

    def test_normalization_preserves_chinese_and_removes_spaces(self) -> None:
        self.assertEqual(normalize_plate_number(" 粤b 12345 "), "粤B12345")
        self.assertIsNone(normalize_plate_number("   "))

    def test_quality_observation_requires_track_age_and_two_confirmations(self) -> None:
        provider = _Provider(
            [PlateObservation("粤B12345", 0.95, (20, 40, 140, 70))]
        )
        coordinator = VehiclePlateCoordinator(_config(), provider)
        track = _track()

        self.assertEqual(coordinator.process_frame(self.frame, [track], 0), ())
        self.assertEqual(coordinator.process_frame(self.frame, [track], 1), ())
        self.assertEqual(coordinator.process_frame(self.frame, [track], 2), ())
        self.assertIsNone(coordinator.track_plate(1))
        for frame_index in range(3, 7):
            coordinator.process_frame(self.frame, [track], frame_index)
        self.assertEqual(coordinator.process_frame(self.frame, [track], 7)[0].plate_number, "粤B12345")
        self.assertEqual(coordinator.track_plate(1), "粤B12345")
        self.assertEqual(provider.calls, 2)

    def test_low_confidence_does_not_become_stable(self) -> None:
        provider = _Provider(
            [PlateObservation("粤B12345", 0.50, (20, 40, 140, 70))]
        )
        coordinator = VehiclePlateCoordinator(_config(), provider)
        track = _track()
        coordinator.process_frame(self.frame, [track], 0)
        coordinator.process_frame(self.frame, [track], 1)
        coordinator.process_frame(self.frame, [track], 2)
        coordinator.process_frame(self.frame, [track], 7)
        self.assertIsNone(coordinator.track_plate(1))

    def test_pending_majority_tolerates_one_ocr_character_flip(self) -> None:
        class SequenceProvider:
            def __init__(self):
                self.values = ["粤B12345", "粤B1234S", "粤B12345"]

            def recognize(self, crop):
                del crop
                return (PlateObservation(self.values.pop(0), 0.95, (20, 40, 140, 70)),)

        coordinator = VehiclePlateCoordinator(
            _config(inference_interval_frames=1, min_track_age_frames=1),
            SequenceProvider(),
        )
        track = _track()
        for frame_index in range(3):
            coordinator.process_frame(self.frame, [track], frame_index)
        self.assertEqual(coordinator.track_plate(1), "粤B12345")

    def test_target_plate_survives_track_rebinding(self) -> None:
        coordinator = VehiclePlateCoordinator(_config(), _Provider(()))
        coordinator.set_target_plate(3, "粤B12345")
        coordinator.bind_target_to_track(3, 17)
        self.assertEqual(coordinator.target_plate(3), "粤B12345")
        self.assertEqual(coordinator.track_plate(17), "粤B12345")


class PlateProviderAdapterTests(unittest.TestCase):
    def test_common_sequence_result_is_converted_to_local_xyxy_observation(self) -> None:
        observation = _adapt_provider_result(
            ("ABC12345", 0.96, (4, 8, 104, 36))
        )
        self.assertEqual(
            observation,
            PlateObservation("ABC12345", 0.96, (4, 8, 104, 36)),
        )

    def test_mapping_result_is_converted_without_leaking_provider_shape(self) -> None:
        observation = _adapt_provider_result(
            {"code": "ABC12345", "score": 0.96, "box": [4, 8, 104, 36]}
        )
        self.assertIsNotNone(observation)
        self.assertEqual(observation.plate_number, "ABC12345")
        self.assertEqual(observation.bbox, (4, 8, 104, 36))


if __name__ == "__main__":
    unittest.main()
