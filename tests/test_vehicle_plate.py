from __future__ import annotations

from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace
import unittest

import numpy as np

from src.models import Track
from src.vehicle_plate import (
    PlateObservation,
    VehiclePlateCoordinator,
    VehiclePlateOCRWorker,
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
        min_inference_interval_ms=500,
        max_result_age_frames=15,
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


@unittest.skipUnless(
    __import__("src.vehicle_plate", fromlist=["cv2"]).cv2 is not None,
    "OpenCV is required for asynchronous plate quality tests",
)
class VehiclePlateAsyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((220, 220, 3), dtype=np.uint8)
        for x in range(30, 140):
            self.frame[50:75, x, :] = 255 if x % 2 else 0
        self.track = _track()

    def _config(self, **overrides):
        values = dict(
            enabled=True,
            min_confidence=0.85,
            min_plate_width=20,
            min_plate_height=10,
            min_sharpness=10.0,
            confirmation_hits=2,
            pending_max_age_frames=30,
            min_track_age_frames=1,
            max_new_plate_inferences_per_frame=1,
            inference_interval_frames=1,
            min_inference_interval_ms=0,
            max_result_age_frames=5,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_slow_provider_does_not_block_runtime_process_frame(self) -> None:
        started = Event()
        release = Event()

        class SlowProvider:
            def recognize(self, crop):
                del crop
                started.set()
                release.wait(timeout=1.0)
                return (PlateObservation("ABC12345", 0.95, (20, 40, 140, 70)),)

        worker = VehiclePlateOCRWorker(SlowProvider())
        worker.start()
        coordinator = VehiclePlateCoordinator(
            self._config(), SlowProvider(), ocr_worker=worker
        )
        try:
            started_at = monotonic()
            coordinator.process_frame(self.frame, [self.track], 0)
            elapsed = monotonic() - started_at
            self.assertLess(elapsed, 0.1)
            self.assertTrue(started.wait(timeout=1.0))
        finally:
            release.set()
            worker.close()

    def test_queue_keeps_one_waiting_task_and_prefers_newest(self) -> None:
        started = Event()
        release = Event()

        class BlockingProvider:
            def recognize(self, crop):
                del crop
                started.set()
                release.wait(timeout=1.0)
                return ()

        worker = VehiclePlateOCRWorker(BlockingProvider())
        worker.start()
        try:
            tasks = [
                worker.new_task(
                    track_id=index,
                    frame_index=index,
                    track_epoch=1,
                    class_id=2,
                    bbox=(10, 10, 190, 190),
                    target_id=None,
                    stable_plate_at_submit=None,
                    crop=self.frame.copy(),
                )
                for index in (1, 2, 3)
            ]
            self.assertIsNone(worker.submit(tasks[0]))
            self.assertTrue(started.wait(timeout=1.0))
            self.assertIsNone(worker.submit(tasks[1]))
            self.assertIs(worker.submit(tasks[2]), tasks[1])
            release.set()
            sleep(0.05)
            stats = worker.stats()
            self.assertEqual(stats.dropped_tasks, 1)
        finally:
            release.set()
            worker.close()

    def test_expired_result_is_discarded_before_plate_state_update(self) -> None:
        release = Event()

        class SlowProvider:
            def recognize(self, crop):
                del crop
                release.wait(timeout=1.0)
                return (PlateObservation("ABC12345", 0.95, (20, 40, 140, 70)),)

        worker = VehiclePlateOCRWorker(SlowProvider())
        worker.start()
        coordinator = VehiclePlateCoordinator(
            self._config(max_result_age_frames=1), SlowProvider(), ocr_worker=worker
        )
        try:
            coordinator.process_frame(self.frame, [self.track], 0)
            coordinator.process_frame(self.frame, [self.track], 1)
            coordinator.process_frame(self.frame, [self.track], 2)
            release.set()
            sleep(0.05)
            coordinator.process_frame(self.frame, [self.track], 3)
            self.assertIsNone(coordinator.track_plate(self.track.track_id))
            self.assertEqual(coordinator.ocr_stats()["expired_results"], 1)
        finally:
            release.set()
            worker.close()

    def test_async_results_still_require_two_confirmation_hits(self) -> None:
        class FastProvider:
            def recognize(self, crop):
                del crop
                return (PlateObservation("ABC12345", 0.95, (20, 40, 140, 70)),)

        worker = VehiclePlateOCRWorker(FastProvider())
        worker.start()
        coordinator = VehiclePlateCoordinator(
            self._config(), FastProvider(), ocr_worker=worker
        )
        confirmations = []
        try:
            for frame_index in range(40):
                confirmations.extend(
                    coordinator.process_frame(
                        self.frame, [self.track], frame_index
                    )
                )
                if confirmations:
                    break
                sleep(0.01)
            self.assertEqual(len(confirmations), 1)
            self.assertEqual(coordinator.track_plate(1), "ABC12345")
            self.assertGreaterEqual(coordinator.ocr_stats()["completed"], 2)
        finally:
            worker.close()

    def test_global_interval_limits_submissions_across_tracks(self) -> None:
        class FastProvider:
            def recognize(self, crop):
                del crop
                return ()

        worker = VehiclePlateOCRWorker(FastProvider())
        worker.start()
        coordinator = VehiclePlateCoordinator(
            self._config(min_inference_interval_ms=500),
            FastProvider(),
            ocr_worker=worker,
        )
        second_track = _track(2)
        try:
            for frame_index in range(10):
                coordinator.process_frame(
                    self.frame, [self.track, second_track], frame_index
                )
            self.assertEqual(worker.stats().submitted, 1)
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
