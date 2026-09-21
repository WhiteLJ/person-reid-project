from __future__ import annotations

import unittest

import numpy as np

from src.config import VehicleReIDQualityConfig, VehicleRecoveryConfig
from src.models import TargetState, Track
from src.reid_frame_cache import ReIDFrameCache
from src.target_manager import TargetManager
from src.vehicle_recovery import VehicleRecoveryCoordinator
from src.vehicle_reid_quality import assess_vehicle_reid_quality


class _FakeVehicleReID:
    def __init__(self, dimension: int = 2048) -> None:
        self.dimension = dimension
        self.extract_calls = 0
        self.batch_calls = 0
        self.batch_sizes: list[int] = []

    def extract(self, crop: np.ndarray) -> np.ndarray:
        del crop
        self.extract_calls += 1
        embedding = np.zeros(self.dimension, dtype=np.float32)
        embedding[0] = 1.0
        return embedding

    def extract_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        self.batch_calls += 1
        self.batch_sizes.append(len(crops))
        result = np.zeros((len(crops), self.dimension), dtype=np.float32)
        result[:, 0] = 1.0
        return result


def _track(track_id: int, bbox=(20, 20, 80, 120)) -> Track:
    return Track(track_id, bbox, 0.95, 2)


def _quality_config() -> VehicleReIDQualityConfig:
    return VehicleReIDQualityConfig(
        min_track_confidence=0.1,
        max_edge_truncation_ratio=0.3,
        max_vehicle_overlap_ratio=0.5,
        min_frame_edge_margin_ratio=0.0,
        min_crop_width=10,
        min_crop_height=10,
    )


def _recovery_config(**overrides) -> VehicleRecoveryConfig:
    values = dict(
        lost_grace_frames=1,
        reference_update_interval_frames=15,
        recovery_interval_frames=1,
        max_reference_embeddings=8,
        recovery_threshold=0.80,
        recovery_margin=0.05,
        reference_update_threshold=0.80,
        recovery_reference_support_threshold=0.75,
        recovery_reference_support_top_k=3,
        recovery_min_track_age_frames=1,
        recovery_confirmation_hits=1,
        recovery_pending_max_age_frames=60,
        recovery_candidates_per_frame=1,
    )
    values.update(overrides)
    return VehicleRecoveryConfig(**values)


class VehicleRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((240, 640, 3), dtype=np.uint8)
        self.initial_track = _track(4)

    def _coordinator(self, extractor=None, **overrides):
        manager = TargetManager()
        extractor = extractor or _FakeVehicleReID()
        coordinator = VehicleRecoveryCoordinator(
            manager,
            extractor,
            reid_config=type("VehicleConfig", (), {})(),
            recovery_config=_recovery_config(**overrides),
            quality_config=_quality_config(),
            vehicle_class_ids=(2,),
            embedding_cache=ReIDFrameCache(),
        )
        return manager, coordinator, extractor

    def test_active_vehicle_becomes_lost_and_recovers_with_new_track_id(self) -> None:
        manager, coordinator, _extractor = self._coordinator()
        target = coordinator.select_from_track(
            self.frame,
            self.initial_track,
            0,
            tracks=[self.initial_track],
        )
        self.assertIsNotNone(target)
        assert target is not None

        coordinator.process_frame(self.frame, [], 1)
        self.assertEqual(target.state, TargetState.LOST)
        self.assertEqual(target.last_track_id, 4)
        self.assertIsNone(target.current_track_id)

        recovered_track = _track(38, (120, 20, 180, 120))
        matches = coordinator.process_frame(self.frame, [recovered_track], 2)

        self.assertEqual(len(matches), 1)
        self.assertEqual(target.state, TargetState.ACTIVE)
        self.assertEqual(target.target_id, 1)
        self.assertEqual(target.current_track_id, 38)
        self.assertEqual(target.last_track_id, 38)

    def test_incremental_sweep_respects_vehicle_budget(self) -> None:
        manager, coordinator, extractor = self._coordinator(
            recovery_candidates_per_frame=1,
            recovery_margin=0.0,
        )
        target = coordinator.select_from_track(
            self.frame,
            self.initial_track,
            0,
            tracks=[self.initial_track],
        )
        assert target is not None
        coordinator.process_frame(self.frame, [], 1)
        candidates = [
            _track(10, (100, 20, 160, 120)),
            _track(11, (200, 20, 260, 120)),
            _track(12, (300, 20, 360, 120)),
        ]

        self.assertEqual(coordinator.process_frame(self.frame, candidates, 2), [])
        self.assertEqual(extractor.batch_sizes, [1])
        self.assertEqual(target.state, TargetState.LOST)
        self.assertEqual(coordinator.process_frame(self.frame, candidates, 3), [])
        self.assertEqual(len(coordinator.process_frame(self.frame, candidates, 4)), 1)
        self.assertEqual(extractor.batch_sizes, [1, 1, 1])
        self.assertEqual(target.state, TargetState.ACTIVE)
        self.assertEqual(target.current_track_id, 12)
        self.assertEqual(manager.target_recovered_count, 1)

    def test_recovery_does_not_immediately_add_candidate_to_reference_bank(self) -> None:
        _manager, coordinator, _extractor = self._coordinator()
        target = coordinator.select_from_track(
            self.frame,
            self.initial_track,
            0,
            tracks=[self.initial_track],
        )
        assert target is not None
        before = [reference.copy() for reference in target.reference_embeddings]
        coordinator.process_frame(self.frame, [], 1)

        coordinator.process_frame(
            self.frame,
            [_track(38, (120, 20, 180, 120))],
            2,
        )

        self.assertEqual(len(target.reference_embeddings), len(before))
        np.testing.assert_array_equal(target.reference_embeddings[0], before[0])

    def test_vehicle_quality_ignores_person_overlap_but_rejects_vehicle_overlap(self) -> None:
        target = _track(4, (40, 40, 140, 160))
        person = Track(9, (40, 40, 140, 160), 0.95, 0)
        accepted = assess_vehicle_reid_quality(
            self.frame,
            target,
            [target, person],
            _quality_config(),
        )
        self.assertTrue(accepted.accepted)

        other_vehicle = _track(10, (40, 40, 140, 160))
        rejected = assess_vehicle_reid_quality(
            self.frame,
            target,
            [target, other_vehicle],
            _quality_config(),
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "vehicle_overlap")

    def test_vehicle_cache_is_independent_from_person_cache_with_same_track_id(self) -> None:
        person_cache = ReIDFrameCache()
        vehicle_cache = ReIDFrameCache()
        person_cache.begin_frame(10)
        vehicle_cache.begin_frame(10)
        person_embedding = np.asarray((1.0, 0.0), dtype=np.float32)
        vehicle_embedding = np.asarray((0.0, 1.0), dtype=np.float32)
        person_cache.put(4, person_embedding, 10)
        vehicle_cache.put(4, vehicle_embedding, 10)

        np.testing.assert_array_equal(person_cache.get(4, 10), person_embedding)
        np.testing.assert_array_equal(vehicle_cache.get(4, 10), vehicle_embedding)
