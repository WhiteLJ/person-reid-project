from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.config import ReIDConfig, ReIDRecoveryConfig
from src.models import SessionTarget, TargetState, Track
from src.reid_frame_cache import ReIDFrameCache
from src.target_manager import TargetManager
from src.target_recovery import (
    RecoveryCandidate,
    TargetRecoveryCoordinator,
    assign_recovery_matches,
    recovery_reference_support_score,
)


class _FakeReIDExtractor:
    def __init__(self, initial: np.ndarray, batch_embedding: np.ndarray) -> None:
        self.initial = initial.astype(np.float32)
        self.batch_embedding = batch_embedding.astype(np.float32)
        self.extract_calls = 0
        self.batch_calls = 0
        self.batch_sizes: list[int] = []

    def extract(self, crop: np.ndarray) -> np.ndarray:
        del crop
        self.extract_calls += 1
        return self.initial.copy()

    def extract_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        self.batch_calls += 1
        self.batch_sizes.append(len(crops))
        return np.repeat(self.batch_embedding[None, :], len(crops), axis=0)


class _SequenceReIDExtractor(_FakeReIDExtractor):
    def __init__(self, initial: np.ndarray, batch_embeddings: list[np.ndarray]) -> None:
        super().__init__(initial, batch_embeddings[0])
        self.batch_embeddings = [embedding.astype(np.float32) for embedding in batch_embeddings]

    def extract_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        self.batch_calls += 1
        self.batch_sizes.append(len(crops))
        if not self.batch_embeddings:
            raise AssertionError("fake extractor ran out of outputs")
        embedding = self.batch_embeddings.pop(0)
        return np.repeat(embedding[None, :], len(crops), axis=0)


def _reid_config() -> ReIDConfig:
    return ReIDConfig(
        model_name="osnet_x0_25",
        weight=Path("unused.pth"),
        image_height=256,
        image_width=128,
        min_crop_width=20,
        min_crop_height=50,
    )


def _recovery_config(**overrides: object) -> ReIDRecoveryConfig:
    values: dict[str, object] = {
        "lost_grace_frames": 1,
        "reference_update_interval_frames": 15,
        "recovery_interval_frames": 1,
        "max_reference_embeddings": 3,
        "recovery_threshold": 0.80,
        "recovery_margin": 0.05,
        "reference_update_threshold": 0.80,
        "recovery_reference_support_threshold": 0.80,
        "recovery_reference_support_top_k": 3,
        # These tests retain focused MVP-5 behavior; dedicated MVP-8.1 tests
        # below enable age and confirmation explicitly.
        "recovery_min_track_age_frames": 1,
        "recovery_confirmation_hits": 1,
        "recovery_pending_max_age_frames": 60,
    }
    values.update(overrides)
    return ReIDRecoveryConfig(**values)  # type: ignore[arg-type]


def _target(target_id: int, centroid: tuple[float, ...]) -> SessionTarget:
    vector = np.asarray(centroid, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return SessionTarget(
        target_id=target_id,
        current_track_id=None,
        last_track_id=target_id,
        state=TargetState.LOST,
        reference_embeddings=[vector.copy()],
        centroid=vector,
    )


def _target_with_references(
    target_id: int,
    references: list[np.ndarray],
) -> SessionTarget:
    normalized = [
        reference.astype(np.float32) / np.linalg.norm(reference)
        for reference in references
    ]
    centroid = np.mean(np.stack(normalized), axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    return SessionTarget(
        target_id=target_id,
        current_track_id=None,
        last_track_id=target_id,
        state=TargetState.LOST,
        reference_embeddings=[reference.copy() for reference in normalized],
        centroid=centroid.astype(np.float32),
    )


class AssignmentTests(unittest.TestCase):
    def test_single_target_single_candidate_has_no_second_best_requirement(self) -> None:
        target = _target(1, (1.0, 0.0))
        candidate = RecoveryCandidate(
            Track(11, (0, 0, 20, 60), 0.9, 0),
            np.asarray((1.0, 0.0), dtype=np.float32),
        )

        matches = assign_recovery_matches([target], [candidate], 0.75, 0.20)

        self.assertEqual([(match.target_id, match.candidate.track.track_id) for match in matches], [(1, 11)])

    def test_two_by_two_matching_is_one_to_one_and_margin_is_applied(self) -> None:
        targets = [_target(1, (1.0, 0.0)), _target(2, (0.0, 1.0))]
        candidates = [
            RecoveryCandidate(
                Track(11, (0, 0, 20, 60), 0.9, 0),
                np.asarray((0.95, 0.31), dtype=np.float32),
            ),
            RecoveryCandidate(
                Track(12, (30, 0, 50, 60), 0.9, 0),
                np.asarray((0.31, 0.95), dtype=np.float32),
            ),
        ]

        matches = assign_recovery_matches(targets, candidates, 0.75, 0.10)

        self.assertEqual(
            {(match.target_id, match.candidate.track.track_id) for match in matches},
            {(1, 11), (2, 12)},
        )

    def test_ambiguous_two_by_two_scores_are_not_forced(self) -> None:
        targets = [_target(1, (1.0, 0.0)), _target(2, (1.0, 0.0))]
        candidates = [
            RecoveryCandidate(
                Track(11, (0, 0, 20, 60), 0.9, 0),
                np.asarray((1.0, 0.0), dtype=np.float32),
            ),
            RecoveryCandidate(
                Track(12, (30, 0, 50, 60), 0.9, 0),
                np.asarray((1.0, 0.0), dtype=np.float32),
            ),
        ]

        matches = assign_recovery_matches(targets, candidates, 0.75, 0.05)

        self.assertEqual(matches, [])

    def test_reference_support_uses_top_k_mean_not_maximum(self) -> None:
        target = _target_with_references(
            1,
            [
                np.asarray((1.0, 0.0)),
                np.asarray((0.6, 0.8)),
                np.asarray((0.0, 1.0)),
            ],
        )
        candidate_embedding = np.asarray(
            (np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))),
            dtype=np.float32,
        )

        support = recovery_reference_support_score(
            target,
            candidate_embedding,
            top_k=3,
        )

        self.assertLess(support, 0.80)
        self.assertGreater(float(np.dot(target.centroid, candidate_embedding)), 0.85)
        self.assertEqual(
            assign_recovery_matches(
                [target],
                [
                    RecoveryCandidate(
                        Track(11, (10, 5, 50, 110), 0.9, 0),
                        candidate_embedding,
                    )
                ],
                0.85,
                0.05,
                0.80,
                3,
            ),
            [],
        )

    def test_one_reference_uses_available_reference_when_top_k_is_three(self) -> None:
        target = _target(1, (1.0, 0.0))
        candidate = RecoveryCandidate(
            Track(11, (10, 5, 50, 110), 0.9, 0),
            np.asarray((1.0, 0.0), dtype=np.float32),
        )

        matches = assign_recovery_matches(
            [target], [candidate], 0.85, 0.05, 0.80, 3
        )

        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0].centroid_similarity, 1.0)
        self.assertAlmostEqual(matches[0].reference_support_similarity, 1.0)

    def test_two_references_use_both_when_top_k_is_three(self) -> None:
        target = _target_with_references(
            1,
            [np.asarray((1.0, 0.0)), np.asarray((0.8, 0.6))],
        )
        candidate = np.asarray((1.0, 0.0), dtype=np.float32)

        support = recovery_reference_support_score(target, candidate, top_k=3)

        self.assertAlmostEqual(support, (1.0 + 0.8) / 2.0, places=5)


class CoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((120, 100, 3), dtype=np.uint8)
        self.track_a = Track(3, (10, 5, 50, 110), 0.9, 0)
        self.track_b = Track(8, (55, 5, 90, 110), 0.9, 0)

    def _coordinator(
        self,
        extractor: _FakeReIDExtractor,
        **overrides: object,
    ) -> tuple[TargetManager, TargetRecoveryCoordinator]:
        manager = TargetManager()
        coordinator = TargetRecoveryCoordinator(
            manager,
            extractor,  # type: ignore[arg-type]
            _reid_config(),
            _recovery_config(**overrides),
        )
        return manager, coordinator

    def test_selection_creates_session_target_with_initial_reference(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((1, 0)))
        manager, coordinator = self._coordinator(extractor)

        target = coordinator.select_from_track(self.frame, self.track_a, frame_index=0)

        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.target_id, 1)
        self.assertEqual(target.current_track_id, 3)
        self.assertEqual(target.last_track_id, 3)
        self.assertEqual(target.state, TargetState.ACTIVE)
        self.assertEqual(len(target.reference_embeddings), 1)
        self.assertEqual(extractor.extract_calls, 1)
        self.assertIs(manager.targets[1], target)

    def test_accepted_reference_emits_one_drained_update_event(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((1, 0)))
        _manager, coordinator = self._coordinator(
            extractor,
            reference_update_interval_frames=1,
        )
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None

        coordinator.process_frame(self.frame, [self.track_a], 1)

        events = coordinator.drain_reference_updates()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].target_id, target.target_id)
        self.assertEqual(events[0].frame_index, 1)
        self.assertEqual(len(events[0].reference_embeddings), 2)
        self.assertEqual(events[0].centroid.shape, (2,))
        self.assertTrue(
            np.allclose(events[0].accepted_embedding, np.asarray((1, 0)))
        )
        self.assertEqual(coordinator.drain_reference_updates(), ())

    def test_rejected_reference_emits_no_update_event(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((0, 1)))
        _manager, coordinator = self._coordinator(
            extractor,
            reference_update_interval_frames=1,
        )
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None

        coordinator.process_frame(self.frame, [self.track_a], 1)

        self.assertEqual(len(target.reference_embeddings), 1)
        self.assertEqual(coordinator.drain_reference_updates(), ())

    def test_frame_edge_reference_crop_is_rejected_without_reid_or_update(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((1, 0)))
        _manager, coordinator = self._coordinator(
            extractor,
            reference_update_interval_frames=1,
        )
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None
        original_references = [
            reference.copy() for reference in target.reference_embeddings
        ]
        original_centroid = target.centroid.copy()

        edge_track = Track(3, (0, 10, 40, 110), 0.9, 0)
        coordinator.process_frame(self.frame, [edge_track], 1)

        self.assertEqual(extractor.extract_calls, 1)
        self.assertEqual(extractor.batch_calls, 0)
        self.assertEqual(len(target.reference_embeddings), len(original_references))
        np.testing.assert_allclose(
            target.reference_embeddings[0], original_references[0]
        )
        np.testing.assert_allclose(target.centroid, original_centroid)
        self.assertEqual(coordinator.drain_reference_updates(), ())

    def test_frame_edge_recovery_candidate_is_excluded(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((1, 0)))
        manager, coordinator = self._coordinator(
            extractor,
            recovery_confirmation_hits=1,
        )
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None
        coordinator.process_frame(self.frame, [], 1)

        edge_candidate = Track(11, (0, 10, 40, 110), 0.99, 0)
        complete_candidate = Track(12, (50, 10, 90, 110), 0.99, 0)
        matches = coordinator.process_frame(
            self.frame,
            [edge_candidate, complete_candidate],
            2,
        )

        self.assertEqual(
            [match.candidate.track.track_id for match in matches],
            [12],
        )
        self.assertEqual(target.current_track_id, 12)
        self.assertEqual(extractor.batch_sizes, [1])
        self.assertEqual(manager.target_recovered_count, 1)

    def test_recovery_reference_is_not_emitted_as_gallery_update(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((1, 0)))
        _manager, coordinator = self._coordinator(
            extractor,
            recovery_confirmation_hits=1,
        )
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None
        coordinator.process_frame(self.frame, [], 1)

        matches = coordinator.process_frame(self.frame, [self.track_b], 2)

        self.assertEqual(len(matches), 1)
        self.assertEqual(coordinator.drain_reference_updates(), ())

    def test_invalid_or_small_selection_crop_does_not_create_target(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((1, 0)))
        manager, coordinator = self._coordinator(extractor)
        too_small = Track(3, (0, 0, 5, 5), 0.9, 0)

        self.assertIsNone(coordinator.select_from_track(self.frame, too_small, 0))
        self.assertEqual(manager.targets, {})
        self.assertEqual(extractor.extract_calls, 0)

    def test_short_missing_track_does_not_enter_lost(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((1, 0)))
        manager, coordinator = self._coordinator(extractor, lost_grace_frames=2)
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None

        coordinator.process_frame(self.frame, [], 1)

        self.assertEqual(target.state, TargetState.ACTIVE)
        self.assertEqual(target.current_track_id, 3)
        self.assertEqual(target.last_track_id, 3)

    def test_lost_target_recovers_with_new_track_and_keeps_session_identity(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((1, 0)))
        manager, coordinator = self._coordinator(extractor)
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None
        target_id = target.target_id

        coordinator.process_frame(self.frame, [], 1)
        self.assertEqual(target.state, TargetState.LOST)
        self.assertIsNone(target.current_track_id)
        self.assertEqual(target.last_track_id, 3)

        matches = coordinator.process_frame(self.frame, [self.track_b], 2)

        self.assertEqual(len(matches), 1)
        self.assertIs(manager.targets[target_id], target)
        self.assertEqual(target.current_track_id, 8)
        self.assertEqual(target.last_track_id, 8)
        self.assertEqual(target.state, TargetState.ACTIVE)
        self.assertEqual(manager.selected_track_ids, {8})

    def test_recovery_does_not_immediately_add_candidate_to_reference_bank(self) -> None:
        extractor = _FakeReIDExtractor(
            np.asarray((1, 0)),
            np.asarray((0.98, 0.20)),
        )
        manager, coordinator = self._coordinator(
            extractor,
            recovery_confirmation_hits=1,
            recovery_threshold=0.85,
        )
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None
        coordinator.process_frame(self.frame, [], 1)

        matches = coordinator.process_frame(self.frame, [self.track_b], 2)

        self.assertEqual(len(matches), 1)
        self.assertEqual(len(target.reference_embeddings), 1)
        np.testing.assert_allclose(target.reference_embeddings[0], np.asarray((1, 0)))
        np.testing.assert_allclose(target.centroid, np.asarray((1, 0)))

    def test_open_set_strangers_never_recover_permanently_lost_target(self) -> None:
        stranger = np.asarray((0.84, np.sqrt(1.0 - 0.84**2)), dtype=np.float32)
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), stranger)
        manager, coordinator = self._coordinator(
            extractor,
            recovery_threshold=0.85,
            recovery_reference_support_threshold=0.80,
            recovery_confirmation_hits=2,
            recovery_min_track_age_frames=1,
            recovery_interval_frames=1,
        )
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None
        coordinator.process_frame(self.frame, [], 1)

        for frame_index in range(2, 1002):
            coordinator.process_frame(self.frame, [self.track_b], frame_index)

        self.assertEqual(target.state, TargetState.LOST)
        self.assertIsNone(target.current_track_id)
        self.assertEqual(coordinator.recovery_accepted_count, 0)
        self.assertEqual(manager.target_recovered_count, 0)

    def test_hard_negative_centroid_passes_but_support_blocks_recovery(self) -> None:
        target = _target_with_references(
            1,
            [
                np.asarray((np.cos(np.deg2rad(-60)), np.sin(np.deg2rad(-60)))),
                np.asarray((1.0, 0.0)),
                np.asarray((np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60)))),
            ],
        )
        candidate_embedding = np.asarray(
            (np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))),
            dtype=np.float32,
        )
        candidate = RecoveryCandidate(
            Track(11, (10, 5, 50, 110), 0.9, 0),
            candidate_embedding,
        )

        matches = assign_recovery_matches(
            [target],
            [candidate],
            recovery_threshold=0.85,
            recovery_margin=0.05,
            recovery_reference_support_threshold=0.80,
            recovery_reference_support_top_k=3,
        )

        self.assertGreater(recovery_reference_support_score(target, candidate_embedding), 0.0)
        self.assertEqual(matches, [])

    def test_true_return_passes_centroid_support_and_confirmation(self) -> None:
        extractor = _FakeReIDExtractor(
            np.asarray((1, 0)),
            np.asarray((0.98, 0.20)),
        )
        manager, coordinator = self._coordinator(
            extractor,
            recovery_threshold=0.85,
            recovery_reference_support_threshold=0.80,
            recovery_confirmation_hits=2,
        )
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None
        coordinator.process_frame(self.frame, [], 1)

        self.assertEqual(coordinator.process_frame(self.frame, [self.track_b], 2), [])
        self.assertEqual(coordinator.pending, {(target.target_id, 8): 1})
        matches = coordinator.process_frame(self.frame, [self.track_b], 3)

        self.assertEqual(len(matches), 1)
        self.assertEqual(target.state, TargetState.ACTIVE)
        self.assertEqual(target.current_track_id, 8)
        self.assertEqual(manager.target_recovered_count, 1)

    def test_low_similarity_candidate_stays_lost(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((0, 1)))
        manager, coordinator = self._coordinator(extractor)
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None

        coordinator.process_frame(self.frame, [], 1)
        matches = coordinator.process_frame(self.frame, [self.track_b], 2)

        self.assertEqual(matches, [])
        self.assertEqual(target.state, TargetState.LOST)
        self.assertIsNone(target.current_track_id)
        self.assertEqual(manager.selected_track_ids, set())

    def test_recovery_interval_prevents_candidate_reid_on_every_frame(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((0, 1)))
        _manager, coordinator = self._coordinator(
            extractor,
            recovery_interval_frames=3,
        )
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None

        coordinator.process_frame(self.frame, [], 1)
        coordinator.process_frame(self.frame, [self.track_b], 2)
        coordinator.process_frame(self.frame, [self.track_b], 3)
        coordinator.process_frame(self.frame, [self.track_b], 4)

        self.assertEqual(extractor.batch_calls, 1)
        self.assertEqual(extractor.batch_sizes, [1])

    def test_active_track_is_excluded_from_recovery_candidates(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((1, 0)))
        manager, coordinator = self._coordinator(extractor)
        target_a = coordinator.select_from_track(self.frame, self.track_a, 0)
        target_b = coordinator.select_from_track(self.frame, self.track_b, 0)
        assert target_a is not None and target_b is not None

        # Lose A while B remains ACTIVE; only B's replacement candidate may be
        # considered, and B itself must not be sent through recovery ReID.
        coordinator.process_frame(self.frame, [self.track_b], 1)
        self.assertEqual(target_a.state, TargetState.LOST)
        self.assertEqual(target_b.state, TargetState.ACTIVE)
        self.assertEqual(manager.recovery_candidates([self.track_b]), [])

    def test_no_lost_target_does_not_run_candidate_reid(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((1, 0)))
        _manager, coordinator = self._coordinator(extractor)
        coordinator.select_from_track(self.frame, self.track_a, 0)

        coordinator.process_frame(self.frame, [self.track_a, self.track_b], 0)

        self.assertEqual(extractor.batch_calls, 0)

    def test_recovery_requires_two_valid_attempts_when_configured(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((1, 0)))
        manager, coordinator = self._coordinator(
            extractor,
            recovery_confirmation_hits=2,
            recovery_min_track_age_frames=1,
        )
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None

        coordinator.process_frame(self.frame, [], 1)
        self.assertEqual(coordinator.process_frame(self.frame, [self.track_b], 2), [])
        self.assertEqual(coordinator.pending, {(target.target_id, 8): 1})
        matches = coordinator.process_frame(self.frame, [self.track_b], 3)

        self.assertEqual(len(matches), 1)
        self.assertEqual(target.current_track_id, 8)
        self.assertEqual(coordinator.pending, {})

    def test_recovery_candidate_age_is_continuous(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((1, 0)))
        manager, coordinator = self._coordinator(
            extractor,
            recovery_min_track_age_frames=3,
            recovery_confirmation_hits=1,
        )
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None
        coordinator.process_frame(self.frame, [], 1)

        coordinator.process_frame(self.frame, [self.track_b], 2)
        coordinator.process_frame(self.frame, [self.track_b], 3)
        self.assertEqual(extractor.batch_calls, 0)
        matches = coordinator.process_frame(self.frame, [self.track_b], 4)

        self.assertEqual(len(matches), 1)
        self.assertEqual(coordinator.track_ages[8], 3)

    def test_quality_rejection_does_not_count_as_recovery_mismatch(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((1, 0)))
        manager, coordinator = self._coordinator(
            extractor,
            recovery_confirmation_hits=2,
            recovery_min_track_age_frames=1,
        )
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None
        coordinator.process_frame(self.frame, [], 1)
        coordinator.process_frame(self.frame, [self.track_b], 2)
        self.assertEqual(coordinator.pending, {(target.target_id, 8): 1})

        overlapping = Track(9, (60, 0, 90, 110), 0.9, 0)
        coordinator.process_frame(self.frame, [self.track_b, overlapping], 3)

        self.assertEqual(coordinator.pending, {(target.target_id, 8): 1})
        self.assertEqual(target.state, TargetState.LOST)

    def test_lost_target_recovery_reuses_current_frame_cache(self) -> None:
        extractor = _FakeReIDExtractor(np.asarray((1, 0)), np.asarray((1, 0)))
        manager = TargetManager()
        coordinator = TargetRecoveryCoordinator(
            manager,
            extractor,  # type: ignore[arg-type]
            _reid_config(),
            _recovery_config(recovery_confirmation_hits=1),
            embedding_cache=ReIDFrameCache(),
        )
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None
        coordinator.process_frame(self.frame, [], 1)
        cache = coordinator.embedding_cache
        assert cache is not None
        cache.begin_frame(2)
        cache.put(8, np.asarray((1, 0), dtype=np.float32), 2)

        matches = coordinator.process_frame(self.frame, [self.track_b], 2)

        self.assertEqual(len(matches), 1)
        self.assertEqual(extractor.batch_calls, 0)

    def test_valid_failed_recovery_attempt_clears_pending(self) -> None:
        extractor = _SequenceReIDExtractor(
            np.asarray((1, 0)),
            [np.asarray((1, 0)), np.asarray((0, 1))],
        )
        manager, coordinator = self._coordinator(
            extractor,
            recovery_confirmation_hits=2,
            recovery_min_track_age_frames=1,
        )
        target = coordinator.select_from_track(self.frame, self.track_a, 0)
        assert target is not None
        coordinator.process_frame(self.frame, [], 1)
        coordinator.process_frame(self.frame, [self.track_b], 2)
        self.assertEqual(coordinator.pending, {(target.target_id, 8): 1})

        coordinator.process_frame(self.frame, [self.track_b], 3)

        self.assertEqual(coordinator.pending, {})
        self.assertEqual(target.state, TargetState.LOST)


if __name__ == "__main__":
    unittest.main()
