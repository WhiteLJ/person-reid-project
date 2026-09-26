from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.active_identity_guard import (
    ActiveIdentityGuard,
    min_area_overlap_ratio,
)
from src.config import (
    ActiveIdentityGuardConfig,
    ReIDConfig,
    ReIDQualityConfig,
    ReIDRecoveryConfig,
)
from src.models import TargetState, Track
from src.reid_frame_cache import ReIDFrameCache
from src.target_manager import TargetManager
from src.target_recovery import TargetRecoveryCoordinator


class _SequenceExtractor:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = [np.asarray(output, dtype=np.float32) for output in outputs]
        self.batch_calls = 0

    def extract_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        self.batch_calls += 1
        if not self.outputs:
            raise AssertionError("unexpected Identity Guard ReID call")
        output = self.outputs.pop(0)
        if output.ndim == 1:
            return np.repeat(output[None, :], len(crops), axis=0)
        if output.shape[0] != len(crops):
            raise AssertionError("fake output batch size mismatch")
        return output


def _reid_config() -> ReIDConfig:
    return ReIDConfig(
        model_name="osnet_x0_25",
        weight=Path("unused.pth"),
        image_height=256,
        image_width=128,
        min_crop_width=20,
        min_crop_height=50,
    )


def _recovery_config() -> ReIDRecoveryConfig:
    return ReIDRecoveryConfig(
        lost_grace_frames=1,
        reference_update_interval_frames=15,
        recovery_interval_frames=5,
        max_reference_embeddings=8,
        recovery_threshold=0.85,
        recovery_margin=0.05,
        reference_update_threshold=0.80,
        recovery_reference_support_threshold=0.80,
        recovery_reference_support_top_k=3,
        recovery_min_track_age_frames=1,
        recovery_confirmation_hits=2,
        recovery_pending_max_age_frames=60,
    )


def _guard_config(**overrides: object) -> ActiveIdentityGuardConfig:
    values: dict[str, object] = {
        "enabled": True,
        "bbox_history_frames": 5,
        "max_width_growth_ratio": 1.45,
        "max_area_growth_ratio": 1.70,
        "max_center_shift_ratio": 0.35,
        "overlap_trigger_ratio": 0.15,
        "clear_overlap_ratio": 0.10,
        "confirmation_hits": 2,
        "max_candidates": 3,
    }
    values.update(overrides)
    return ActiveIdentityGuardConfig(**values)  # type: ignore[arg-type]


def _make_guard(extractor: _SequenceExtractor, manager: TargetManager) -> ActiveIdentityGuard:
    return ActiveIdentityGuard(
        target_manager=manager,
        reid_extractor=extractor,  # type: ignore[arg-type]
        reid_config=_reid_config(),
        recovery_config=_recovery_config(),
        quality_config=ReIDQualityConfig(
            max_person_overlap_ratio=0.40,
            min_frame_edge_margin_ratio=0.0,
        ),
        guard_config=_guard_config(),
        embedding_cache=ReIDFrameCache(),
    )


class ActiveIdentityGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((300, 300, 3), dtype=np.uint8)
        self.track_a = Track(11, (50, 50, 130, 250), 0.9, 0)
        self.manager = TargetManager()
        self.target = self.manager.select(
            self.track_a,
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            frame_index=0,
        )

    def test_bbox_growth_with_person_overlap_enters_ambiguity(self) -> None:
        extractor = _SequenceExtractor([])
        guard = _make_guard(extractor, self.manager)

        guard.process_frame(self.frame, [self.track_a], 0)
        expanded = Track(11, (50, 50, 180, 250), 0.9, 0)
        other = Track(12, (100, 50, 180, 250), 0.9, 0)
        result = guard.process_frame(self.frame, [expanded, other], 1)

        self.assertEqual(result.blocked_reference_update_target_ids, {self.target.target_id})
        self.assertEqual(extractor.batch_calls, 0)

    def test_arm_like_bbox_growth_without_overlap_does_not_run_reid(self) -> None:
        extractor = _SequenceExtractor([])
        guard = _make_guard(extractor, self.manager)

        guard.process_frame(self.frame, [self.track_a], 0)
        wider = Track(11, (50, 50, 180, 250), 0.9, 0)
        result = guard.process_frame(self.frame, [wider], 1)

        self.assertEqual(result.blocked_reference_update_target_ids, set())
        self.assertEqual(extractor.batch_calls, 0)
        self.assertEqual(self.target.state, TargetState.ACTIVE)

    def test_physical_identity_drift_corrects_after_two_clean_verifications(self) -> None:
        extractor = _SequenceExtractor(
            [
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            ]
        )
        guard = _make_guard(extractor, self.manager)
        guard.process_frame(self.frame, [self.track_a], 0)
        guard.process_frame(
            self.frame,
            [
                Track(11, (50, 50, 180, 250), 0.9, 0),
                Track(12, (100, 50, 180, 250), 0.9, 0),
            ],
            1,
        )

        wrong_track = Track(11, (180, 50, 260, 250), 0.9, 0)
        real_track = Track(12, (50, 50, 130, 250), 0.9, 0)
        first = guard.process_frame(self.frame, [wrong_track, real_track], 2)
        self.assertEqual(first.corrected_target_ids, set())
        self.assertEqual(self.target.current_track_id, 11)
        second = guard.process_frame(self.frame, [wrong_track, real_track], 3)

        self.assertEqual(second.corrected_target_ids, {self.target.target_id})
        self.assertEqual(self.target.current_track_id, 12)
        self.assertEqual(self.target.state, TargetState.ACTIVE)
        self.assertEqual(self.manager.target_lost_count, 0)
        self.assertEqual(self.manager.target_recovered_count, 0)
        self.assertEqual(extractor.batch_calls, 2)

    def test_identity_mismatch_moves_target_to_lost_after_two_clean_failures(self) -> None:
        extractor = _SequenceExtractor(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        guard = _make_guard(extractor, self.manager)
        guard.process_frame(self.frame, [self.track_a], 0)
        guard.process_frame(
            self.frame,
            [
                Track(11, (50, 50, 180, 250), 0.9, 0),
                Track(12, (100, 50, 180, 250), 0.9, 0),
            ],
            1,
        )
        tracks = [
            Track(11, (180, 50, 260, 250), 0.9, 0),
            Track(12, (50, 50, 130, 250), 0.9, 0),
        ]
        guard.process_frame(self.frame, tracks, 2)
        result = guard.process_frame(self.frame, tracks, 3)

        self.assertEqual(result.identity_lost_target_ids, {self.target.target_id})
        self.assertEqual(self.target.state, TargetState.LOST)
        self.assertIsNone(self.target.current_track_id)
        self.assertEqual(self.manager.target_lost_count, 1)

    def test_overlap_formula_uses_smaller_bbox_area(self) -> None:
        self.assertAlmostEqual(
            min_area_overlap_ratio((0, 0, 100, 100), (50, 0, 100, 100)),
            1.0,
        )

    def test_ambiguity_blocks_active_reference_update(self) -> None:
        extractor = _SequenceExtractor([])
        guard = _make_guard(extractor, self.manager)
        guard.process_frame(self.frame, [self.track_a], 0)
        expanded = Track(11, (50, 50, 180, 250), 0.9, 0)
        other = Track(12, (100, 50, 180, 250), 0.9, 0)
        result = guard.process_frame(self.frame, [expanded, other], 1)
        reference_count = len(self.target.reference_embeddings)
        recovery = TargetRecoveryCoordinator(
            target_manager=self.manager,
            reid_extractor=extractor,  # type: ignore[arg-type]
            reid_config=_reid_config(),
            recovery_config=_recovery_config(),
            quality_config=ReIDQualityConfig(
                max_person_overlap_ratio=0.40,
                min_frame_edge_margin_ratio=0.0,
            ),
        )

        recovery.process_frame(
            self.frame,
            [expanded, other],
            1,
            reference_update_blocked_target_ids=(
                result.blocked_reference_update_target_ids
            ),
        )

        self.assertEqual(len(self.target.reference_embeddings), reference_count)
        self.assertEqual(extractor.batch_calls, 0)


if __name__ == "__main__":
    unittest.main()
