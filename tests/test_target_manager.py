from __future__ import annotations

import unittest

import numpy as np

from src.models import TargetState, Track
from src.gallery import TargetGallery
from src.target_manager import TargetManager


def _embedding(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


class TargetManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track_a = Track(7, (0, 0, 20, 40), 0.9, 0)
        self.track_b = Track(8, (30, 0, 50, 40), 0.9, 0)
        self.track_c = Track(9, (60, 0, 80, 40), 0.9, 0)

    def test_selects_multiple_tracks(self) -> None:
        manager = TargetManager()

        manager.select(self.track_a, _embedding(1, 0))
        manager.select(self.track_b, _embedding(0, 1))

        self.assertEqual(manager.selected_track_ids, {7, 8})
        self.assertEqual(
            manager.selected_tracks([self.track_a, self.track_b]),
            [self.track_a, self.track_b],
        )
        self.assertEqual(len(manager.targets), 2)

    def test_selecting_same_track_is_idempotent(self) -> None:
        manager = TargetManager()

        first = manager.select(self.track_a, _embedding(1, 0))
        second = manager.select(self.track_a, _embedding(0, 1))

        self.assertIs(first, second)
        self.assertEqual(manager.selected_track_ids, {7})
        np.testing.assert_array_equal(first.centroid, _embedding(1, 0))

    def test_deselecting_one_track_preserves_other_targets(self) -> None:
        manager = TargetManager()
        manager.select(self.track_a, _embedding(1, 0))
        manager.select(self.track_b, _embedding(0, 1))

        self.assertTrue(manager.deselect(self.track_a))

        self.assertEqual(manager.selected_track_ids, {8})
        self.assertEqual(len(manager.targets), 1)

    def test_deselecting_non_selected_track_changes_nothing(self) -> None:
        manager = TargetManager()
        manager.select(self.track_a, _embedding(1, 0))

        self.assertFalse(manager.deselect(self.track_c))

        self.assertEqual(manager.selected_track_ids, {7})

    def test_clear_removes_all_targets_and_references(self) -> None:
        manager = TargetManager()
        manager.select(self.track_a, _embedding(1, 0))
        manager.select(self.track_b, _embedding(0, 1))

        manager.clear()

        self.assertEqual(manager.selected_track_ids, set())
        self.assertEqual(manager.targets, {})

    def test_missing_selected_track_is_not_rebound(self) -> None:
        manager = TargetManager()
        target = manager.select(self.track_a, _embedding(1, 0))

        manager.update_visibility([self.track_b], lost_grace_frames=2, frame_index=1)
        self.assertEqual(target.state, TargetState.ACTIVE)
        self.assertEqual(target.current_track_id, 7)
        self.assertEqual(manager.selected_tracks([self.track_b]), [])

    def test_lost_target_retains_last_track_id(self) -> None:
        manager = TargetManager()
        target = manager.select(self.track_a, _embedding(1, 0))

        manager.update_visibility([], lost_grace_frames=1, frame_index=5)

        self.assertEqual(target.state, TargetState.LOST)
        self.assertIsNone(target.current_track_id)
        self.assertEqual(target.last_track_id, 7)
        self.assertEqual(manager.selected_track_ids, set())

    def test_active_target_never_expires(self) -> None:
        manager = TargetManager()
        target = manager.select(self.track_a, _embedding(1, 0))

        self.assertEqual(manager.expire_lost_targets(1.0), ())
        self.assertIn(target.target_id, manager.targets)
        self.assertEqual(target.state, TargetState.ACTIVE)

    def test_lost_target_before_ttl_remains_recoverable(self) -> None:
        manager = TargetManager()
        target = manager.select(self.track_a, _embedding(1, 0))
        manager.update_visibility([], lost_grace_frames=1, frame_index=1)

        lost_at = manager.lost_since(target.target_id)
        self.assertIsNotNone(lost_at)
        assert lost_at is not None
        self.assertEqual(
            manager.expire_lost_targets(10.0, now=lost_at + 9.0),
            (),
        )
        manager.recover(
            target_id=target.target_id,
            track=self.track_b,
            embedding=_embedding(1, 0),
            similarity=1.0,
            frame_index=2,
            max_reference_embeddings=8,
            reference_update_threshold=0.80,
        )

        self.assertEqual(target.state, TargetState.ACTIVE)
        self.assertEqual(target.current_track_id, self.track_b.track_id)

    def test_lost_target_after_ttl_is_removed(self) -> None:
        manager = TargetManager()
        target = manager.select(self.track_a, _embedding(1, 0))
        manager.update_visibility([], lost_grace_frames=1, frame_index=1)
        lost_at = manager.lost_since(target.target_id)
        assert lost_at is not None

        expired = manager.expire_lost_targets(5.0, now=lost_at + 5.0)

        self.assertEqual(tuple(item.target_id for item in expired), (target.target_id,))
        self.assertNotIn(target.target_id, manager.targets)
        self.assertIsNone(manager.lost_since(target.target_id))

    def test_expiring_target_does_not_delete_gallery_identity(self) -> None:
        manager = TargetManager()
        target = manager.select(self.track_a, _embedding(1, 0))
        gallery = TargetGallery()
        person = gallery.enroll(target)
        gallery.attach_session_target(target.target_id, person.person_id)
        manager.update_visibility([], lost_grace_frames=1, frame_index=1)
        lost_at = manager.lost_since(target.target_id)
        assert lost_at is not None

        manager.expire_lost_targets(1.0, now=lost_at + 1.0)
        gallery.detach_session_target(target.target_id)

        self.assertIsNotNone(gallery.get(person.person_id))
        self.assertIsNone(gallery.person_for_session_target(target.target_id))

    def test_zero_retention_preserves_permanent_lost_behavior(self) -> None:
        manager = TargetManager()
        target = manager.select(self.track_a, _embedding(1, 0))
        manager.update_visibility([], lost_grace_frames=1, frame_index=1)

        self.assertEqual(manager.expire_lost_targets(0.0), ())
        self.assertIn(target.target_id, manager.targets)
        self.assertEqual(target.state, TargetState.LOST)

    def test_short_loss_returns_to_active_without_rebinding(self) -> None:
        manager = TargetManager()
        target = manager.select(self.track_a, _embedding(1, 0))

        manager.update_visibility([], lost_grace_frames=2, frame_index=1)
        manager.update_visibility([self.track_a], lost_grace_frames=2, frame_index=2)

        self.assertEqual(target.state, TargetState.ACTIVE)
        self.assertEqual(target.current_track_id, 7)
        self.assertEqual(target.missing_frames, 0)

    def test_reference_update_is_consistency_gated_and_bounded(self) -> None:
        manager = TargetManager()
        target = manager.select(self.track_a, _embedding(1, 0), frame_index=0)

        rejected = manager.add_reference(
            target.target_id,
            _embedding(0, 1),
            frame_index=1,
            max_reference_embeddings=3,
            reference_update_threshold=0.80,
        )
        accepted = manager.add_reference(
            target.target_id,
            _embedding(0.99, 0.1),
            frame_index=2,
            max_reference_embeddings=2,
            reference_update_threshold=0.80,
        )

        self.assertFalse(rejected)
        self.assertTrue(accepted)
        self.assertEqual(len(target.reference_embeddings), 2)
        self.assertTrue(
            all(np.allclose(reference, _embedding(1, 0)) or reference[0] > 0.9
                for reference in target.reference_embeddings)
        )

    def test_runtime_reference_bank_remains_bounded_fifo(self) -> None:
        manager = TargetManager()
        target = manager.select(self.track_a, _embedding(1, 0), frame_index=0)

        self.assertTrue(
            manager.add_reference(
                target.target_id,
                _embedding(0.90, 0.435),
                frame_index=1,
                max_reference_embeddings=2,
                reference_update_threshold=0.80,
            )
        )
        self.assertTrue(
            manager.add_reference(
                target.target_id,
                _embedding(0.85, 0.527),
                frame_index=2,
                max_reference_embeddings=2,
                reference_update_threshold=0.80,
            )
        )

        self.assertEqual(len(target.reference_embeddings), 2)
        self.assertFalse(
            np.allclose(target.reference_embeddings[0], _embedding(1, 0))
        )

    def test_target_reference_banks_are_independent(self) -> None:
        manager = TargetManager()
        target_a = manager.select(self.track_a, _embedding(1, 0))
        target_b = manager.select(self.track_b, _embedding(0, 1))

        target_a.reference_embeddings.append(_embedding(1, 1))

        self.assertEqual(len(target_a.reference_embeddings), 2)
        self.assertEqual(len(target_b.reference_embeddings), 1)


if __name__ == "__main__":
    unittest.main()
