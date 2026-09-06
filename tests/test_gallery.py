from __future__ import annotations

import unittest

import numpy as np

from src.gallery import TargetGallery, format_person_id
from src.models import SessionTarget, TargetState, Track
from src.target_manager import TargetManager


def _session_target(target_id: int, values: tuple[float, ...]) -> SessionTarget:
    embedding = np.asarray(values, dtype=np.float32)
    embedding /= np.linalg.norm(embedding)
    return SessionTarget(
        target_id=target_id,
        current_track_id=target_id + 10,
        last_track_id=target_id + 10,
        state=TargetState.ACTIVE,
        reference_embeddings=[embedding.copy()],
        centroid=embedding.copy(),
    )


class TargetGalleryTests(unittest.TestCase):
    def test_person_id_and_label_are_generated(self) -> None:
        gallery = TargetGallery()

        person = gallery.enroll(_session_target(3, (1, 0)))

        self.assertEqual(person.person_id, 1)
        self.assertEqual(person.label, "Target P001")
        self.assertEqual(format_person_id(2), "P002")

    def test_enrollment_snapshots_reference_bank_and_centroid(self) -> None:
        target = _session_target(3, (1, 0))
        gallery = TargetGallery()

        person = gallery.enroll(target)
        target.reference_embeddings[0][0] = 0.0
        target.centroid[0] = 0.0
        person.reference_embeddings[0][1] = 0.0

        self.assertNotEqual(float(person.reference_embeddings[0][0]), 0.0)
        self.assertNotEqual(float(person.centroid[0]), 0.0)
        self.assertEqual(person.reference_embeddings[0].dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(person.centroid)), 1.0, places=6)

    def test_two_targets_get_independent_people_and_references(self) -> None:
        target_a = _session_target(1, (1, 0))
        target_b = _session_target(2, (0, 1))
        gallery = TargetGallery()

        person_a = gallery.enroll(target_a)
        person_b = gallery.enroll(target_b)
        person_a.reference_embeddings.append(np.asarray((1, 1), dtype=np.float32))

        self.assertNotEqual(person_a.person_id, person_b.person_id)
        self.assertEqual(len(person_a.reference_embeddings), 2)
        self.assertEqual(len(person_b.reference_embeddings), 1)

    def test_duplicate_enrollment_is_idempotent(self) -> None:
        target = _session_target(3, (1, 0))
        gallery = TargetGallery()

        first = gallery.enroll(target)
        second = gallery.enroll(target)

        self.assertIs(first, second)
        self.assertEqual(len(gallery.all_people()), 1)

    def test_get_and_all_people_are_available_in_person_id_order(self) -> None:
        gallery = TargetGallery()
        first = gallery.enroll(_session_target(1, (1, 0)))
        second = gallery.enroll(_session_target(2, (0, 1)))

        self.assertIs(gallery.get(first.person_id), first)
        self.assertEqual(gallery.get(999), None)
        self.assertEqual(gallery.all_people(), (first, second))

    def test_remove_person_clears_mapping_but_keeps_session_target(self) -> None:
        target = _session_target(3, (1, 0))
        gallery = TargetGallery()
        person = gallery.enroll(target)

        self.assertTrue(gallery.remove(person.person_id))

        self.assertIsNone(gallery.get(person.person_id))
        self.assertIsNone(gallery.person_for_session_target(target.target_id))
        self.assertEqual(target.current_track_id, 13)
        self.assertFalse(gallery.remove(person.person_id))

    def test_removed_person_allows_reenrollment_as_new_person(self) -> None:
        target = _session_target(3, (1, 0))
        gallery = TargetGallery()
        first = gallery.enroll(target)
        gallery.remove(first.person_id)

        second = gallery.enroll(target)

        self.assertEqual(second.person_id, 2)
        self.assertIs(gallery.person_for_session_target(target.target_id), second)

    def test_detaching_target_keeps_person_but_removes_association(self) -> None:
        target = _session_target(3, (1, 0))
        gallery = TargetGallery()
        person = gallery.enroll(target)

        self.assertTrue(gallery.detach_session_target(target.target_id))

        self.assertIs(gallery.get(person.person_id), person)
        self.assertIsNone(gallery.person_for_session_target(target.target_id))
        self.assertFalse(gallery.detach_session_target(target.target_id))

    def test_detach_all_keeps_people_and_clears_all_associations(self) -> None:
        target_a = _session_target(1, (1, 0))
        target_b = _session_target(2, (0, 1))
        gallery = TargetGallery()
        person_a = gallery.enroll(target_a)
        person_b = gallery.enroll(target_b)

        gallery.detach_all_session_targets()

        self.assertEqual(len(gallery.all_people()), 2)
        self.assertIs(gallery.get(person_a.person_id), person_a)
        self.assertIs(gallery.get(person_b.person_id), person_b)
        self.assertIsNone(gallery.person_for_session_target(target_a.target_id))
        self.assertIsNone(gallery.person_for_session_target(target_b.target_id))

    def test_clear_gallery_does_not_clear_session_target(self) -> None:
        target = _session_target(3, (1, 0))
        gallery = TargetGallery()
        gallery.enroll(target)

        gallery.clear()

        self.assertEqual(gallery.all_people(), ())
        self.assertEqual(target.target_id, 3)
        self.assertEqual(target.state, TargetState.ACTIVE)

    def test_empty_or_invalid_target_reference_is_rejected(self) -> None:
        gallery = TargetGallery()
        empty_target = SessionTarget(
            target_id=1,
            current_track_id=11,
            last_track_id=11,
            state=TargetState.ACTIVE,
        )

        with self.assertRaises(ValueError):
            gallery.enroll(empty_target)

    def test_new_target_id_does_not_inherit_old_gallery_mapping(self) -> None:
        manager = TargetManager()
        track_old = Track(11, (0, 0, 20, 40), 0.9, 0)
        track_new = Track(12, (30, 0, 50, 40), 0.9, 0)
        target_old = manager.select(track_old, np.asarray((1, 0), dtype=np.float32))
        gallery = TargetGallery()
        old_person = gallery.enroll(target_old)
        gallery.detach_session_target(target_old.target_id)
        manager.clear()
        target_new = manager.select(track_new, np.asarray((0, 1), dtype=np.float32))

        self.assertNotEqual(target_old.target_id, target_new.target_id)
        self.assertIsNone(gallery.person_for_session_target(target_new.target_id))
        new_person = gallery.enroll(target_new)

        self.assertNotEqual(old_person.person_id, new_person.person_id)
        self.assertIs(gallery.person_for_session_target(target_new.target_id), new_person)

    def test_track_id_change_does_not_change_gallery_person(self) -> None:
        target = _session_target(3, (1, 0))
        gallery = TargetGallery()
        person = gallery.enroll(target)

        target.current_track_id = 21
        target.last_track_id = 21

        self.assertIs(gallery.person_for_session_target(target.target_id), person)

    def test_r_deletes_session_target_but_not_gallery_person(self) -> None:
        manager = TargetManager()
        track = Track(7, (0, 0, 20, 40), 0.9, 0)
        target = manager.select(track, np.asarray((1, 0), dtype=np.float32))
        gallery = TargetGallery()
        person = gallery.enroll(target)

        gallery.detach_session_target(target.target_id)
        self.assertTrue(manager.deselect(track))

        self.assertIs(gallery.get(person.person_id), person)
        self.assertIsNone(gallery.person_for_session_target(target.target_id))

    def test_c_clears_session_targets_but_not_gallery_people(self) -> None:
        manager = TargetManager()
        track_a = Track(7, (0, 0, 20, 40), 0.9, 0)
        track_b = Track(8, (30, 0, 50, 40), 0.9, 0)
        target_a = manager.select(track_a, np.asarray((1, 0), dtype=np.float32))
        target_b = manager.select(track_b, np.asarray((0, 1), dtype=np.float32))
        gallery = TargetGallery()
        person_a = gallery.enroll(target_a)
        person_b = gallery.enroll(target_b)

        gallery.detach_all_session_targets(tuple(manager.targets))
        manager.clear()

        self.assertEqual(manager.targets, {})
        self.assertIs(gallery.get(person_a.person_id), person_a)
        self.assertIs(gallery.get(person_b.person_id), person_b)
        self.assertIsNone(gallery.person_for_session_target(target_a.target_id))
        self.assertIsNone(gallery.person_for_session_target(target_b.target_id))


if __name__ == "__main__":
    unittest.main()
