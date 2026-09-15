from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src.database import GalleryRepository, RepositoryError
from src.config import GalleryEnrichmentConfig
from src.gallery import TargetGallery
from src.gallery_service import GalleryPersistenceService
from src.models import SessionTarget, TargetState
from src.target_recovery import ReferenceUpdateEvent


_TEST_TEMP_PARENT = Path(__file__).resolve().parents[1] / ".test_tmp"
_TEST_TEMP_PARENT.mkdir(parents=True, exist_ok=True)


def _remove_database_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)


def _embedding() -> np.ndarray:
    value = np.zeros((512,), dtype=np.float32)
    value[0] = 1.0
    return value


def _target(target_id: int) -> SessionTarget:
    embedding = _embedding()
    return SessionTarget(
        target_id=target_id,
        current_track_id=target_id + 10,
        last_track_id=target_id + 10,
        state=TargetState.ACTIVE,
        reference_embeddings=[embedding.copy()],
        centroid=embedding.copy(),
    )


class FailingRepository:
    def __init__(self) -> None:
        self.save_calls = 0

    def save_person(self, person) -> None:
        self.save_calls += 1
        raise RepositoryError("forced save failure")


class FailingUpdateRepository:
    def __init__(self, delegate: GalleryRepository) -> None:
        self.delegate = delegate

    def save_person(self, person) -> None:
        self.delegate.save_person(person)

    def update_person_features(self, person_id, reference_embeddings, centroid) -> None:
        del person_id, reference_embeddings, centroid
        raise RepositoryError("forced update failure")

    def load_person(self, person_id):
        return self.delegate.load_person(person_id)


def _second_embedding() -> np.ndarray:
    value = np.zeros((512,), dtype=np.float32)
    value[1] = 1.0
    return value


def _event(target: SessionTarget, references: list[np.ndarray]) -> ReferenceUpdateEvent:
    return ReferenceUpdateEvent(
        target_id=target.target_id,
        frame_index=20,
        reference_embeddings=tuple(reference.copy() for reference in references),
        centroid=references[-1].copy(),
        target_state=TargetState.ACTIVE,
    )


class GalleryPersistenceServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database_path = _TEST_TEMP_PARENT / "gallery_service_tests.db"
        _remove_database_files(self.database_path)
        self.repository = GalleryRepository(self.database_path)
        self.gallery = TargetGallery()
        self.service = GalleryPersistenceService(self.gallery, self.repository)

    def tearDown(self) -> None:
        _remove_database_files(self.database_path)

    def test_new_enrollment_is_saved_and_duplicate_is_not_written_again(self) -> None:
        target = _target(1)
        first = self.service.enroll(target)
        second = self.service.enroll(target)

        self.assertIs(first, second)
        self.assertEqual(tuple(p.person_id for p in self.repository.load_all()), (1,))

    def test_new_enrollment_failure_rolls_back_only_new_person(self) -> None:
        failing_service = GalleryPersistenceService(TargetGallery(), FailingRepository())

        with self.assertRaises(RepositoryError):
            failing_service.enroll(_target(1))

        self.assertEqual(failing_service.gallery.all_people(), ())
        self.assertIsNone(failing_service.gallery.person_for_session_target(1))

    def test_duplicate_enrollment_failure_does_not_remove_existing_person(self) -> None:
        gallery = TargetGallery()
        target = _target(1)
        existing = gallery.enroll(target)
        repository = FailingRepository()
        service = GalleryPersistenceService(gallery, repository)

        result = service.enroll(target)

        self.assertIs(result, existing)
        self.assertEqual(repository.save_calls, 0)
        self.assertIs(gallery.get(existing.person_id), existing)
        self.assertIs(gallery.person_for_session_target(target.target_id), existing)

    def test_remove_deletes_database_then_memory(self) -> None:
        target = _target(1)
        person = self.service.enroll(target)

        self.assertTrue(self.service.remove(person.person_id))

        self.assertIsNone(self.gallery.get(person.person_id))
        self.assertEqual(self.repository.load_all(), ())

    def test_clear_deletes_database_then_memory(self) -> None:
        self.service.enroll(_target(1))
        self.service.enroll(_target(2))

        self.service.clear()

        self.assertEqual(self.gallery.all_people(), ())
        self.assertEqual(self.repository.load_all(), ())

    def test_accepted_reference_enriches_explicit_gallery_person(self) -> None:
        target = _target(1)
        person = self.service.enroll(target)
        updated = self.service.enrich_reference_update(
            _event(target, [_embedding(), _second_embedding()])
        )

        self.assertTrue(updated)
        persisted = self.repository.load_all()[0]
        self.assertEqual(persisted.person_id, person.person_id)
        self.assertEqual(len(persisted.reference_embeddings), 2)
        in_memory = self.gallery.get(person.person_id)
        assert in_memory is not None
        self.assertEqual(len(in_memory.reference_embeddings), 2)

    def test_enrichment_respects_post_recovery_stable_cooldown(self) -> None:
        service = GalleryPersistenceService(
            self.gallery,
            self.repository,
            enrichment_config=GalleryEnrichmentConfig(
                post_recovery_stable_frames=2
            ),
        )
        target = _target(1)
        service.enroll(target)

        target.state = TargetState.LOST
        target.current_track_id = None
        service.update_runtime_state([target], 1)
        target.state = TargetState.ACTIVE
        target.current_track_id = 11
        target.last_recovery_frame = 2
        service.update_runtime_state([target], 2)
        self.assertFalse(
            service.enrich_reference_update(
                _event(target, [_embedding(), _second_embedding()])
            )
        )

        service.update_runtime_state([target], 3)
        self.assertFalse(
            service.enrich_reference_update(
                _event(target, [_embedding(), _second_embedding()])
            )
        )
        service.update_runtime_state([target], 4)
        self.assertTrue(
            service.enrich_reference_update(
                _event(target, [_embedding(), _second_embedding()])
            )
        )

        target.state = TargetState.LOST
        target.current_track_id = None
        service.update_runtime_state([target], 5)
        target.state = TargetState.ACTIVE
        target.current_track_id = 12
        target.last_recovery_frame = 6
        service.update_runtime_state([target], 6)
        self.assertFalse(
            service.enrich_reference_update(
                _event(target, [_embedding(), _second_embedding()])
            )
        )

    def test_missing_active_grace_frame_does_not_count_as_stable(self) -> None:
        service = GalleryPersistenceService(
            self.gallery,
            self.repository,
            enrichment_config=GalleryEnrichmentConfig(
                post_recovery_stable_frames=2
            ),
        )
        target = _target(1)
        service.enroll(target)
        target.state = TargetState.LOST
        target.current_track_id = None
        service.update_runtime_state([target], 1)

        target.state = TargetState.ACTIVE
        target.current_track_id = 11
        target.last_recovery_frame = 2
        target.missing_frames = 0
        service.update_runtime_state([target], 2)

        target.missing_frames = 1
        service.update_runtime_state([target], 3)
        target.missing_frames = 0
        service.update_runtime_state([target], 4)
        self.assertFalse(
            service.enrich_reference_update(
                _event(target, [_embedding(), _second_embedding()])
            )
        )

        service.update_runtime_state([target], 5)
        self.assertTrue(
            service.enrich_reference_update(
                _event(target, [_embedding(), _second_embedding()])
            )
        )

    def test_auto_recognized_target_requires_explicit_g_before_enrichment(self) -> None:
        target = _target(1)
        person = self.service.enroll(target)
        self.gallery.detach_session_target(target.target_id)

        # Simulate a new-process auto-recognized target mapping without G.
        auto_target = _target(2)
        self.gallery.attach_session_target(auto_target.target_id, person.person_id)
        event = _event(auto_target, [_embedding(), _second_embedding()])
        self.assertFalse(self.service.enrich_reference_update(event))
        self.assertEqual(len(self.repository.load_all()[0].reference_embeddings), 1)

        # A later explicit G is idempotent: it marks provenance but does not
        # save/insert a second GalleryPerson.
        enrolled = self.service.enroll(auto_target)
        self.assertEqual(enrolled.person_id, person.person_id)
        self.assertTrue(self.service.enrich_reference_update(event))
        self.assertEqual(
            tuple(p.person_id for p in self.repository.load_all()),
            (person.person_id,),
        )
        self.assertEqual(len(self.repository.load_all()[0].reference_embeddings), 2)

    def test_failed_enrichment_keeps_memory_and_database_old_features(self) -> None:
        target = _target(1)
        person = self.service.enroll(target)
        failing_service = GalleryPersistenceService(
            self.gallery,
            FailingUpdateRepository(self.repository),
        )
        # Establish explicit provenance in the service under test without a
        # second INSERT: the existing mapping is already present.
        failing_service.enroll(target)

        self.assertFalse(
            failing_service.enrich_reference_update(
                _event(target, [_embedding(), _second_embedding()])
            )
        )
        in_memory = self.gallery.get(person.person_id)
        persisted = self.repository.load_all()[0]
        assert in_memory is not None
        self.assertEqual(len(in_memory.reference_embeddings), 1)
        self.assertEqual(len(persisted.reference_embeddings), 1)


if __name__ == "__main__":
    unittest.main()
