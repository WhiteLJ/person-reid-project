from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src.database import GalleryRepository, RepositoryError
from src.gallery import TargetGallery
from src.gallery_service import GalleryPersistenceService
from src.models import SessionTarget, TargetState


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


if __name__ == "__main__":
    unittest.main()
