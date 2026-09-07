from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src.database import GalleryRepository, RepositoryError
from src.gallery import GalleryPerson, TargetGallery
from src.gallery_service import GalleryPersistenceService
from src.models import SessionTarget, TargetState


_TEST_TEMP_PARENT = Path(__file__).resolve().parents[1] / ".test_tmp"
_TEST_TEMP_PARENT.mkdir(parents=True, exist_ok=True)


def _remove_database_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)


def _embedding(index: int) -> np.ndarray:
    value = np.zeros((512,), dtype=np.float32)
    value[index] = 1.0
    return value


def _person(person_id: int, reference_count: int = 1) -> GalleryPerson:
    references = [_embedding(index) for index in range(reference_count)]
    centroid = _embedding(0)
    return GalleryPerson(
        person_id=person_id,
        label=f"Person {person_id}",
        reference_embeddings=references,
        centroid=centroid,
    )


def _session_target(target_id: int) -> SessionTarget:
    embedding = _embedding(0)
    return SessionTarget(
        target_id=target_id,
        current_track_id=target_id + 100,
        last_track_id=target_id + 100,
        state=TargetState.ACTIVE,
        reference_embeddings=[embedding.copy()],
        centroid=embedding.copy(),
    )


class GalleryRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database_path = _TEST_TEMP_PARENT / "database_tests.db"
        _remove_database_files(self.database_path)
        self.repository = GalleryRepository(self.database_path)

    def tearDown(self) -> None:
        _remove_database_files(self.database_path)

    def test_initialize_creates_parent_directory_and_empty_database(self) -> None:
        nested_path = _TEST_TEMP_PARENT / "auto_created_parent" / "gallery.db"
        _remove_database_files(nested_path)
        if nested_path.parent.exists():
            nested_path.parent.rmdir()
        repository = GalleryRepository(nested_path)
        self.assertFalse(nested_path.parent.exists())

        repository.initialize()

        self.assertTrue(nested_path.parent.is_dir())
        self.assertTrue(nested_path.is_file())
        self.assertEqual(repository.load_all(), ())
        self.assertEqual(repository.load_next_person_id(), 1)

    def test_every_connection_enables_foreign_keys(self) -> None:
        self.repository.initialize()

        connection = self.repository._connect()
        try:
            enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        finally:
            connection.close()

        self.assertEqual(enabled, 1)

    def test_save_load_and_reopen_round_trip_preserves_data(self) -> None:
        person = _person(1, reference_count=2)
        self.repository.save_person(person)

        reopened = GalleryRepository(self.database_path)
        loaded = reopened.load_all()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].person_id, 1)
        self.assertEqual(loaded[0].label, "Person 1")
        self.assertEqual(loaded[0].centroid.shape, (512,))
        self.assertEqual(loaded[0].centroid.dtype, np.float32)
        self.assertEqual(len(loaded[0].reference_embeddings), 2)
        self.assertEqual(loaded[0].reference_embeddings[1].dtype, np.float32)
        self.assertTrue(
            np.allclose(loaded[0].reference_embeddings[1], person.reference_embeddings[1])
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(loaded[0].centroid)), 1.0, places=6
        )

    def test_loaded_arrays_are_independent(self) -> None:
        self.repository.save_person(_person(1, reference_count=2))

        first = self.repository.load_all()[0]
        second = self.repository.load_all()[0]
        first.reference_embeddings[0][0] = 0.0
        first.centroid[0] = 0.0

        self.assertNotEqual(float(second.reference_embeddings[0][0]), 0.0)
        self.assertNotEqual(float(second.centroid[0]), 0.0)

    def test_person_id_and_next_id_survive_restart_without_reuse(self) -> None:
        self.repository.save_person(_person(1))
        self.repository.save_person(_person(2))
        reopened = GalleryRepository(self.database_path)

        self.assertEqual(reopened.load_next_person_id(), 3)
        reopened.delete_person(2)
        self.assertEqual(reopened.load_next_person_id(), 3)
        with self.assertRaises(RepositoryError):
            reopened.save_person(_person(2))
        reopened.save_person(_person(3))
        self.assertEqual(
            tuple(person.person_id for person in reopened.load_all()), (1, 3)
        )

    def test_metadata_is_repaired_above_current_max_person_id(self) -> None:
        self.repository.save_person(_person(5))
        connection = self.repository._connect()
        try:
            connection.execute(
                "UPDATE gallery_meta SET value = 1 WHERE key = 'next_person_id'"
            )
            connection.commit()
        finally:
            connection.close()

        self.assertEqual(self.repository.load_next_person_id(), 6)

    def test_foreign_key_cascade_deletes_embeddings(self) -> None:
        self.repository.save_person(_person(1, reference_count=2))
        connection = self.repository._connect()
        try:
            count_before = connection.execute(
                "SELECT COUNT(*) FROM gallery_embedding WHERE person_id = 1"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(count_before, 2)

        self.assertTrue(self.repository.delete_person(1))
        connection = self.repository._connect()
        try:
            count_after = connection.execute(
                "SELECT COUNT(*) FROM gallery_embedding WHERE person_id = 1"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(count_after, 0)

    def test_clear_removes_people_and_embeddings(self) -> None:
        self.repository.save_person(_person(1, reference_count=2))
        self.repository.save_person(_person(2))

        self.repository.clear()

        self.assertEqual(self.repository.load_all(), ())
        connection = self.repository._connect()
        try:
            embedding_count = connection.execute(
                "SELECT COUNT(*) FROM gallery_embedding"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(embedding_count, 0)

    def test_save_person_embeddings_and_next_id_are_one_transaction(self) -> None:
        self.repository.initialize()
        connection = self.repository._connect()
        try:
            connection.execute(
                """
                CREATE TRIGGER fail_second_embedding
                BEFORE INSERT ON gallery_embedding
                WHEN NEW.embedding_index = 1
                BEGIN
                    SELECT RAISE(ABORT, 'forced test failure');
                END
                """
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaises(RepositoryError):
            self.repository.save_person(_person(1, reference_count=2))

        connection = self.repository._connect()
        try:
            person_count = connection.execute(
                "SELECT COUNT(*) FROM gallery_person"
            ).fetchone()[0]
            embedding_count = connection.execute(
                "SELECT COUNT(*) FROM gallery_embedding"
            ).fetchone()[0]
            next_id = connection.execute(
                "SELECT value FROM gallery_meta WHERE key = 'next_person_id'"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual((person_count, embedding_count, next_id), (0, 0, 1))

    def test_update_features_preserves_person_label_and_next_id(self) -> None:
        self.repository.save_person(_person(1))
        self.repository.update_person_features(
            1,
            [_embedding(0), _embedding(1), _embedding(2)],
            _embedding(2),
        )

        reopened = GalleryRepository(self.database_path)
        people = reopened.load_all()
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0].person_id, 1)
        self.assertEqual(people[0].label, "Person 1")
        self.assertEqual(len(people[0].reference_embeddings), 3)
        self.assertTrue(np.array_equal(people[0].centroid, _embedding(2)))
        self.assertEqual(reopened.load_next_person_id(), 2)

    def test_update_unknown_person_raises_clear_error(self) -> None:
        with self.assertRaises(RepositoryError):
            self.repository.update_person_features(99, [_embedding(0)], _embedding(0))

    def test_update_features_is_atomic_on_embedding_failure(self) -> None:
        self.repository.save_person(_person(1))
        connection = self.repository._connect()
        try:
            connection.execute(
                """
                CREATE TRIGGER fail_update_second_embedding
                BEFORE INSERT ON gallery_embedding
                WHEN NEW.person_id = 1 AND NEW.embedding_index = 1
                BEGIN
                    SELECT RAISE(ABORT, 'forced update failure');
                END
                """
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaises(RepositoryError):
            self.repository.update_person_features(
                1,
                [_embedding(0), _embedding(1)],
                _embedding(1),
            )

        persisted = self.repository.load_all()[0]
        self.assertEqual(len(persisted.reference_embeddings), 1)
        self.assertTrue(np.array_equal(persisted.reference_embeddings[0], _embedding(0)))
        self.assertTrue(np.array_equal(persisted.centroid, _embedding(0)))

    def test_schema_does_not_persist_session_or_track_state(self) -> None:
        self.repository.initialize()
        connection = self.repository._connect()
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            person_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(gallery_person)")
            }
        finally:
            connection.close()

        self.assertEqual(
            tables,
            {"gallery_person", "gallery_embedding", "gallery_meta"},
        )
        self.assertNotIn("track_id", person_columns)
        self.assertNotIn("current_track_id", person_columns)
        self.assertNotIn("session_target_id", person_columns)

    def test_invalid_embedding_dimension_is_rejected(self) -> None:
        invalid = GalleryPerson(
            person_id=1,
            label="Invalid",
            reference_embeddings=[np.ones((2,), dtype=np.float32)],
            centroid=np.ones((2,), dtype=np.float32),
        )

        with self.assertRaises(RepositoryError):
            self.repository.save_person(invalid)

    def test_corrupt_dimension_metadata_raises_clear_error(self) -> None:
        self.repository.save_person(_person(1))
        connection = self.repository._connect()
        try:
            connection.execute("PRAGMA ignore_check_constraints = ON")
            connection.execute(
                "UPDATE gallery_person SET centroid_dim = 511 WHERE person_id = 1"
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaises(RepositoryError):
            self.repository.load_all()

    def test_loaded_people_restore_without_old_session_mapping(self) -> None:
        self.repository.save_person(_person(1))
        gallery = TargetGallery()
        service = GalleryPersistenceService(gallery, self.repository)

        loaded = service.load()

        self.assertEqual(tuple(person.person_id for person in loaded), (1,))
        self.assertIsNone(gallery.person_for_session_target(101))
        restored_target = _session_target(101)
        restored_person = service.enroll(restored_target)
        self.assertEqual(restored_person.person_id, 2)


if __name__ == "__main__":
    unittest.main()
