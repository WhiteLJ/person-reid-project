from __future__ import annotations

from pathlib import Path
import sqlite3
import unittest

import numpy as np

from src.vehicle_database import VehicleGalleryRepository, VehicleRepositoryError
from src.vehicle_gallery import GalleryVehicle


_TEST_TEMP_PARENT = Path(__file__).resolve().parents[1] / ".test_tmp"
_TEST_TEMP_PARENT.mkdir(parents=True, exist_ok=True)


def _remove_database_files(path: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _embedding(index: int = 0) -> np.ndarray:
    value = np.zeros((2048,), dtype=np.float32)
    value[index] = 1.0
    return value


def _vehicle(vehicle_id: int, reference_count: int = 1) -> GalleryVehicle:
    references = [_embedding(index) for index in range(reference_count)]
    centroid = np.mean(np.stack(references), axis=0).astype(np.float32)
    centroid /= np.linalg.norm(centroid)
    return GalleryVehicle(
        vehicle_id=vehicle_id,
        label=f"Target V{vehicle_id:03d}",
        reference_embeddings=references,
        centroid=centroid,
    )


class VehicleGalleryRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = _TEST_TEMP_PARENT / "vehicle_database_tests.db"
        _remove_database_files(self.path)
        self.repository = VehicleGalleryRepository(self.path)

    def tearDown(self) -> None:
        _remove_database_files(self.path)

    def test_initialize_creates_parent_and_empty_database(self) -> None:
        path = _TEST_TEMP_PARENT / "vehicle_nested" / "gallery.db"
        _remove_database_files(path)
        if path.parent.exists():
            path.parent.rmdir()
        repository = VehicleGalleryRepository(path)

        repository.initialize()

        self.assertTrue(path.is_file())
        self.assertEqual(repository.load_all(), ())
        self.assertEqual(repository.load_next_vehicle_id(), 1)

    def test_foreign_keys_are_enabled_on_each_connection(self) -> None:
        self.repository.initialize()
        connection = self.repository._connect()
        try:
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        finally:
            connection.close()

    def test_2048d_round_trip_after_reopen(self) -> None:
        vehicle = _vehicle(1, reference_count=3)
        self.repository.save_vehicle(vehicle)

        loaded = VehicleGalleryRepository(self.path).load_all()[0]

        self.assertEqual(loaded.vehicle_id, 1)
        self.assertEqual(loaded.label, vehicle.label)
        self.assertEqual(loaded.centroid.shape, (2048,))
        self.assertEqual(loaded.centroid.dtype, np.float32)
        self.assertEqual(len(loaded.reference_embeddings), 3)
        for expected, actual in zip(vehicle.reference_embeddings, loaded.reference_embeddings):
            self.assertTrue(np.array_equal(expected, actual))
        self.assertTrue(np.array_equal(vehicle.centroid, loaded.centroid))

    def test_reference_order_and_cascade_delete(self) -> None:
        self.repository.save_vehicle(_vehicle(1, reference_count=3))
        connection = self.repository._connect()
        try:
            count = connection.execute(
                "SELECT COUNT(*) FROM gallery_vehicle_embedding WHERE vehicle_id=1"
            ).fetchone()[0]
            indexes = [
                row[0]
                for row in connection.execute(
                    "SELECT embedding_index FROM gallery_vehicle_embedding "
                    "WHERE vehicle_id=1 ORDER BY embedding_index"
                ).fetchall()
            ]
        finally:
            connection.close()
        self.assertEqual(count, 3)
        self.assertEqual(indexes, [0, 1, 2])

        self.assertTrue(self.repository.delete_vehicle(1))
        connection = self.repository._connect()
        try:
            remaining = connection.execute(
                "SELECT COUNT(*) FROM gallery_vehicle_embedding"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(remaining, 0)

    def test_id_allocator_is_monotonic_across_delete_clear_and_reopen(self) -> None:
        self.repository.save_vehicle(_vehicle(1))
        self.repository.save_vehicle(_vehicle(2))
        self.repository.delete_vehicle(1)
        self.assertEqual(self.repository.load_next_vehicle_id(), 3)
        self.repository.clear()
        self.assertEqual(self.repository.load_next_vehicle_id(), 3)
        reopened = VehicleGalleryRepository(self.path)
        self.assertEqual(reopened.load_next_vehicle_id(), 3)
        with self.assertRaises(VehicleRepositoryError):
            reopened.save_vehicle(_vehicle(1))
        reopened.save_vehicle(_vehicle(3))
        self.assertEqual(tuple(v.vehicle_id for v in reopened.load_all()), (3,))

    def test_metadata_is_repaired_above_current_max(self) -> None:
        self.repository.save_vehicle(_vehicle(5))
        connection = self.repository._connect()
        try:
            connection.execute(
                "UPDATE gallery_vehicle_meta SET value=1 WHERE key='next_vehicle_id'"
            )
            connection.commit()
        finally:
            connection.close()

        self.assertEqual(self.repository.load_next_vehicle_id(), 6)

    def test_invalid_dimension_dtype_and_nonfinite_are_rejected(self) -> None:
        bad_dimension = _vehicle(1)
        bad_dimension.reference_embeddings = [np.ones((512,), dtype=np.float32)]
        with self.assertRaises(VehicleRepositoryError):
            self.repository.save_vehicle(bad_dimension)

        bad_dtype = _vehicle(1)
        bad_dtype.reference_embeddings = [np.ones((2048,), dtype=np.float64)]
        with self.assertRaises(VehicleRepositoryError):
            self.repository.save_vehicle(bad_dtype)

        bad_nonfinite = _vehicle(1)
        bad_nonfinite.reference_embeddings[0][3] = np.nan
        with self.assertRaises(VehicleRepositoryError):
            self.repository.save_vehicle(bad_nonfinite)

    def test_save_transaction_rolls_back_vehicle_embeddings_and_allocator(self) -> None:
        self.repository.initialize()
        connection = self.repository._connect()
        try:
            connection.execute(
                """
                CREATE TRIGGER fail_second_vehicle_embedding
                BEFORE INSERT ON gallery_vehicle_embedding
                WHEN NEW.embedding_index = 1
                BEGIN
                    SELECT RAISE(ABORT, 'forced test failure');
                END
                """
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaises(VehicleRepositoryError):
            self.repository.save_vehicle(_vehicle(1, reference_count=2))

        connection = self.repository._connect()
        try:
            counts = (
                connection.execute("SELECT COUNT(*) FROM gallery_vehicle").fetchone()[0],
                connection.execute(
                    "SELECT COUNT(*) FROM gallery_vehicle_embedding"
                ).fetchone()[0],
                connection.execute(
                    "SELECT value FROM gallery_vehicle_meta "
                    "WHERE key='next_vehicle_id'"
                ).fetchone()[0],
            )
        finally:
            connection.close()
        self.assertEqual(counts, (0, 0, 1))

    def test_update_features_is_atomic_and_preserves_label_and_next_id(self) -> None:
        self.repository.save_vehicle(_vehicle(1, reference_count=1))
        before = self.repository.load_vehicle(1)
        assert before is not None
        next_before = self.repository.load_next_vehicle_id()

        self.repository.update_vehicle_features(
            1,
            [_embedding(0), _embedding(1)],
            np.mean(np.stack([_embedding(0), _embedding(1)]), axis=0).astype(
                np.float32
            )
            / np.sqrt(0.5),
        )
        after = self.repository.load_vehicle(1)
        assert after is not None
        self.assertEqual(after.label, before.label)
        self.assertEqual(len(after.reference_embeddings), 2)
        self.assertEqual(self.repository.load_next_vehicle_id(), next_before)

    def test_vehicle_schema_does_not_create_person_tables(self) -> None:
        self.repository.initialize()
        connection = sqlite3.connect(self.path)
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            connection.close()
        self.assertNotIn("gallery_person", tables)
        self.assertIn("gallery_vehicle", tables)


if __name__ == "__main__":
    unittest.main()
