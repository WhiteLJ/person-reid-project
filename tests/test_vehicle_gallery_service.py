from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src.models import SessionTarget, TargetState
from src.vehicle_database import VehicleGalleryRepository, VehicleRepositoryError
from src.vehicle_gallery import VehicleTargetGallery
from src.vehicle_gallery_service import VehicleGalleryPersistenceService


_TEST_TEMP_PARENT = Path(__file__).resolve().parents[1] / ".test_tmp"
_TEST_TEMP_PARENT.mkdir(parents=True, exist_ok=True)


def _remove_database_files(path: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _embedding(index: int = 0) -> np.ndarray:
    value = np.zeros((2048,), dtype=np.float32)
    value[index] = 1.0
    return value


def _target(target_id: int) -> SessionTarget:
    embedding = _embedding(target_id % 4)
    return SessionTarget(
        target_id=target_id,
        current_track_id=target_id + 10,
        last_track_id=target_id + 10,
        state=TargetState.ACTIVE,
        reference_embeddings=[embedding.copy()],
        centroid=embedding.copy(),
    )


class _FailingRepository:
    def save_vehicle(self, vehicle) -> None:
        del vehicle
        raise VehicleRepositoryError("forced save failure")


class VehicleGalleryPersistenceServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = _TEST_TEMP_PARENT / "vehicle_gallery_service_tests.db"
        _remove_database_files(self.path)
        self.repository = VehicleGalleryRepository(self.path)
        self.gallery = VehicleTargetGallery()
        self.service = VehicleGalleryPersistenceService(self.gallery, self.repository)

    def tearDown(self) -> None:
        _remove_database_files(self.path)

    def test_enroll_persists_and_duplicate_is_idempotent(self) -> None:
        target = _target(1)
        first = self.service.enroll(target)
        second = self.service.enroll(target)

        self.assertIs(first, second)
        self.assertEqual(tuple(v.vehicle_id for v in self.repository.load_all()), (1,))

    def test_failed_new_enrollment_rolls_back_memory_only(self) -> None:
        gallery = VehicleTargetGallery()
        service = VehicleGalleryPersistenceService(gallery, _FailingRepository())

        with self.assertRaises(VehicleRepositoryError):
            service.enroll(_target(1))

        self.assertEqual(gallery.all_vehicles(), ())
        self.assertIsNone(gallery.vehicle_for_session_target(1))

    def test_remove_detaches_mapping_but_does_not_touch_person_database(self) -> None:
        target = _target(1)
        vehicle = self.service.enroll(target)
        self.assertTrue(self.service.remove(vehicle.vehicle_id))
        self.assertIsNone(self.gallery.get(vehicle.vehicle_id))
        self.assertIsNone(self.gallery.vehicle_for_session_target(target.target_id))
        self.assertEqual(self.repository.load_all(), ())

    def test_deleted_vehicle_can_be_reenrolled_with_a_new_id(self) -> None:
        first = self.service.enroll(_target(1))
        self.assertTrue(self.service.remove(first.vehicle_id))

        second = self.service.enroll(_target(2))

        self.assertEqual(first.vehicle_id, 1)
        self.assertEqual(second.vehicle_id, 2)
        self.assertEqual(
            tuple(v.vehicle_id for v in self.repository.load_all()),
            (2,),
        )

    def test_r_and_c_detach_session_targets_but_preserve_gallery(self) -> None:
        first = self.service.enroll(_target(1))
        second = self.service.enroll(_target(2))
        self.service.detach_session_target(1)
        self.assertIsNotNone(self.gallery.get(first.vehicle_id))
        self.assertIsNotNone(self.gallery.vehicle_for_session_target(2))

        self.service.detach_all_session_targets()
        self.assertEqual(
            tuple(v.vehicle_id for v in self.repository.load_all()),
            (first.vehicle_id, second.vehicle_id),
        )
        self.assertIsNone(self.gallery.vehicle_for_session_target(2))

    def test_load_never_restores_old_session_mapping(self) -> None:
        target = _target(1)
        vehicle = self.service.enroll(target)
        self.service.detach_session_target(target.target_id)
        self.assertIsNotNone(vehicle)

        restarted_gallery = VehicleTargetGallery()
        restarted_service = VehicleGalleryPersistenceService(
            restarted_gallery,
            VehicleGalleryRepository(self.path),
        )
        restarted_service.load()

        self.assertIsNotNone(restarted_gallery.get(vehicle.vehicle_id))
        self.assertIsNone(restarted_gallery.vehicle_for_session_target(target.target_id))

    def test_vehicle_database_is_independent_of_person_database(self) -> None:
        self.service.enroll(_target(1))
        connection = self.repository._connect()
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            connection.close()
        self.assertIn("gallery_vehicle", tables)
        self.assertNotIn("gallery_person", tables)


if __name__ == "__main__":
    unittest.main()
