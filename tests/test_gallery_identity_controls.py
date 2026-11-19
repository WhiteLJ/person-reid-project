from __future__ import annotations

from pathlib import Path
import re
import sqlite3
import unittest

import numpy as np

from src.database import GalleryRepository
from src.gallery import TargetGallery
from src.gallery_service import GalleryPersistenceService
from src.models import SessionTarget, TargetState
from src.vehicle_database import VehicleGalleryRepository
from src.vehicle_gallery import VehicleTargetGallery
from src.vehicle_gallery_service import VehicleGalleryPersistenceService


ROOT = Path(__file__).resolve().parents[1] / ".test_tmp"
ROOT.mkdir(parents=True, exist_ok=True)


def _person_embedding() -> np.ndarray:
    value = np.zeros((512,), dtype=np.float32)
    value[0] = 1.0
    return value


def _vehicle_embedding() -> np.ndarray:
    value = np.zeros((2048,), dtype=np.float32)
    value[0] = 1.0
    return value


def _session_target(target_id: int, embedding: np.ndarray) -> SessionTarget:
    return SessionTarget(
        target_id=target_id,
        current_track_id=target_id + 100,
        last_track_id=target_id + 100,
        state=TargetState.ACTIVE,
        reference_embeddings=[embedding.copy()],
        centroid=embedding.copy(),
    )


def _remove(path: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


class GalleryIdentityControlTests(unittest.TestCase):
    def test_manual_person_enrollment_defaults_enabled_and_persists(self) -> None:
        path = ROOT / "manual_person_enabled.db"
        _remove(path)
        repository = GalleryRepository(path)
        repository.initialize()
        gallery = TargetGallery()
        service = GalleryPersistenceService(gallery, repository)
        person = service.enroll(_session_target(1, _person_embedding()))

        self.assertTrue(person.recognition_enabled)
        restored = GalleryRepository(path).load_all()[0]
        self.assertTrue(restored.recognition_enabled)
        _remove(path)

    def test_manual_vehicle_enrollment_defaults_enabled_and_persists(self) -> None:
        path = ROOT / "manual_vehicle_enabled.db"
        _remove(path)
        repository = VehicleGalleryRepository(path)
        repository.initialize()
        gallery = VehicleTargetGallery()
        service = VehicleGalleryPersistenceService(gallery, repository)
        vehicle = service.enroll(_session_target(1, _vehicle_embedding()))

        self.assertTrue(vehicle.recognition_enabled)
        restored = VehicleGalleryRepository(path).load_all()[0]
        self.assertTrue(restored.recognition_enabled)
        _remove(path)

    def test_person_metadata_and_recognition_flag_round_trip(self) -> None:
        path = ROOT / "identity_controls_person.db"
        _remove(path)
        repository = GalleryRepository(path)
        repository.initialize()
        gallery = TargetGallery()
        service = GalleryPersistenceService(gallery, repository)
        person = service.import_external_identity(_person_embedding(), b"jpeg")

        self.assertFalse(person.recognition_enabled)
        self.assertRegex(person.created_at or "", r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
        updated = service.set_recognition_enabled(person.person_id, False)
        self.assertFalse(updated.recognition_enabled)
        restored = GalleryRepository(path).load_all()[0]
        self.assertFalse(restored.recognition_enabled)
        self.assertEqual(restored.created_at, person.created_at)
        _remove(path)

    def test_vehicle_metadata_and_recognition_flag_round_trip(self) -> None:
        path = ROOT / "identity_controls_vehicle.db"
        _remove(path)
        repository = VehicleGalleryRepository(path)
        repository.initialize()
        gallery = VehicleTargetGallery()
        service = VehicleGalleryPersistenceService(gallery, repository)
        vehicle = service.import_external_identity(_vehicle_embedding(), b"jpeg")

        self.assertFalse(vehicle.recognition_enabled)
        self.assertIsNone(vehicle.plate_number)
        self.assertRegex(vehicle.created_at or "", r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
        updated = service.set_recognition_enabled(vehicle.vehicle_id, False)
        self.assertFalse(updated.recognition_enabled)
        restored = VehicleGalleryRepository(path).load_all()[0]
        self.assertFalse(restored.recognition_enabled)
        self.assertEqual(restored.created_at, vehicle.created_at)
        _remove(path)

    def test_legacy_person_schema_gets_new_columns_without_recreating_table(self) -> None:
        path = ROOT / "identity_controls_legacy_person.db"
        _remove(path)
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            CREATE TABLE gallery_person(
                person_id INTEGER PRIMARY KEY,
                label TEXT NOT NULL,
                centroid BLOB NOT NULL,
                centroid_dim INTEGER NOT NULL,
                centroid_dtype TEXT NOT NULL
            );
            CREATE TABLE gallery_embedding(
                person_id INTEGER NOT NULL,
                embedding_index INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                embedding_dtype TEXT NOT NULL,
                PRIMARY KEY(person_id, embedding_index)
            );
            CREATE TABLE gallery_meta(key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            CREATE TABLE gallery_person_snapshot(person_id INTEGER PRIMARY KEY, image_jpeg BLOB NOT NULL);
            """
        )
        connection.commit()
        connection.close()

        GalleryRepository(path).initialize()
        connection = sqlite3.connect(path)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(gallery_person)")}
        defaults = connection.execute(
            "SELECT recognition_enabled, created_at FROM gallery_person"
        ).fetchall()
        connection.close()
        self.assertIn("recognition_enabled", columns)
        self.assertIn("created_at", columns)
        self.assertEqual(defaults, [])
        _remove(path)


if __name__ == "__main__":
    unittest.main()
