from __future__ import annotations

import unittest

import numpy as np

from src.models import SessionTarget, TargetState
from src.vehicle_gallery import (
    GalleryVehicle,
    VehicleTargetGallery,
    format_vehicle_id,
)


def _embedding(index: int = 0) -> np.ndarray:
    value = np.zeros((2048,), dtype=np.float32)
    value[index] = 1.0
    return value


def _target(target_id: int, reference_count: int = 1) -> SessionTarget:
    references = [_embedding(index) for index in range(reference_count)]
    centroid = np.mean(np.stack(references), axis=0).astype(np.float32)
    centroid /= np.linalg.norm(centroid)
    return SessionTarget(
        target_id=target_id,
        current_track_id=target_id + 10,
        last_track_id=target_id + 10,
        state=TargetState.ACTIVE,
        reference_embeddings=references,
        centroid=centroid,
    )


class VehicleGalleryTests(unittest.TestCase):
    def test_vehicle_id_format(self) -> None:
        self.assertEqual(format_vehicle_id(1), "V001")
        self.assertEqual(format_vehicle_id(12), "V012")

    def test_enroll_snapshots_2048d_features(self) -> None:
        target = _target(1, reference_count=2)
        gallery = VehicleTargetGallery()

        vehicle = gallery.enroll(target)
        target.reference_embeddings[0][0] = 0.0
        target.centroid[0] = 0.0
        vehicle.reference_embeddings[0][1] = 0.0

        self.assertEqual(vehicle.vehicle_id, 1)
        self.assertEqual(vehicle.label, "Target V001")
        self.assertEqual(len(vehicle.reference_embeddings), 2)
        self.assertEqual(vehicle.reference_embeddings[0].shape, (2048,))
        self.assertEqual(vehicle.reference_embeddings[0].dtype, np.float32)
        self.assertAlmostEqual(
            float(np.linalg.norm(vehicle.centroid)), 1.0, places=5
        )
        self.assertNotEqual(float(vehicle.reference_embeddings[0][0]), 0.0)
        self.assertEqual(float(target.reference_embeddings[0][0]), 0.0)

    def test_repeated_enrollment_is_idempotent(self) -> None:
        gallery = VehicleTargetGallery()
        target = _target(1)

        first = gallery.enroll(target)
        second = gallery.enroll(target)

        self.assertIs(first, second)
        self.assertEqual(tuple(v.vehicle_id for v in gallery.all_vehicles()), (1,))

    def test_two_targets_receive_distinct_ids(self) -> None:
        gallery = VehicleTargetGallery()

        first = gallery.enroll(_target(1))
        second = gallery.enroll(_target(2))

        self.assertEqual((first.vehicle_id, second.vehicle_id), (1, 2))

    def test_attach_is_strictly_one_to_one(self) -> None:
        gallery = VehicleTargetGallery()
        first = gallery.enroll(_target(1))
        second = gallery.enroll(_target(2))
        gallery.detach_session_target(1)
        gallery.detach_session_target(2)

        self.assertTrue(gallery.attach_session_target(10, first.vehicle_id))
        self.assertTrue(gallery.attach_session_target(10, first.vehicle_id))
        with self.assertRaises(ValueError):
            gallery.attach_session_target(10, second.vehicle_id)
        with self.assertRaises(ValueError):
            gallery.attach_session_target(11, first.vehicle_id)

    def test_remove_and_clear_detach_mappings_without_target_state(self) -> None:
        gallery = VehicleTargetGallery()
        first = gallery.enroll(_target(1))
        second = gallery.enroll(_target(2))
        gallery.detach_all_session_targets()
        gallery.attach_session_target(101, first.vehicle_id)
        gallery.attach_session_target(102, second.vehicle_id)

        self.assertTrue(gallery.remove(first.vehicle_id))
        self.assertIsNone(gallery.vehicle_for_session_target(101))
        self.assertIsNotNone(gallery.get(second.vehicle_id))

        gallery.clear()
        self.assertEqual(gallery.all_vehicles(), ())
        self.assertIsNone(gallery.vehicle_for_session_target(102))

    def test_allocator_is_not_reset_by_delete_or_clear(self) -> None:
        gallery = VehicleTargetGallery()
        first = gallery.enroll(_target(1))
        gallery.remove(first.vehicle_id)
        second = gallery.enroll(_target(2))
        gallery.clear()
        third = gallery.enroll(_target(3))

        self.assertEqual(second.vehicle_id, 2)
        self.assertEqual(third.vehicle_id, 3)

    def test_restore_clears_old_session_mappings(self) -> None:
        gallery = VehicleTargetGallery()
        first = gallery.enroll(_target(1))
        gallery.detach_session_target(1)
        gallery.attach_session_target(99, first.vehicle_id)

        gallery.restore_vehicles(
            [
                GalleryVehicle(
                    vehicle_id=first.vehicle_id,
                    label=first.label,
                    reference_embeddings=[item.copy() for item in first.reference_embeddings],
                    centroid=first.centroid.copy(),
                )
            ],
            next_vehicle_id=2,
        )

        self.assertIsNone(gallery.vehicle_for_session_target(99))
        self.assertEqual(gallery.get(1).vehicle_id, 1)  # type: ignore[union-attr]

    def test_invalid_dimension_is_rejected(self) -> None:
        bad = SessionTarget(
            target_id=1,
            current_track_id=1,
            last_track_id=1,
            state=TargetState.ACTIVE,
            reference_embeddings=[np.ones((512,), dtype=np.float32)],
            centroid=np.ones((512,), dtype=np.float32),
        )

        with self.assertRaises(ValueError):
            VehicleTargetGallery().enroll(bad)


if __name__ == "__main__":
    unittest.main()
