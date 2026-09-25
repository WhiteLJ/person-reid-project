from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src.config import (
    GalleryRecognitionConfig,
    ReIDRecoveryConfig,
    VehicleGalleryEnrichmentConfig,
    VehicleReIDConfig,
    VehicleReIDQualityConfig,
)
from src.models import SessionTarget, TargetState, Track
from src.reid_frame_cache import ReIDFrameCache
from src.target_manager import TargetManager
from src.vehicle_database import VehicleGalleryRepository
from src.vehicle_gallery import GalleryVehicle, VehicleTargetGallery
from src.vehicle_gallery_recognition import VehicleGalleryRecognitionCoordinator
from src.vehicle_gallery_service import VehicleGalleryPersistenceService


def _unit(index: int = 0) -> np.ndarray:
    value = np.zeros((2048,), dtype=np.float32)
    value[index] = 1.0
    return value


def _vehicle(vehicle_id: int, index: int = 0) -> GalleryVehicle:
    reference = _unit(index)
    return GalleryVehicle(
        vehicle_id=vehicle_id,
        label=f"Target V{vehicle_id:03d}",
        reference_embeddings=[reference.copy()],
        centroid=reference.copy(),
    )


def _track(track_id: int, x1: int = 10) -> Track:
    return Track(track_id, (x1, 5, x1 + 40, 110), 0.95, 2)


def _vehicle_reid_config() -> VehicleReIDConfig:
    return VehicleReIDConfig(
        enabled=True,
        model_name="sbs_R50-ibn",
        weight=Path("unused.pth"),
        config=Path("unused.yml"),
        device="cpu",
        image_height=256,
        image_width=256,
    )


def _recovery_config() -> ReIDRecoveryConfig:
    return ReIDRecoveryConfig(
        lost_grace_frames=1,
        reference_update_interval_frames=15,
        recovery_interval_frames=5,
        max_reference_embeddings=8,
        recovery_threshold=0.60,
        recovery_margin=0.08,
        reference_update_threshold=0.55,
        recovery_reference_support_threshold=0.55,
        recovery_reference_support_top_k=3,
        recovery_min_track_age_frames=1,
        recovery_confirmation_hits=1,
        recovery_pending_max_age_frames=60,
        recovery_candidates_per_frame=1,
    )


def _recognition_config(**overrides: object) -> GalleryRecognitionConfig:
    values: dict[str, object] = {
        "enabled": True,
        "recognition_interval_frames": 1,
        "min_track_age_frames": 1,
        "recognition_threshold": 0.60,
        "recognition_margin": 0.08,
        "confirmation_hits": 1,
    }
    values.update(overrides)
    return GalleryRecognitionConfig(**values)  # type: ignore[arg-type]


def _quality_config() -> VehicleReIDQualityConfig:
    return VehicleReIDQualityConfig(
        min_track_confidence=0.1,
        max_edge_truncation_ratio=0.3,
        max_vehicle_overlap_ratio=0.6,
        min_frame_edge_margin_ratio=0.0,
        min_crop_width=10,
        min_crop_height=10,
    )


class _FakeVehicleReID:
    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = [np.asarray(output, dtype=np.float32) for output in outputs]
        self.batch_calls = 0
        self.batch_sizes: list[int] = []

    def extract_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        self.batch_calls += 1
        self.batch_sizes.append(len(crops))
        if not self.outputs:
            raise AssertionError("fake extractor ran out of outputs")
        output = self.outputs.pop(0)
        if output.ndim == 1:
            return np.repeat(output[None, :], len(crops), axis=0)
        return output.copy()

    def extract(self, crop: np.ndarray) -> np.ndarray:
        return self.extract_batch([crop])[0]


def _gallery(*vehicles: GalleryVehicle) -> VehicleTargetGallery:
    gallery = VehicleTargetGallery()
    gallery.restore_vehicles(vehicles)
    return gallery


def _coordinator(
    manager: TargetManager,
    gallery: VehicleTargetGallery,
    extractor: _FakeVehicleReID,
    *,
    cache: ReIDFrameCache | None = None,
    **overrides: object,
) -> VehicleGalleryRecognitionCoordinator:
    return VehicleGalleryRecognitionCoordinator(
        target_manager=manager,
        gallery=gallery,
        reid_extractor=extractor,
        reid_config=_vehicle_reid_config(),
        recognition_config=_recognition_config(**overrides),
        recovery_config=_recovery_config(),
        quality_config=_quality_config(),
        vehicle_class_ids=(2,),
        embedding_cache=cache,
    )


class VehicleGalleryRecognitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((120, 160, 3), dtype=np.uint8)

    def test_loaded_vehicle_is_recognized_as_new_target_with_live_only_bank(self) -> None:
        live = _unit(0)
        gallery = _gallery(_vehicle(1, 0))
        manager = TargetManager()
        extractor = _FakeVehicleReID([live])
        coordinator = _coordinator(manager, gallery, extractor)

        matches = coordinator.process_frame(self.frame, [_track(27)], 0)

        self.assertEqual(len(matches), 1)
        target = manager.target_for_track(27)
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.target_id, 1)
        self.assertEqual(len(target.reference_embeddings), 1)
        np.testing.assert_array_equal(target.reference_embeddings[0], live)
        self.assertIs(gallery.vehicle_for_session_target(1), gallery.get(1))
        self.assertEqual(len(gallery.get(1).reference_embeddings), 1)  # type: ignore[union-attr]

    def test_confirmation_requires_same_vehicle_and_track(self) -> None:
        live = _unit(0)
        extractor = _FakeVehicleReID([live, live, live])
        gallery = _gallery(_vehicle(1, 0))
        manager = TargetManager()
        coordinator = _coordinator(
            manager,
            gallery,
            extractor,
            confirmation_hits=2,
        )

        self.assertEqual(coordinator.process_frame(self.frame, [_track(27)], 0), [])
        self.assertEqual(coordinator.pending, {(1, 27): 1})
        self.assertEqual(coordinator.process_frame(self.frame, [_track(28)], 1), [])
        self.assertEqual(coordinator.pending, {(1, 28): 1})
        matches = coordinator.process_frame(self.frame, [_track(28)], 2)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].person_id, 1)
        self.assertEqual(matches[0].candidate.track.track_id, 28)

    def test_one_to_one_and_occupied_identity(self) -> None:
        extractor = _FakeVehicleReID(
            [np.asarray((_unit(0), _unit(1)), dtype=np.float32)]
        )
        gallery = _gallery(_vehicle(1, 0))
        manager = TargetManager()
        coordinator = _coordinator(manager, gallery, extractor)

        matches = coordinator.process_frame(
            self.frame,
            [_track(27), _track(28, 70)],
            0,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].candidate.track.track_id, 27)
        self.assertEqual(extractor.batch_sizes, [2])
        self.assertEqual(coordinator.process_frame(self.frame, [_track(28, 70)], 1), [])
        self.assertEqual(extractor.batch_sizes, [2])

    def test_recovery_protected_track_is_not_recognized(self) -> None:
        extractor = _FakeVehicleReID([_unit(0)])
        gallery = _gallery(_vehicle(1, 0))
        coordinator = _coordinator(TargetManager(), gallery, extractor)

        self.assertEqual(
            coordinator.process_frame(
                self.frame,
                [_track(27)],
                0,
                protected_track_ids=(27,),
            ),
            [],
        )
        self.assertEqual(extractor.batch_calls, 0)

    def test_unclaimed_recovery_candidate_remains_eligible(self) -> None:
        extractor = _FakeVehicleReID([_unit(0)])
        gallery = _gallery(_vehicle(1, 0))
        coordinator = _coordinator(TargetManager(), gallery, extractor)

        matches = coordinator.process_frame(self.frame, [_track(27)], 0)

        self.assertEqual(len(matches), 1)
        self.assertEqual(extractor.batch_calls, 1)

    def test_quality_gate_rejects_edge_candidate(self) -> None:
        extractor = _FakeVehicleReID([_unit(0)])
        gallery = _gallery(_vehicle(1, 0))
        coordinator = _coordinator(TargetManager(), gallery, extractor)
        edge_track = Track(27, (0, 5, 40, 110), 0.95, 2)

        self.assertEqual(coordinator.process_frame(self.frame, [edge_track], 0), [])
        self.assertEqual(extractor.batch_calls, 0)
        self.assertEqual(coordinator.quality_rejected_count, 1)

    def test_cache_reuses_same_frame_embedding(self) -> None:
        cache = ReIDFrameCache()
        cache.begin_frame(0)
        cache.put(27, _unit(0), 0)
        extractor = _FakeVehicleReID([])
        coordinator = _coordinator(
            TargetManager(),
            _gallery(_vehicle(1, 0)),
            extractor,
            cache=cache,
        )

        matches = coordinator.process_frame(self.frame, [_track(27)], 0)

        self.assertEqual(len(matches), 1)
        self.assertEqual(extractor.batch_calls, 0)


class VehicleGalleryEnrichmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).resolve().parents[1] / ".test_tmp" / "vehicle_pc6_enrichment.db"
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{self.path}{suffix}").unlink(missing_ok=True)

    def tearDown(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{self.path}{suffix}").unlink(missing_ok=True)

    def test_auto_recognized_vehicle_enriches_only_after_stable_cooldown(self) -> None:
        repository = VehicleGalleryRepository(self.path)
        initial = _vehicle(1, 0)
        repository.save_vehicle(initial)
        gallery = VehicleTargetGallery()
        service = VehicleGalleryPersistenceService(
            gallery,
            repository,
            VehicleGalleryEnrichmentConfig(post_recovery_stable_frames=2),
        )
        service.load()
        target = SessionTarget(
            target_id=1,
            current_track_id=27,
            last_track_id=27,
            state=TargetState.ACTIVE,
            reference_embeddings=[_unit(0)],
            centroid=_unit(0),
        )
        gallery.attach_session_target(target.target_id, 1)
        service.mark_auto_recognized(target.target_id)
        event = _reference_event(target, _unit(1))

        service.update_runtime_state([target], 0)
        self.assertFalse(service.enrich_reference_update(event))
        service.update_runtime_state([target], 1)
        self.assertTrue(service.enrich_reference_update(event))
        self.assertEqual(len(repository.load_vehicle(1).reference_embeddings), 2)  # type: ignore[union-attr]

    def test_explicit_g_after_auto_recognition_bypasses_cooldown(self) -> None:
        repository = VehicleGalleryRepository(self.path)
        repository.save_vehicle(_vehicle(1, 0))
        gallery = VehicleTargetGallery()
        service = VehicleGalleryPersistenceService(
            gallery,
            repository,
            VehicleGalleryEnrichmentConfig(post_recovery_stable_frames=30),
        )
        service.load()
        target = SessionTarget(
            target_id=1,
            current_track_id=27,
            last_track_id=27,
            state=TargetState.ACTIVE,
            reference_embeddings=[_unit(0)],
            centroid=_unit(0),
        )
        gallery.attach_session_target(1, 1)
        service.mark_auto_recognized(1)
        self.assertEqual(service.enroll(target).vehicle_id, 1)

        self.assertTrue(service.enrich_reference_update(_reference_event(target, _unit(1))))


def _reference_event(target: SessionTarget, embedding: np.ndarray):
    from src.target_recovery import ReferenceUpdateEvent

    return ReferenceUpdateEvent(
        target_id=target.target_id,
        frame_index=15,
        reference_embeddings=(target.reference_embeddings[0].copy(), embedding.copy()),
        centroid=embedding.copy(),
        accepted_embedding=embedding.copy(),
        target_state=TargetState.ACTIVE,
    )


if __name__ == "__main__":
    unittest.main()
