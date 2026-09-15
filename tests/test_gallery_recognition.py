from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.config import GalleryRecognitionConfig, ReIDConfig, ReIDRecoveryConfig
from src.database import GalleryRepository
from src.gallery import GalleryPerson, TargetGallery
from src.gallery_recognition import (
    GalleryRecognitionCandidate,
    GalleryRecognitionCoordinator,
    assign_gallery_matches,
    gallery_recognition_score,
    gallery_reference_support_score,
)
from src.gallery_service import GalleryPersistenceService
from src.models import SessionTarget, TargetState, Track
from src.reid_frame_cache import ReIDFrameCache
from src.target_manager import TargetManager
from src.target_recovery import TargetRecoveryCoordinator


class _FakeReIDExtractor:
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


class _NoEnrollGallery(TargetGallery):
    def enroll(self, session_target: SessionTarget) -> GalleryPerson:
        del session_target
        raise AssertionError("automatic recognition must not enroll")


def _reid_config() -> ReIDConfig:
    return ReIDConfig(
        model_name="osnet_x0_25",
        weight=Path("unused.pth"),
        image_height=256,
        image_width=128,
        min_crop_width=20,
        min_crop_height=50,
    )


def _recovery_config(**overrides: object) -> ReIDRecoveryConfig:
    values: dict[str, object] = {
        "lost_grace_frames": 1,
        "reference_update_interval_frames": 15,
        "recovery_interval_frames": 10,
        "max_reference_embeddings": 8,
        "recovery_threshold": 0.75,
        "recovery_margin": 0.05,
        "reference_update_threshold": 0.80,
        # Legacy integration cases focus on Gallery hand-off; dedicated
        # MVP-8.1 recovery tests exercise the production confirmation values.
        "recovery_min_track_age_frames": 1,
        "recovery_confirmation_hits": 1,
        "recovery_pending_max_age_frames": 60,
    }
    values.update(overrides)
    return ReIDRecoveryConfig(**values)  # type: ignore[arg-type]


def _recognition_config(**overrides: object) -> GalleryRecognitionConfig:
    values: dict[str, object] = {
        "enabled": True,
        "recognition_interval_frames": 1,
        "min_track_age_frames": 1,
        "recognition_threshold": 0.80,
        "recognition_margin": 0.05,
        "confirmation_hits": 1,
        # Existing focused tests use one probe unless they specifically test
        # the production multi-probe safety policy.
        "probe_embeddings": 1,
        "reference_support_threshold": 0.75,
        "reference_support_top_k": 3,
    }
    values.update(overrides)
    return GalleryRecognitionConfig(**values)  # type: ignore[arg-type]


def _unit(value: tuple[float, ...]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _person(person_id: int, value: tuple[float, ...]) -> GalleryPerson:
    vector = _unit(value)
    return GalleryPerson(
        person_id=person_id,
        label=f"Target P{person_id:03d}",
        reference_embeddings=[vector.copy()],
        centroid=vector.copy(),
    )


def _unit512(first: float, second: float = 0.0) -> np.ndarray:
    vector = np.zeros((512,), dtype=np.float32)
    vector[0] = first
    vector[1] = second
    return vector / np.linalg.norm(vector)


def _gallery(*people: GalleryPerson) -> TargetGallery:
    gallery = TargetGallery()
    gallery.restore_people(people)
    return gallery


def _track(track_id: int, x1: int = 0) -> Track:
    return Track(track_id, (x1, 0, x1 + 40, 110), 0.9, 0)


def _coordinator(
    manager: TargetManager,
    gallery: TargetGallery,
    extractor: _FakeReIDExtractor,
    **overrides: object,
) -> GalleryRecognitionCoordinator:
    return GalleryRecognitionCoordinator(
        manager,
        gallery,
        extractor,  # type: ignore[arg-type]
        _reid_config(),
        _recognition_config(**overrides),
        _recovery_config(),
        embedding_cache=ReIDFrameCache(),
    )


class GalleryRecognitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((120, 100, 3), dtype=np.uint8)

    def test_empty_gallery_does_not_run_reid(self) -> None:
        extractor = _FakeReIDExtractor([_unit((1, 0))])
        coordinator = _coordinator(TargetManager(), TargetGallery(), extractor)

        self.assertEqual(coordinator.process_frame(self.frame, [_track(1)], 0), [])
        self.assertEqual(extractor.batch_calls, 0)

    def test_interval_and_track_age_are_enforced(self) -> None:
        extractor = _FakeReIDExtractor([_unit((1, 0)), _unit((1, 0))])
        coordinator = _coordinator(
            TargetManager(),
            _gallery(_person(1, (1, 0))),
            extractor,
            recognition_interval_frames=2,
            min_track_age_frames=2,
            confirmation_hits=2,
        )

        coordinator.process_frame(self.frame, [_track(1)], 0)
        self.assertEqual(extractor.batch_calls, 0)
        coordinator.process_frame(self.frame, [_track(1)], 1)
        self.assertEqual(extractor.batch_calls, 1)
        coordinator.process_frame(self.frame, [_track(1)], 2)
        self.assertEqual(extractor.batch_calls, 1)
        coordinator.process_frame(self.frame, [_track(1)], 3)
        self.assertEqual(extractor.batch_calls, 2)

    def test_candidates_are_batch_extracted_and_high_similarity_is_recognized(self) -> None:
        extractor = _FakeReIDExtractor([np.asarray(((1, 0), (0, 1)), dtype=np.float32)])
        manager = TargetManager()
        gallery = _gallery(_person(1, (1, 0)), _person(2, (0, 1)))
        coordinator = _coordinator(manager, gallery, extractor)

        matches = coordinator.process_frame(
            self.frame,
            [_track(11), _track(12, 50)],
            0,
        )

        self.assertEqual(extractor.batch_sizes, [2])
        self.assertEqual(
            {(match.person_id, match.candidate.track.track_id) for match in matches},
            {(1, 11), (2, 12)},
        )
        self.assertEqual(gallery.session_target_for_person_id(1), 1)
        self.assertEqual(gallery.session_target_for_person_id(2), 2)

    def test_low_similarity_is_not_recognized(self) -> None:
        extractor = _FakeReIDExtractor([_unit((0, 1))])
        manager = TargetManager()
        gallery = _gallery(_person(1, (1, 0)))
        coordinator = _coordinator(manager, gallery, extractor)

        self.assertEqual(coordinator.process_frame(self.frame, [_track(1)], 0), [])
        self.assertEqual(manager.targets, {})

    def test_two_sided_margin_rejects_ambiguous_assignments(self) -> None:
        people = [_person(1, (1, 0)), _person(2, (0.99, 0.14))]
        candidates = [
            GalleryRecognitionCandidate(_track(11), _unit((1, 0))),
            GalleryRecognitionCandidate(_track(12, 50), _unit((0.99, 0.14))),
        ]

        matches = assign_gallery_matches(people, candidates, 0.80, 0.05)

        self.assertEqual(matches, [])

    def test_single_person_and_candidate_have_no_second_best_margin_requirement(self) -> None:
        matches = assign_gallery_matches(
            [_person(1, (1, 0))],
            [GalleryRecognitionCandidate(_track(11), _unit((1, 0)))],
            recognition_threshold=0.80,
            recognition_margin=0.50,
        )

        self.assertEqual([(match.person_id, match.candidate.track.track_id) for match in matches], [(1, 11)])

    def test_single_person_similar_stranger_fails_reference_support_gate(self) -> None:
        reference_a = _unit512(1.0, 0.0)
        reference_b = _unit512(0.8, 0.6)
        reference_c = _unit512(0.8, -0.6)
        person = GalleryPerson(
            person_id=1,
            label="Target P001",
            reference_embeddings=[reference_a, reference_b, reference_c],
            # The centroid of this symmetric bank is the first basis vector.
            centroid=reference_a.copy(),
        )
        stranger = _unit512(0.85, 0.5267827)

        self.assertGreater(
            gallery_recognition_score(person, stranger),
            0.80,
        )
        self.assertLess(
            gallery_reference_support_score(person, stranger, top_k=3),
            0.75,
        )
        matches = assign_gallery_matches(
            [person],
            [GalleryRecognitionCandidate(_track(11), stranger)],
            recognition_threshold=0.80,
            recognition_margin=0.05,
            reference_support_threshold=0.75,
            reference_support_top_k=3,
        )

        self.assertEqual(matches, [])

    def test_reference_support_uses_top_k_mean_not_max(self) -> None:
        reference_a = _unit((1, 0))
        reference_b = _unit((0.8, 0.6))
        reference_c = _unit((0.8, -0.6))
        person = GalleryPerson(
            person_id=1,
            label="Target P001",
            reference_embeddings=[reference_a, reference_b, reference_c],
            centroid=reference_a.copy(),
        )

        self.assertAlmostEqual(
            gallery_reference_support_score(person, reference_a, top_k=2),
            0.9,
            places=5,
        )

    def test_probe_bank_must_be_full_before_binding(self) -> None:
        extractor = _FakeReIDExtractor([_unit((1, 0))])
        manager = TargetManager()
        gallery = _gallery(_person(1, (1, 0)))
        coordinator = _coordinator(
            manager,
            gallery,
            extractor,
            probe_embeddings=3,
        )

        self.assertEqual(coordinator.process_frame(self.frame, [_track(7)], 0), [])
        self.assertEqual(coordinator.probe_counts, {7: 1})
        self.assertEqual(manager.targets, {})

    def test_probe_bank_is_cleared_when_track_disappears(self) -> None:
        extractor = _FakeReIDExtractor([_unit((1, 0))])
        coordinator = _coordinator(
            TargetManager(),
            _gallery(_person(1, (1, 0))),
            extractor,
            probe_embeddings=3,
        )

        coordinator.process_frame(self.frame, [_track(7)], 0)
        coordinator.process_frame(self.frame, [], 1)

        self.assertEqual(coordinator.probe_counts, {})

    def test_auto_binding_copies_gallery_references_without_current_probe(self) -> None:
        gallery_person = GalleryPerson(
            person_id=1,
            label="Target P001",
            reference_embeddings=[_unit((1, 0)), _unit((0.99, 0.1))],
            centroid=_unit((1, 0)),
        )
        gallery = _gallery(gallery_person)
        extractor = _FakeReIDExtractor([_unit((1, 0))])
        manager = TargetManager()
        coordinator = _coordinator(manager, gallery, extractor)

        matches = coordinator.process_frame(self.frame, [_track(5)], 0)

        self.assertEqual(len(matches), 1)
        target = manager.target_for_track(5)
        assert target is not None
        self.assertEqual(len(target.reference_embeddings), 2)
        self.assertTrue(
            all(
                not np.shares_memory(runtime_reference, gallery_reference)
                for runtime_reference, gallery_reference in zip(
                    target.reference_embeddings,
                    gallery_person.reference_embeddings,
                )
            )
        )

    def test_restart_load_rejects_stranger_then_recognizes_existing_person(self) -> None:
        database_path = Path(".test_tmp") / "gallery_recognition_restart.db"
        database_path.parent.mkdir(parents=True, exist_ok=True)
        database_path.unlink(missing_ok=True)

        reference_a = _unit512(1.0, 0.0)
        reference_b = _unit512(0.8, 0.6)
        reference_c = _unit512(0.8, -0.6)
        persisted_person = GalleryPerson(
            person_id=1,
            label="Target P001",
            reference_embeddings=[reference_a, reference_b, reference_c],
            centroid=reference_a.copy(),
        )
        repository = GalleryRepository(database_path)
        try:
            repository.initialize()
            repository.save_person(persisted_person)

            # A new process starts with a fresh manager/gallery/coordinator.
            manager = TargetManager()
            gallery = TargetGallery()
            service = GalleryPersistenceService(gallery, repository)
            service.load()
            self.assertIsNone(gallery.session_target_for_person_id(1))

            stranger = _unit512(0.85, 0.5267827)
            extractor = _FakeReIDExtractor(
                [stranger, stranger, stranger]
                + [reference_a, reference_a, reference_a, reference_a]
            )
            coordinator = _coordinator(
                manager,
                gallery,
                extractor,
                probe_embeddings=3,
                confirmation_hits=2,
            )

            for frame_index in range(3):
                self.assertEqual(
                    coordinator.process_frame(
                        self.frame,
                        [_track(10)],
                        frame_index,
                    ),
                    [],
                )
            self.assertEqual(manager.targets, {})
            coordinator.process_frame(self.frame, [], 3)

            for frame_index in range(4, 8):
                coordinator.process_frame(
                    self.frame,
                    [_track(11)],
                    frame_index,
                )

            self.assertEqual(len(manager.targets), 1)
            target = manager.target_for_track(11)
            assert target is not None
            self.assertEqual(gallery.person_for_session_target(target.target_id).person_id, 1)
            self.assertEqual([person.person_id for person in gallery.all_people()], [1])
            self.assertEqual(len(target.reference_embeddings), 3)
        finally:
            database_path.unlink(missing_ok=True)

    def test_frame_cache_is_not_reused_after_frame_changes(self) -> None:
        cache = ReIDFrameCache()
        cache.begin_frame(0)
        cache.put(11, _unit((0, 1)), 0)
        extractor = _FakeReIDExtractor([_unit((1, 0))])
        coordinator = GalleryRecognitionCoordinator(
            TargetManager(),
            _gallery(_person(1, (1, 0))),
            extractor,  # type: ignore[arg-type]
            _reid_config(),
            _recognition_config(),
            _recovery_config(),
            embedding_cache=cache,
        )
        track = _track(11)

        coordinator.process_frame(self.frame, [track], 0)
        matches = coordinator.process_frame(self.frame, [track], 1)

        self.assertEqual(extractor.batch_calls, 1)
        self.assertEqual(len(matches), 1)

    def test_recovery_checked_but_unmatched_track_remains_gallery_eligible(self) -> None:
        manager = TargetManager()
        old_track = _track(3)
        target = manager.select(old_track, _unit((0, 1)))
        extractor = _FakeReIDExtractor([_unit((1, 0))])
        cache = ReIDFrameCache()
        recovery = TargetRecoveryCoordinator(
            manager,
            extractor,  # type: ignore[arg-type]
            _reid_config(),
            _recovery_config(),
            embedding_cache=cache,
        )
        gallery = _gallery(_person(1, (1, 0)))
        recognition = GalleryRecognitionCoordinator(
            manager,
            gallery,
            extractor,  # type: ignore[arg-type]
            _reid_config(),
            _recognition_config(),
            _recovery_config(),
            embedding_cache=cache,
        )

        recovery.process_frame(self.frame, [old_track], 0)
        new_track = _track(11)
        recovery.process_frame(self.frame, [new_track], 1)
        matches = recognition.process_frame(
            self.frame,
            [new_track],
            1,
            protected_track_ids=recovery.last_recovered_track_ids,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].candidate.track.track_id, 11)
        self.assertEqual(extractor.batch_calls, 1)

    def test_successfully_recovered_track_is_excluded(self) -> None:
        manager = TargetManager()
        old_track = _track(3)
        manager.select(old_track, _unit((1, 0)))
        extractor = _FakeReIDExtractor([_unit((1, 0))])
        cache = ReIDFrameCache()
        recovery = TargetRecoveryCoordinator(
            manager,
            extractor,  # type: ignore[arg-type]
            _reid_config(),
            _recovery_config(),
            embedding_cache=cache,
        )
        gallery = _gallery(_person(1, (1, 0)))
        recognition = GalleryRecognitionCoordinator(
            manager,
            gallery,
            extractor,  # type: ignore[arg-type]
            _reid_config(),
            _recognition_config(),
            _recovery_config(),
            embedding_cache=cache,
        )

        recovery.process_frame(self.frame, [old_track], 0)
        new_track = _track(11)
        recovery.process_frame(self.frame, [new_track], 1)
        matches = recognition.process_frame(
            self.frame,
            [new_track],
            1,
            protected_track_ids=recovery.last_recovered_track_ids,
        )

        self.assertEqual(len(recovery.last_recovered_track_ids), 1)
        self.assertEqual(matches, [])
        self.assertEqual(len(manager.targets), 1)
        self.assertEqual(extractor.batch_calls, 1)

    def test_manual_session_target_is_attached_without_new_target(self) -> None:
        manager = TargetManager()
        track = _track(5)
        target = manager.select(track, _unit((1, 0)))
        gallery = _gallery(_person(1, (1, 0)))
        extractor = _FakeReIDExtractor([_unit((1, 0))])
        coordinator = _coordinator(manager, gallery, extractor)

        matches = coordinator.process_frame(self.frame, [track], 0)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].person_id, 1)
        self.assertEqual(tuple(manager.targets), (target.target_id,))
        self.assertEqual(gallery.session_target_for_person_id(1), target.target_id)

    def test_recognition_does_not_enroll_or_create_a_gallery_person(self) -> None:
        manager = TargetManager()
        gallery = _NoEnrollGallery()
        gallery.restore_people([_person(1, (1, 0))])
        extractor = _FakeReIDExtractor([_unit((1, 0))])
        coordinator = _coordinator(manager, gallery, extractor)

        matches = coordinator.process_frame(self.frame, [_track(5)], 0)

        self.assertEqual(len(matches), 1)
        self.assertEqual([person.person_id for person in gallery.all_people()], [1])

    def test_lost_gallery_binding_remains_occupied(self) -> None:
        manager = TargetManager()
        target = manager.select(_track(3), _unit((1, 0)))
        target.state = TargetState.LOST
        target.current_track_id = None
        gallery = _gallery(_person(1, (1, 0)))
        gallery.attach_session_target(target.target_id, 1)
        extractor = _FakeReIDExtractor([_unit((1, 0))])
        coordinator = _coordinator(manager, gallery, extractor)

        self.assertEqual(coordinator.process_frame(self.frame, [_track(11)], 0), [])
        self.assertEqual(extractor.batch_calls, 0)

    def test_recognized_target_has_copied_runtime_references(self) -> None:
        manager = TargetManager()
        source = _person(1, (1, 0))
        gallery = _gallery(source)
        extractor = _FakeReIDExtractor([_unit((1, 0))])
        coordinator = _coordinator(manager, gallery, extractor)

        coordinator.process_frame(self.frame, [_track(5)], 0)
        target = manager.target_for_track(5)
        assert target is not None
        target.reference_embeddings[0][0] = 0.0
        target.centroid[0] = 0.0

        self.assertNotEqual(float(source.reference_embeddings[0][0]), 0.0)
        self.assertNotEqual(float(source.centroid[0]), 0.0)

    def test_confirmation_pending_survives_intermediate_frames(self) -> None:
        extractor = _FakeReIDExtractor(
            [_unit((1, 0)), _unit((1, 0))]
        )
        gallery = _gallery(_person(1, (1, 0)))
        coordinator = _coordinator(
            TargetManager(),
            gallery,
            extractor,
            recognition_interval_frames=3,
            confirmation_hits=2,
        )
        track = _track(7)

        coordinator.process_frame(self.frame, [track], 0)
        self.assertEqual(coordinator.pending, {(1, 7): 1})
        coordinator.process_frame(self.frame, [track], 1)
        coordinator.process_frame(self.frame, [track], 2)
        self.assertEqual(coordinator.pending, {(1, 7): 1})
        matches = coordinator.process_frame(self.frame, [track], 3)

        self.assertEqual(len(matches), 1)
        self.assertEqual(coordinator.pending, {})

    def test_pending_is_cleared_when_track_disappears(self) -> None:
        extractor = _FakeReIDExtractor([_unit((1, 0))])
        gallery = _gallery(_person(1, (1, 0)))
        coordinator = _coordinator(
            TargetManager(),
            gallery,
            extractor,
            confirmation_hits=2,
        )
        track = _track(7)
        coordinator.process_frame(self.frame, [track], 0)
        coordinator.process_frame(self.frame, [], 1)
        self.assertEqual(coordinator.pending, {})

    def test_occupied_gallery_person_is_not_recognized_again(self) -> None:
        manager = TargetManager()
        track = _track(3)
        target = manager.select(track, _unit((1, 0)))
        gallery = _gallery(_person(1, (1, 0)))
        gallery.attach_session_target(target.target_id, 1)
        extractor = _FakeReIDExtractor([_unit((1, 0))])
        coordinator = _coordinator(manager, gallery, extractor)

        self.assertEqual(coordinator.process_frame(self.frame, [track], 0), [])
        self.assertEqual(extractor.batch_calls, 0)
        self.assertEqual(len(manager.targets), 1)


if __name__ == "__main__":
    unittest.main()
