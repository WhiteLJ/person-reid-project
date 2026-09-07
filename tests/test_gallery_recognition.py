from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.config import GalleryRecognitionConfig, ReIDConfig, ReIDRecoveryConfig
from src.gallery import GalleryPerson, TargetGallery
from src.gallery_recognition import (
    GalleryRecognitionCandidate,
    GalleryRecognitionCoordinator,
    assign_gallery_matches,
)
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
