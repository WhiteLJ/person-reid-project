from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src.config import GalleryRecognitionConfig, ReIDConfig, ReIDRecoveryConfig
from src.gallery import GalleryPerson, TargetGallery
from src.gallery_recognition import GalleryRecognitionCoordinator
from src.models import Track
from src.reid_frame_cache import ReIDFrameCache
from src.target_manager import TargetManager


def _unit(index: int) -> np.ndarray:
    value = np.zeros((4,), dtype=np.float32)
    value[index] = 1.0
    return value


def _track(track_id: int) -> Track:
    left = 10 + track_id * 50
    return Track(track_id, (left, 10, left + 30, 100), 0.95, 0)


def _gallery() -> TargetGallery:
    gallery = TargetGallery()
    vector = _unit(0)
    gallery.restore_people(
        [
            GalleryPerson(
                person_id=1,
                label="Target P001",
                reference_embeddings=[vector.copy()],
                centroid=vector.copy(),
            )
        ]
    )
    return gallery


def _recognition_config(**overrides: object) -> GalleryRecognitionConfig:
    values: dict[str, object] = {
        "enabled": True,
        "recognition_interval_frames": 1,
        "min_track_age_frames": 1,
        "recognition_threshold": 0.80,
        "recognition_margin": 0.05,
        "confirmation_hits": 1,
        "recognition_candidates_per_frame": 3,
        "unmatched_retry_interval_frames": 15,
    }
    values.update(overrides)
    return GalleryRecognitionConfig(**values)  # type: ignore[arg-type]


def _coordinator(
    extractor: "_FakeExtractor",
    *,
    confirmation_hits: int = 1,
    candidates_per_frame: int = 3,
    retry_interval: int = 15,
) -> GalleryRecognitionCoordinator:
    return GalleryRecognitionCoordinator(
        TargetManager(),
        _gallery(),
        extractor,  # type: ignore[arg-type]
        ReIDConfig(
            model_name="test",
            weight=Path("unused.pth"),
            image_height=256,
            image_width=128,
            min_crop_width=10,
            min_crop_height=10,
        ),
        _recognition_config(
            confirmation_hits=confirmation_hits,
            recognition_candidates_per_frame=candidates_per_frame,
            unmatched_retry_interval_frames=retry_interval,
        ),
        ReIDRecoveryConfig(
            lost_grace_frames=1,
            reference_update_interval_frames=15,
            recovery_interval_frames=5,
            max_reference_embeddings=8,
            recovery_threshold=0.85,
            recovery_margin=0.05,
            reference_update_threshold=0.80,
        ),
        embedding_cache=ReIDFrameCache(),
    )


class _FakeExtractor:
    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = [np.asarray(output, dtype=np.float32) for output in outputs]
        self.batch_sizes: list[int] = []

    def extract_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        self.batch_sizes.append(len(crops))
        output = self.outputs.pop(0)
        if output.ndim == 1:
            return np.repeat(output[None, :], len(crops), axis=0)
        return output.copy()


class GalleryRecognitionSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((120, 500, 3), dtype=np.uint8)

    def test_person_budget_splits_seven_candidates_3_3_1(self) -> None:
        high = _unit(0)
        low = _unit(1)
        extractor = _FakeExtractor(
            [
                np.asarray((high, low, low), dtype=np.float32),
                np.asarray((low, low, low), dtype=np.float32),
                low,
                high,
            ]
        )
        coordinator = _coordinator(extractor)
        tracks = [_track(index) for index in range(1, 8)]

        self.assertEqual(coordinator.process_frame(self.frame, tracks, 0), [])
        self.assertEqual(extractor.batch_sizes, [3])
        self.assertFalse(coordinator.last_frame_recognition_stats.sweep_completed)
        self.assertEqual(coordinator.process_frame(self.frame, tracks, 1), [])
        self.assertEqual(extractor.batch_sizes, [3, 3])
        self.assertEqual(coordinator.process_frame(self.frame, tracks, 2)[0].person_id, 1)
        self.assertEqual(extractor.batch_sizes, [3, 3, 1, 1])

    def test_partial_sweep_never_matches_and_confirmation_counts_sweeps(self) -> None:
        high = _unit(0)
        low = _unit(1)
        extractor = _FakeExtractor([high, low, low, high, high])
        coordinator = _coordinator(
            extractor,
            candidates_per_frame=1,
            confirmation_hits=2,
        )
        tracks = [_track(1), _track(2), _track(3)]

        self.assertEqual(coordinator.process_frame(self.frame, tracks, 0), [])
        self.assertEqual(coordinator.pending, {})
        self.assertEqual(coordinator.process_frame(self.frame, tracks, 1), [])
        self.assertEqual(coordinator.pending, {})
        self.assertEqual(coordinator.process_frame(self.frame, tracks, 2), [])
        self.assertEqual(coordinator.pending, {(1, 1): 1})
        # The second sweep contains only the still-pending track: the two
        # unmatched tracks entered retry cooldown after the first complete
        # sweep.  It still requires a complete sweep before hit two.
        self.assertEqual(
            coordinator.process_frame(self.frame, tracks, 3)[0].person_id,
            1,
        )

    def test_unmatched_track_retry_cooldown_skips_extraction(self) -> None:
        extractor = _FakeExtractor([_unit(1), _unit(1)])
        coordinator = _coordinator(extractor, retry_interval=15)
        track = _track(1)

        self.assertEqual(coordinator.process_frame(self.frame, [track], 0), [])
        self.assertEqual(extractor.batch_sizes, [1])
        for frame_index in range(1, 15):
            coordinator.process_frame(self.frame, [track], frame_index)
        self.assertEqual(extractor.batch_sizes, [1])
        self.assertEqual(coordinator.retry_skipped_count, 14)
        coordinator.process_frame(self.frame, [track], 15)
        self.assertEqual(extractor.batch_sizes, [1, 1])

    def test_pending_match_is_not_put_in_retry_cooldown(self) -> None:
        extractor = _FakeExtractor([_unit(0), _unit(0)])
        coordinator = _coordinator(
            extractor,
            confirmation_hits=2,
            retry_interval=15,
        )
        track = _track(1)

        coordinator.process_frame(self.frame, [track], 0)
        self.assertEqual(coordinator.pending, {(1, 1): 1})
        self.assertEqual(coordinator.retry_until_frame, {})
        coordinator.process_frame(self.frame, [track], 1)
        self.assertEqual(coordinator.retry_until_frame, {})

    def test_quality_rejection_does_not_create_retry_cooldown(self) -> None:
        extractor = _FakeExtractor([_unit(0)])
        coordinator = _coordinator(extractor)
        rejected = Track(1, (0, 10, 30, 100), 0.95, 0)
        valid = _track(1)

        coordinator.process_frame(self.frame, [rejected], 0)
        self.assertEqual(extractor.batch_sizes, [])
        coordinator.process_frame(self.frame, [valid], 1)
        self.assertEqual(extractor.batch_sizes, [1])
        self.assertEqual(coordinator.retry_until_frame, {})

    def test_track_disappearance_clears_retry_and_new_track_has_no_cooldown(self) -> None:
        extractor = _FakeExtractor([_unit(1), _unit(1)])
        coordinator = _coordinator(extractor)
        track = _track(1)

        coordinator.process_frame(self.frame, [track], 0)
        self.assertEqual(coordinator.retry_until_frame, {1: 15})
        coordinator.process_frame(self.frame, [], 1)
        self.assertEqual(coordinator.retry_until_frame, {})
        coordinator.process_frame(self.frame, [_track(2)], 2)
        self.assertEqual(extractor.batch_sizes, [1, 1])

    def test_gallery_change_invalidates_retry_state(self) -> None:
        extractor = _FakeExtractor([_unit(1), _unit(1)])
        coordinator = _coordinator(extractor)
        track = _track(1)

        coordinator.process_frame(self.frame, [track], 0)
        coordinator.notify_gallery_changed()
        coordinator.process_frame(self.frame, [track], 1)
        self.assertEqual(extractor.batch_sizes, [1, 1])


if __name__ == "__main__":
    unittest.main()
