from pathlib import Path
import unittest

import numpy as np

from src.config import GalleryRecognitionConfig, ReIDConfig, ReIDRecoveryConfig
from src.gallery import GalleryPerson, TargetGallery
from src.gallery_recognition import GalleryRecognitionCoordinator
from src.models import TargetState, Track
from src.reid_frame_budget import ReIDFrameBudget
from src.reid_frame_cache import ReIDFrameCache
from src.target_manager import TargetManager
from src.target_recovery import TargetRecoveryCoordinator


class _SequenceExtractor:
    def __init__(self, initial: np.ndarray, outputs: list[np.ndarray]) -> None:
        self.initial = initial.astype(np.float32)
        self.outputs = [value.astype(np.float32) for value in outputs]

    def extract(self, crop: np.ndarray) -> np.ndarray:
        del crop
        return self.initial.copy()

    def extract_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        del crops
        if not self.outputs:
            raise AssertionError("unexpected ReID extraction")
        value = self.outputs.pop(0)
        return value[None, :] if value.ndim == 1 else value.copy()


def _track(track_id: int, x1: int = 10) -> Track:
    return Track(track_id, (x1, 10, x1 + 40, 110), 0.95, 0)


def _reid_config() -> ReIDConfig:
    return ReIDConfig(
        model_name="test",
        weight=Path("unused.pth"),
        image_height=256,
        image_width=128,
        min_crop_width=10,
        min_crop_height=10,
    )


def _recovery_config() -> ReIDRecoveryConfig:
    return ReIDRecoveryConfig(
        lost_grace_frames=1,
        reference_update_interval_frames=100,
        recovery_interval_frames=1,
        max_reference_embeddings=8,
        recovery_threshold=0.80,
        recovery_margin=0.0,
        reference_update_threshold=0.80,
        recovery_reference_support_threshold=0.80,
        recovery_reference_support_top_k=3,
        recovery_min_track_age_frames=1,
        recovery_confirmation_hits=1,
        recovery_pending_max_age_frames=60,
        recovery_candidates_per_frame=1,
    )


class TemporalReIDTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((140, 220, 3), dtype=np.uint8)
        self.person_a = np.asarray((1.0, 0.0), dtype=np.float32)
        self.person_b = np.asarray((0.0, 1.0), dtype=np.float32)

    def test_recovery_revalidates_same_track_id_after_content_switch(self) -> None:
        extractor = _SequenceExtractor(
            self.person_a,
            [self.person_a, self.person_b, self.person_b],
        )
        manager = TargetManager()
        coordinator = TargetRecoveryCoordinator(
            manager,
            extractor,  # type: ignore[arg-type]
            _reid_config(),
            _recovery_config(),
            embedding_cache=ReIDFrameCache(),
            reid_budget=ReIDFrameBudget(2),
        )
        target = coordinator.select_from_track(self.frame, _track(1), 0)
        assert target is not None
        coordinator.process_frame(self.frame, [], 1)

        # Track 64 is first extracted as A.  Before the snapshot completes,
        # the same tracker ID now represents B.
        coordinator.process_frame(
            self.frame,
            [_track(64, 10), _track(65, 100)],
            2,
        )
        matches = coordinator.process_frame(
            self.frame,
            [_track(64, 10), _track(65, 100)],
            3,
        )

        self.assertEqual(matches, [])
        self.assertEqual(target.state, TargetState.LOST)
        self.assertIsNone(target.current_track_id)

    def test_gallery_recognition_revalidates_same_track_id_after_content_switch(self) -> None:
        gallery = TargetGallery()
        gallery.restore_people(
            [
                GalleryPerson(
                    person_id=1,
                    label="Target P001",
                    reference_embeddings=[self.person_a.copy()],
                    centroid=self.person_a.copy(),
                )
            ]
        )
        extractor = _SequenceExtractor(
            self.person_a,
            [self.person_a, self.person_b, self.person_b],
        )
        coordinator = GalleryRecognitionCoordinator(
            TargetManager(),
            gallery,
            extractor,  # type: ignore[arg-type]
            _reid_config(),
            GalleryRecognitionConfig(
                enabled=True,
                recognition_interval_frames=1,
                min_track_age_frames=1,
                recognition_threshold=0.80,
                recognition_margin=0.0,
                confirmation_hits=1,
                recognition_candidates_per_frame=1,
                unmatched_retry_interval_frames=15,
            ),
            _recovery_config(),
            embedding_cache=ReIDFrameCache(),
            reid_budget=ReIDFrameBudget(2),
        )

        coordinator.process_frame(
            self.frame,
            [_track(64, 10), _track(65, 100)],
            0,
        )
        matches = coordinator.process_frame(
            self.frame,
            [_track(64, 10), _track(65, 100)],
            1,
        )

        self.assertEqual(matches, [])
        self.assertEqual(coordinator.gallery_adapter.attached_identity_ids(), set())


if __name__ == "__main__":
    unittest.main()
