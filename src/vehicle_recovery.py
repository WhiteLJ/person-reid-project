"""Vehicle Recovery wrapper over the shared SessionTarget recovery core."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from logging import getLogger

import numpy as np

from .config import (
    VehicleReIDConfig,
    VehicleReIDQualityConfig,
    VehicleRecoveryConfig,
)
from .reid_frame_cache import ReIDFrameCache
from .models import TargetState, Track
from .target_manager import TargetManager
from .target_recovery import RecoveryMatch, TargetRecoveryCoordinator
from .vehicle_reid_quality import assess_vehicle_reid_quality


LOGGER = getLogger(__name__)


class VehicleRecoveryCoordinator(TargetRecoveryCoordinator):
    """Apply the shared incremental recovery algorithm to Vehicle tracks.

    The parent owns visibility state, bounded references, evidence scoring,
    confirmation, one-to-one matching, and sweep scheduling.  This wrapper
    supplies only the Vehicle extractor/configuration/class-specific quality
    gate and keeps the Vehicle cache independent from Person.
    """

    def __init__(
        self,
        target_manager: TargetManager,
        reid_extractor: object,
        reid_config: VehicleReIDConfig,
        recovery_config: VehicleRecoveryConfig,
        quality_config: VehicleReIDQualityConfig,
        *,
        vehicle_class_ids: Collection[int] = (2,),
        embedding_cache: ReIDFrameCache | None = None,
    ) -> None:
        classes = tuple(sorted({int(class_id) for class_id in vehicle_class_ids}))
        if not classes:
            raise ValueError("Vehicle Recovery requires at least one vehicle class")
        if len(classes) != 1:
            raise ValueError(
                "PC4A Vehicle Recovery currently supports one class: COCO car=2"
            )
        vehicle_class_id = classes[0]

        def quality_assessor(frame, track, tracks):
            return assess_vehicle_reid_quality(
                frame,
                track,
                tracks,
                quality_config,
                vehicle_class_ids=classes,
            )

        super().__init__(
            target_manager=target_manager,
            reid_extractor=reid_extractor,  # type: ignore[arg-type]
            reid_config=reid_config,  # type: ignore[arg-type]
            recovery_config=recovery_config,  # type: ignore[arg-type]
            embedding_cache=embedding_cache,
            track_class_id=vehicle_class_id,
            quality_assessor=quality_assessor,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        frame_index: int,
    ) -> list[RecoveryMatch]:
        """Run the shared recovery core and emit Vehicle-specific state logs."""

        before_states = {
            target.target_id: (
                target.state,
                target.current_track_id,
                target.last_track_id,
            )
            for target in self.target_manager.targets.values()
        }
        pending_before = self.pending
        matches = super().process_frame(frame, tracks, frame_index)

        for target in self.target_manager.targets.values():
            previous = before_states.get(target.target_id)
            if previous is None:
                continue
            previous_state, previous_track_id, previous_last_track_id = previous
            if previous_state is TargetState.ACTIVE and target.state is TargetState.LOST:
                LOGGER.info(
                    "VEHICLE_TARGET_LOST frame=%d target_id=%d old_track_id=%s",
                    frame_index,
                    target.target_id,
                    previous_track_id
                    if previous_track_id is not None
                    else previous_last_track_id,
                )

        for (target_id, candidate_track_id), hits in self.pending.items():
            if pending_before.get((target_id, candidate_track_id)) == hits:
                continue
            LOGGER.info(
                "VEHICLE_RECOVERY_PENDING frame=%d target_id=%d "
                "candidate_track_id=%d hits=%d/%d",
                frame_index,
                target_id,
                candidate_track_id,
                hits,
                self.recovery_config.recovery_confirmation_hits,
            )

        for match in matches:
            previous = before_states.get(match.target_id)
            old_track_id = (
                previous[2]
                if previous is not None
                else None
            )
            LOGGER.info(
                "VEHICLE_TARGET_RECOVERED frame=%d target_id=%d "
                "old_track_id=%s new_track_id=%d centroid_similarity=%.4f "
                "reference_support_similarity=%.4f",
                frame_index,
                match.target_id,
                old_track_id,
                match.candidate.track.track_id,
                match.centroid_similarity,
                match.reference_support_similarity,
            )
        return matches
