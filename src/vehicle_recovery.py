"""Vehicle Recovery wrapper over the shared SessionTarget recovery core."""

from __future__ import annotations

from collections.abc import Collection

from .config import (
    VehicleReIDConfig,
    VehicleReIDQualityConfig,
    VehicleRecoveryConfig,
)
from .reid_frame_cache import ReIDFrameCache
from .target_manager import TargetManager
from .target_recovery import TargetRecoveryCoordinator
from .vehicle_reid_quality import assess_vehicle_reid_quality


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
