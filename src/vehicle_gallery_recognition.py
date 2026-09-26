"""Vehicle Gallery recognition wrapper over the shared recognition core."""

from __future__ import annotations

from collections.abc import Collection

from .config import (
    GalleryRecognitionConfig,
    ReIDRecoveryConfig,
    VehicleReIDConfig,
    VehicleGalleryRecognitionConfig,
    VehicleReIDQualityConfig,
)
from .gallery_recognition import (
    GalleryRecognitionAdapter,
    GalleryRecognitionCoordinator,
)
from .reid_frame_cache import ReIDFrameCache
from .target_manager import TargetManager
from .vehicle_gallery import VehicleTargetGallery, format_vehicle_id
from .vehicle_reid_quality import assess_vehicle_reid_quality


class VehicleGalleryRecognitionCoordinator(GalleryRecognitionCoordinator):
    """Use the Person recognition core with Vehicle-specific dependencies."""

    def __init__(
        self,
        target_manager: TargetManager,
        gallery: VehicleTargetGallery,
        reid_extractor: object,
        reid_config: VehicleReIDConfig,
        recognition_config: (
            GalleryRecognitionConfig | VehicleGalleryRecognitionConfig
        ),
        recovery_config: ReIDRecoveryConfig,
        quality_config: VehicleReIDQualityConfig,
        *,
        vehicle_class_ids: Collection[int] = (2,),
        embedding_cache: ReIDFrameCache | None = None,
    ) -> None:
        classes = tuple(sorted({int(class_id) for class_id in vehicle_class_ids}))
        if not classes:
            raise ValueError("Vehicle Gallery recognition requires a class")

        def quality_assessor(frame, track, tracks):
            return assess_vehicle_reid_quality(
                frame,
                track,
                tracks,
                quality_config,
                vehicle_class_ids=classes,
            )

        adapter = GalleryRecognitionAdapter(
            all_identities=gallery.all_vehicles,
            attached_identity_ids=gallery.attached_vehicle_ids,
            get_identity=gallery.get,
            identity_for_session_target=gallery.vehicle_for_session_target,
            attach_session_target=gallery.attach_session_target,
            identity_id=lambda vehicle: vehicle.vehicle_id,
            identity_label=format_vehicle_id,
        )
        super().__init__(
            target_manager=target_manager,
            gallery=gallery,  # type: ignore[arg-type]
            reid_extractor=reid_extractor,  # type: ignore[arg-type]
            reid_config=reid_config,  # type: ignore[arg-type]
            recognition_config=recognition_config,
            recovery_config=recovery_config,
            person_class_id=2,
            track_class_ids=classes,
            embedding_cache=embedding_cache,
            quality_assessor=quality_assessor,
            gallery_adapter=adapter,
        )
