"""Application coordination for the in-memory Vehicle Gallery and SQLite."""

from __future__ import annotations

from collections.abc import Iterable
from logging import getLogger

from .config import VehicleGalleryEnrichmentConfig
from .gallery_reference_bank import (
    normalized_centroid,
    update_persistent_reference_bank,
)
from .models import SessionTarget, TargetState
from .target_recovery import ReferenceUpdateEvent
from .vehicle_database import VehicleGalleryRepository, VehicleRepositoryError
from .vehicle_gallery import GalleryVehicle, VehicleTargetGallery


LOGGER = getLogger(__name__)


class VehicleGalleryPersistenceService:
    """Coordinate Vehicle Gallery memory and persistence operations.

    The domain gallery owns identity and session mappings.  This service owns
    the ordering around SQLite so a failed first enrollment does not leave a
    newly-created in-memory identity behind.
    """

    def __init__(
        self,
        gallery: VehicleTargetGallery,
        repository: VehicleGalleryRepository,
        enrichment_config: VehicleGalleryEnrichmentConfig | None = None,
    ) -> None:
        self.gallery = gallery
        self.repository = repository
        self.enrichment_config = (
            enrichment_config or VehicleGalleryEnrichmentConfig()
        )
        self._explicitly_enrolled_target_ids: set[int] = set()
        self._post_recovery_blocked: set[int] = set()
        self._stable_active_frames: dict[int, int] = {}

    def load(self) -> tuple[GalleryVehicle, ...]:
        """Load persistent vehicles and deliberately discard old mappings."""

        vehicles = self.repository.load_all()
        next_vehicle_id = self.repository.load_next_vehicle_id()
        self.gallery.restore_vehicles(vehicles, next_vehicle_id=next_vehicle_id)
        self._explicitly_enrolled_target_ids.clear()
        self._post_recovery_blocked.clear()
        self._stable_active_frames.clear()
        return self.gallery.all_vehicles()

    def enroll(self, session_target: SessionTarget) -> GalleryVehicle:
        """Persist one existing Vehicle SessionTarget, idempotently."""

        existing = self.gallery.vehicle_for_session_target(session_target.target_id)
        if existing is not None:
            self._mark_explicit_enrollment(session_target.target_id)
            return existing

        vehicle = self.gallery.enroll(session_target)
        try:
            self.repository.save_vehicle(vehicle)
        except Exception:
            # The allocator intentionally remains advanced: a failed write must
            # never make a later process-local enrollment reuse this ID.
            self.gallery.remove(vehicle.vehicle_id)
            raise
        self._mark_explicit_enrollment(session_target.target_id)
        return vehicle

    def mark_auto_recognized(self, target_id: int) -> None:
        """Start the safety cooldown for a cold-start recognized identity."""

        if self.gallery.vehicle_for_session_target(target_id) is None:
            return
        if target_id in self._explicitly_enrolled_target_ids:
            return
        self._post_recovery_blocked.add(target_id)
        self._stable_active_frames[target_id] = 0

    def update_runtime_state(
        self,
        targets: Iterable[SessionTarget],
        frame_index: int,
    ) -> None:
        """Advance auto-recognition and post-recovery enrichment cooldowns."""

        targets_by_id = {target.target_id: target for target in targets}
        for target_id in list(self._post_recovery_blocked):
            if target_id not in targets_by_id:
                self._post_recovery_blocked.discard(target_id)
                self._stable_active_frames.pop(target_id, None)

        for target_id in list(self._post_recovery_blocked):
            target = targets_by_id[target_id]
            if target.state is not TargetState.ACTIVE or target.missing_frames > 0:
                self._stable_active_frames[target_id] = 0
                continue
            if target.last_recovery_frame == frame_index:
                self._stable_active_frames[target_id] = 0
                continue
            stable_frames = self._stable_active_frames.get(target_id, 0) + 1
            stable_frames = min(
                stable_frames,
                self.enrichment_config.post_recovery_stable_frames,
            )
            self._stable_active_frames[target_id] = stable_frames
            if (
                stable_frames
                >= self.enrichment_config.post_recovery_stable_frames
            ):
                self._post_recovery_blocked.discard(target_id)

    def enrich_reference_update(self, event: ReferenceUpdateEvent) -> bool:
        """Persist one already-accepted Vehicle runtime reference safely."""

        target_id = event.target_id
        if event.target_state is not TargetState.ACTIVE:
            return False
        if target_id in self._post_recovery_blocked:
            LOGGER.debug(
                "VEHICLE_GALLERY_ENRICHMENT_SKIPPED target=%d reason=cooldown",
                target_id,
            )
            return False
        vehicle = self.gallery.vehicle_for_session_target(target_id)
        if vehicle is None:
            return False

        try:
            references, changed = update_persistent_reference_bank(
                vehicle.reference_embeddings,
                event.accepted_embedding,
                max_reference_embeddings=(
                    self.enrichment_config.max_reference_embeddings
                ),
                duplicate_similarity_threshold=(
                    self.enrichment_config.duplicate_similarity_threshold
                ),
            )
            if not changed:
                return False
            snapshot = self.gallery.feature_snapshot(
                vehicle.vehicle_id,
                references,
                normalized_centroid(references),
            )
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.debug(
                "VEHICLE_GALLERY_ENRICHMENT_REJECTED target=%d reason=%s",
                target_id,
                exc,
            )
            return False

        try:
            self.repository.update_vehicle_features(
                snapshot.vehicle_id,
                snapshot.reference_embeddings,
                snapshot.centroid,
            )
        except Exception:
            LOGGER.exception(
                "VEHICLE_GALLERY_ENRICHMENT_FAILED vehicle=%s target=%d",
                snapshot.vehicle_id,
                target_id,
            )
            return False

        try:
            self.gallery.apply_feature_snapshot(snapshot)
        except Exception:
            LOGGER.exception(
                "VEHICLE_GALLERY_ENRICHMENT_CONSISTENCY_ERROR vehicle=%s",
                snapshot.vehicle_id,
            )
            try:
                persisted = self.repository.load_vehicle(snapshot.vehicle_id)
                if persisted is None:
                    raise VehicleRepositoryError(
                        f"updated vehicle disappeared: {snapshot.vehicle_id}"
                    )
                self.gallery.apply_feature_snapshot(persisted)
            except Exception:
                LOGGER.exception(
                    "VEHICLE_GALLERY_ENRICHMENT_CONSISTENCY_RESTORE_FAILED vehicle=%s",
                    snapshot.vehicle_id,
                )
            return False

        LOGGER.debug(
            "VEHICLE_GALLERY_ENRICHED vehicle=%s target=%d references=%d",
            snapshot.vehicle_id,
            target_id,
            len(snapshot.reference_embeddings),
        )
        return True

    def enrich_reference_updates(
        self,
        events: Iterable[ReferenceUpdateEvent],
    ) -> int:
        return sum(self.enrich_reference_update(event) for event in events)

    def _mark_explicit_enrollment(self, target_id: int) -> None:
        self._explicitly_enrolled_target_ids.add(target_id)
        self._post_recovery_blocked.discard(target_id)
        self._stable_active_frames[target_id] = (
            self.enrichment_config.post_recovery_stable_frames
        )

    def remove(self, vehicle_id: int) -> bool:
        """Delete SQLite first, then remove the in-memory identity/mappings."""

        in_memory = self.gallery.get(vehicle_id)
        target_id = self.gallery.session_target_for_vehicle_id(vehicle_id)
        deleted = self.repository.delete_vehicle(vehicle_id)
        if not deleted:
            if in_memory is not None:
                raise VehicleRepositoryError(
                    "Vehicle Gallery memory/database mismatch for "
                    f"vehicle_id={vehicle_id}"
                )
            return False
        self.gallery.remove(vehicle_id)
        if target_id is not None:
            self._explicitly_enrolled_target_ids.discard(target_id)
            self._post_recovery_blocked.discard(target_id)
            self._stable_active_frames.pop(target_id, None)
        return True

    def clear(self) -> None:
        """Clear persisted identities first, preserving the monotonic allocator."""

        self.repository.clear()
        self.gallery.clear()
        self._explicitly_enrolled_target_ids.clear()
        self._post_recovery_blocked.clear()
        self._stable_active_frames.clear()

    def detach_session_target(self, target_id: int) -> bool:
        """Detach a runtime target without touching persistent Vehicle data."""

        detached = self.gallery.detach_session_target(target_id)
        self._explicitly_enrolled_target_ids.discard(target_id)
        self._post_recovery_blocked.discard(target_id)
        self._stable_active_frames.pop(target_id, None)
        return detached

    def detach_all_session_targets(
        self, target_ids: Iterable[int] | None = None
    ) -> None:
        """Detach runtime mappings after R/C without deleting Gallery records."""

        if target_ids is None:
            detached_ids = self.gallery.session_target_ids()
        else:
            detached_ids = tuple(target_ids)
        self.gallery.detach_all_session_targets(detached_ids)
        for target_id in detached_ids:
            self._explicitly_enrolled_target_ids.discard(target_id)
            self._post_recovery_blocked.discard(target_id)
            self._stable_active_frames.pop(target_id, None)
