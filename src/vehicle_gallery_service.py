"""Application coordination for the in-memory Vehicle Gallery and SQLite."""

from __future__ import annotations

from collections.abc import Iterable

from .vehicle_database import VehicleGalleryRepository, VehicleRepositoryError
from .vehicle_gallery import GalleryVehicle, VehicleTargetGallery
from .models import SessionTarget


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
    ) -> None:
        self.gallery = gallery
        self.repository = repository

    def load(self) -> tuple[GalleryVehicle, ...]:
        """Load persistent vehicles and deliberately discard old mappings."""

        vehicles = self.repository.load_all()
        next_vehicle_id = self.repository.load_next_vehicle_id()
        self.gallery.restore_vehicles(vehicles, next_vehicle_id=next_vehicle_id)
        return self.gallery.all_vehicles()

    def enroll(self, session_target: SessionTarget) -> GalleryVehicle:
        """Persist one existing Vehicle SessionTarget, idempotently."""

        existing = self.gallery.vehicle_for_session_target(session_target.target_id)
        if existing is not None:
            return existing

        vehicle = self.gallery.enroll(session_target)
        try:
            self.repository.save_vehicle(vehicle)
        except Exception:
            # The allocator intentionally remains advanced: a failed write must
            # never make a later process-local enrollment reuse this ID.
            self.gallery.remove(vehicle.vehicle_id)
            raise
        return vehicle

    def remove(self, vehicle_id: int) -> bool:
        """Delete SQLite first, then remove the in-memory identity/mappings."""

        in_memory = self.gallery.get(vehicle_id)
        deleted = self.repository.delete_vehicle(vehicle_id)
        if not deleted:
            if in_memory is not None:
                raise VehicleRepositoryError(
                    "Vehicle Gallery memory/database mismatch for "
                    f"vehicle_id={vehicle_id}"
                )
            return False
        self.gallery.remove(vehicle_id)
        return True

    def clear(self) -> None:
        """Clear persisted identities first, preserving the monotonic allocator."""

        self.repository.clear()
        self.gallery.clear()

    def detach_session_target(self, target_id: int) -> bool:
        """Detach a runtime target without touching persistent Vehicle data."""

        return self.gallery.detach_session_target(target_id)

    def detach_all_session_targets(
        self, target_ids: Iterable[int] | None = None
    ) -> None:
        """Detach runtime mappings after R/C without deleting Gallery records."""

        self.gallery.detach_all_session_targets(target_ids)
