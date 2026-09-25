"""In-memory persistent-identity domain for Vehicle Gallery (PC5)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .models import SessionTarget


VEHICLE_EMBEDDING_DIMENSION = 2048
VEHICLE_EMBEDDING_DTYPE = "float32"


def _empty_embedding() -> np.ndarray:
    return np.empty((0,), dtype=np.float32)


def format_vehicle_id(vehicle_id: int) -> str:
    """Format a persistent Vehicle Gallery ID such as ``V001``."""

    _validate_vehicle_id(vehicle_id)
    return f"V{vehicle_id:03d}"


def parse_vehicle_id(value: str | int) -> int:
    """Parse ``V001`` or a positive integer for administrative commands."""

    if isinstance(value, bool):
        raise ValueError(f"invalid vehicle ID: {value}")
    if isinstance(value, int):
        return _validate_vehicle_id(value)
    normalized = str(value).strip().upper()
    if normalized.startswith("V"):
        normalized = normalized[1:]
    if not normalized.isdigit():
        raise ValueError(f"invalid vehicle ID: {value}")
    return _validate_vehicle_id(int(normalized))


@dataclass
class GalleryVehicle:
    """A persistent in-memory Vehicle Gallery identity."""

    vehicle_id: int
    label: str
    reference_embeddings: list[np.ndarray] = field(default_factory=list)
    centroid: np.ndarray = field(default_factory=_empty_embedding)


class VehicleTargetGallery:
    """Own Vehicle Gallery identities and current-session associations.

    Vehicle Gallery records are persistent identities, while target mappings
    are deliberately process-local and are never restored from SQLite.
    """

    def __init__(self) -> None:
        self._vehicles: dict[int, GalleryVehicle] = {}
        self._session_target_to_vehicle: dict[int, int] = {}
        self._next_vehicle_id = 1

    def restore_vehicles(
        self,
        vehicles: Iterable[GalleryVehicle],
        next_vehicle_id: int | None = None,
    ) -> None:
        """Restore persisted vehicles without old SessionTarget mappings."""

        restored: dict[int, GalleryVehicle] = {}
        for vehicle in vehicles:
            vehicle_id = _validate_vehicle_id(vehicle.vehicle_id)
            if vehicle_id in restored:
                raise ValueError(f"duplicate vehicle_id: {vehicle_id}")
            restored[vehicle_id] = _copy_gallery_vehicle(vehicle)

        minimum_next_id = max(restored, default=0) + 1
        if next_vehicle_id is None:
            restored_next_id = minimum_next_id
        else:
            restored_next_id = _validate_vehicle_id(next_vehicle_id)
            restored_next_id = max(restored_next_id, minimum_next_id)

        self._vehicles = restored
        self._session_target_to_vehicle.clear()
        self._next_vehicle_id = restored_next_id

    def enroll(self, session_target: SessionTarget) -> GalleryVehicle:
        """Snapshot a valid Vehicle SessionTarget, idempotently."""

        existing_vehicle_id = self._session_target_to_vehicle.get(
            session_target.target_id
        )
        if existing_vehicle_id is not None:
            existing = self._vehicles.get(existing_vehicle_id)
            if existing is not None:
                return existing
            del self._session_target_to_vehicle[session_target.target_id]

        references = _copy_session_target_references(session_target)
        centroid = _copy_vehicle_embedding(
            session_target.centroid,
            context=f"SessionTarget {session_target.target_id} centroid",
        )
        vehicle_id = self._next_vehicle_id
        self._next_vehicle_id += 1
        vehicle = GalleryVehicle(
            vehicle_id=vehicle_id,
            label=f"Target {format_vehicle_id(vehicle_id)}",
            reference_embeddings=references,
            centroid=centroid,
        )
        self._vehicles[vehicle_id] = vehicle
        self._session_target_to_vehicle[session_target.target_id] = vehicle_id
        return vehicle

    def get(self, vehicle_id: int) -> GalleryVehicle | None:
        return self._vehicles.get(_validate_vehicle_id(vehicle_id))

    def feature_snapshot(
        self,
        vehicle_id: int,
        reference_embeddings: Iterable[np.ndarray],
        centroid: np.ndarray,
    ) -> GalleryVehicle:
        """Build a validated detached feature snapshot without mutation."""

        current = self.get(vehicle_id)
        if current is None:
            raise KeyError(f"unknown Vehicle Gallery identity: {vehicle_id}")
        return _copy_gallery_vehicle(
            GalleryVehicle(
                vehicle_id=current.vehicle_id,
                label=current.label,
                reference_embeddings=[
                    np.asarray(reference).copy()
                    for reference in reference_embeddings
                ],
                centroid=np.asarray(centroid).copy(),
            )
        )

    def apply_feature_snapshot(self, snapshot: GalleryVehicle) -> GalleryVehicle:
        """Apply validated features while preserving SessionTarget mappings."""

        current = self.get(snapshot.vehicle_id)
        if current is None:
            raise KeyError(
                f"unknown Vehicle Gallery identity: {snapshot.vehicle_id}"
            )
        if snapshot.label != current.label:
            raise ValueError(
                "feature update cannot change label for "
                f"vehicle_id={snapshot.vehicle_id}"
            )
        updated = _copy_gallery_vehicle(snapshot)
        self._vehicles[snapshot.vehicle_id] = updated
        return updated

    def update_vehicle_features(
        self,
        vehicle_id: int,
        reference_embeddings: Iterable[np.ndarray],
        centroid: np.ndarray,
    ) -> GalleryVehicle:
        """Replace one bounded normalized feature snapshot in memory."""

        return self.apply_feature_snapshot(
            self.feature_snapshot(vehicle_id, reference_embeddings, centroid)
        )

    def all_vehicles(self) -> tuple[GalleryVehicle, ...]:
        return tuple(self._vehicles[index] for index in sorted(self._vehicles))

    def vehicle_for_session_target(
        self, target_id: int
    ) -> GalleryVehicle | None:
        vehicle_id = self._session_target_to_vehicle.get(target_id)
        if vehicle_id is None:
            return None
        vehicle = self._vehicles.get(vehicle_id)
        if vehicle is None:
            del self._session_target_to_vehicle[target_id]
        return vehicle

    def session_target_for_vehicle_id(self, vehicle_id: int) -> int | None:
        vehicle_id = _validate_vehicle_id(vehicle_id)
        for target_id, mapped_vehicle_id in self._session_target_to_vehicle.items():
            if mapped_vehicle_id == vehicle_id:
                return target_id
        return None

    def attached_vehicle_ids(self) -> frozenset[int]:
        return frozenset(self._session_target_to_vehicle.values())

    def session_target_ids(self) -> tuple[int, ...]:
        """Return current runtime mapping keys for lifecycle coordination."""

        return tuple(self._session_target_to_vehicle)

    def attach_session_target(self, target_id: int, vehicle_id: int) -> bool:
        """Attach one current-session target to an existing Vehicle identity."""

        if not isinstance(target_id, int) or isinstance(target_id, bool) or target_id < 1:
            raise ValueError("target_id must be a positive integer")
        vehicle_id = _validate_vehicle_id(vehicle_id)
        if vehicle_id not in self._vehicles:
            raise KeyError(f"unknown Vehicle Gallery identity: {vehicle_id}")

        mapped_vehicle_id = self._session_target_to_vehicle.get(target_id)
        if mapped_vehicle_id is not None:
            if mapped_vehicle_id == vehicle_id:
                return True
            raise ValueError(
                f"session target {target_id} is already attached to "
                f"Vehicle {mapped_vehicle_id}"
            )

        mapped_target_id = self.session_target_for_vehicle_id(vehicle_id)
        if mapped_target_id is not None:
            raise ValueError(
                f"Vehicle {vehicle_id} is already attached to "
                f"session target {mapped_target_id}"
            )
        self._session_target_to_vehicle[target_id] = vehicle_id
        return True

    def detach_session_target(self, target_id: int) -> bool:
        return self._session_target_to_vehicle.pop(target_id, None) is not None

    def detach_all_session_targets(
        self, target_ids: Iterable[int] | None = None
    ) -> None:
        if target_ids is None:
            self._session_target_to_vehicle.clear()
            return
        for target_id in target_ids:
            self._session_target_to_vehicle.pop(target_id, None)

    def remove(self, vehicle_id: int) -> bool:
        vehicle_id = _validate_vehicle_id(vehicle_id)
        if vehicle_id not in self._vehicles:
            return False
        del self._vehicles[vehicle_id]
        for target_id, mapped_vehicle_id in list(
            self._session_target_to_vehicle.items()
        ):
            if mapped_vehicle_id == vehicle_id:
                del self._session_target_to_vehicle[target_id]
        return True

    def clear(self) -> None:
        """Clear persisted identities in memory without resetting the allocator."""

        self._vehicles.clear()
        self._session_target_to_vehicle.clear()


def _validate_vehicle_id(vehicle_id: int) -> int:
    if isinstance(vehicle_id, bool) or not isinstance(vehicle_id, int) or vehicle_id < 1:
        raise ValueError("vehicle_id must be a positive integer")
    return vehicle_id


def _copy_session_target_references(
    session_target: SessionTarget,
) -> list[np.ndarray]:
    if not session_target.reference_embeddings:
        raise ValueError(
            f"SessionTarget {session_target.target_id} has no reference embeddings"
        )
    return [
        _copy_vehicle_embedding(
            reference,
            context=(
                f"SessionTarget {session_target.target_id} "
                f"reference index {index}"
            ),
        )
        for index, reference in enumerate(session_target.reference_embeddings)
    ]


def _copy_gallery_vehicle(vehicle: GalleryVehicle) -> GalleryVehicle:
    vehicle_id = _validate_vehicle_id(vehicle.vehicle_id)
    if not isinstance(vehicle.label, str) or not vehicle.label.strip():
        raise ValueError(f"Vehicle {vehicle_id} has an invalid label")
    if not vehicle.reference_embeddings:
        raise ValueError(f"Vehicle {vehicle_id} has no reference embeddings")
    references = [
        _copy_vehicle_embedding(
            reference,
            context=f"vehicle_id={vehicle_id} reference index {index}",
        )
        for index, reference in enumerate(vehicle.reference_embeddings)
    ]
    centroid = _copy_vehicle_embedding(
        vehicle.centroid,
        context=f"vehicle_id={vehicle_id} centroid",
    )
    return GalleryVehicle(
        vehicle_id=vehicle_id,
        label=vehicle.label,
        reference_embeddings=references,
        centroid=centroid,
    )


def _copy_vehicle_embedding(embedding: np.ndarray, *, context: str) -> np.ndarray:
    array = np.asarray(embedding)
    if array.dtype != np.dtype(np.float32):
        raise ValueError(f"{context} must use dtype float32")
    if array.shape != (VEHICLE_EMBEDDING_DIMENSION,):
        raise ValueError(
            f"{context} must have shape ({VEHICLE_EMBEDDING_DIMENSION},), "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{context} contains non-finite values")
    norm = float(np.linalg.norm(array))
    if not np.isclose(norm, 1.0, atol=1e-3):
        raise ValueError(f"{context} must be L2 normalized")
    return np.asarray(array, dtype=np.float32).copy()
