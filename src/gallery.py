"""In-memory TargetGallery for explicit MVP-6 enrollment."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .models import SessionTarget
from .reid import normalize_embedding


def _empty_embedding() -> np.ndarray:
    return np.empty((0,), dtype=np.float32)


def format_person_id(person_id: int) -> str:
    """Format an in-memory Gallery person ID for display."""

    if person_id < 1:
        raise ValueError("person_id must be positive")
    return f"P{person_id:03d}"


@dataclass
class GalleryPerson:
    """A user-enrolled in-memory Gallery identity."""

    person_id: int
    label: str
    reference_embeddings: list[np.ndarray] = field(default_factory=list)
    centroid: np.ndarray = field(default_factory=_empty_embedding)


class TargetGallery:
    """Own GalleryPerson records and their SessionTarget associations.

    This class intentionally has no persistence and performs no ReID
    inference.  Session-target associations are lifecycle-managed explicitly:
    detaching a target never deletes its GalleryPerson, while removing a
    GalleryPerson removes every association pointing to it.
    """

    def __init__(self) -> None:
        self._people: dict[int, GalleryPerson] = {}
        self._session_target_to_person: dict[int, int] = {}
        self._next_person_id = 1

    def enroll(self, session_target: SessionTarget) -> GalleryPerson:
        """Snapshot a valid SessionTarget into the Gallery, idempotently."""

        existing_person_id = self._session_target_to_person.get(
            session_target.target_id
        )
        if existing_person_id is not None:
            existing = self._people.get(existing_person_id)
            if existing is not None:
                return existing
            # Repair an impossible stale mapping rather than associating a new
            # person with an invalid old record.
            del self._session_target_to_person[session_target.target_id]

        references = _copy_reference_bank(session_target)
        centroid = _copy_centroid(session_target, references)
        person_id = self._next_person_id
        self._next_person_id += 1
        person = GalleryPerson(
            person_id=person_id,
            label=f"Target {format_person_id(person_id)}",
            reference_embeddings=references,
            centroid=centroid,
        )
        self._people[person_id] = person
        self._session_target_to_person[session_target.target_id] = person_id
        return person

    def get(self, person_id: int) -> GalleryPerson | None:
        return self._people.get(person_id)

    def all_people(self) -> tuple[GalleryPerson, ...]:
        """Return people in stable person_id order."""

        return tuple(self._people[person_id] for person_id in sorted(self._people))

    def remove(self, person_id: int) -> bool:
        """Remove a GalleryPerson and every mapping pointing to it."""

        if person_id not in self._people:
            return False
        del self._people[person_id]
        for target_id, mapped_person_id in list(
            self._session_target_to_person.items()
        ):
            if mapped_person_id == person_id:
                del self._session_target_to_person[target_id]
        return True

    def clear(self) -> None:
        """Clear Gallery people and associations without touching targets."""

        self._people.clear()
        self._session_target_to_person.clear()

    def person_for_session_target(self, target_id: int) -> GalleryPerson | None:
        """Return the Gallery person associated with one session target."""

        person_id = self._session_target_to_person.get(target_id)
        if person_id is None:
            return None
        person = self._people.get(person_id)
        if person is None:
            del self._session_target_to_person[target_id]
        return person

    def detach_session_target(self, target_id: int) -> bool:
        """Remove one target mapping while retaining its GalleryPerson."""

        return self._session_target_to_person.pop(target_id, None) is not None

    def detach_all_session_targets(self, target_ids: Iterable[int] | None = None) -> None:
        """Remove target mappings only; GalleryPerson records remain."""

        if target_ids is None:
            self._session_target_to_person.clear()
            return
        for target_id in target_ids:
            self._session_target_to_person.pop(target_id, None)


def _copy_reference_bank(session_target: SessionTarget) -> list[np.ndarray]:
    if not session_target.reference_embeddings:
        raise ValueError(
            f"SessionTarget {session_target.target_id} has no reference embeddings"
        )

    references: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for index, reference in enumerate(session_target.reference_embeddings):
        try:
            normalized = normalize_embedding(reference)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"SessionTarget {session_target.target_id} has invalid reference "
                f"embedding at index {index}"
            ) from exc
        if normalized.ndim != 1:
            raise ValueError(
                f"SessionTarget {session_target.target_id} reference embeddings "
                "must have shape (D,)"
            )
        if expected_shape is None:
            expected_shape = normalized.shape
        elif normalized.shape != expected_shape:
            raise ValueError(
                f"SessionTarget {session_target.target_id} reference dimensions "
                "do not match"
            )
        references.append(normalized.astype(np.float32, copy=True))
    return references


def _copy_centroid(
    session_target: SessionTarget,
    references: list[np.ndarray],
) -> np.ndarray:
    try:
        centroid = normalize_embedding(session_target.centroid)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"SessionTarget {session_target.target_id} has an invalid centroid"
        ) from exc
    if centroid.ndim != 1 or centroid.shape != references[0].shape:
        raise ValueError(
            f"SessionTarget {session_target.target_id} centroid dimension does not match references"
        )
    return centroid.astype(np.float32, copy=True)
