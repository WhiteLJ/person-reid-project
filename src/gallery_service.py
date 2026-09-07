"""Application-level coordination between TargetGallery and SQLite."""

from __future__ import annotations

from .database import GalleryRepository, RepositoryError
from .gallery import GalleryPerson, TargetGallery
from .models import SessionTarget


class GalleryPersistenceService:
    """Keep the in-memory Gallery and its Repository changes coordinated.

    TargetGallery remains a persistence-agnostic domain object.  This service
    owns the ordering needed by the application when a disk operation is
    involved.
    """

    def __init__(
        self,
        gallery: TargetGallery,
        repository: GalleryRepository,
    ) -> None:
        self.gallery = gallery
        self.repository = repository

    def load(self) -> tuple[GalleryPerson, ...]:
        """Restore people only; never restore old SessionTarget mappings."""

        people = self.repository.load_all()
        next_person_id = self.repository.load_next_person_id()
        self.gallery.restore_people(people, next_person_id=next_person_id)
        return self.gallery.all_people()

    def enroll(self, session_target: SessionTarget) -> GalleryPerson:
        """Enroll and persist a target, rolling back only a new person on error."""

        existing = self.gallery.person_for_session_target(session_target.target_id)
        if existing is not None:
            # Idempotent enrollment must not rewrite or remove the existing
            # persisted person, even if a repository failure is injected.
            return existing

        person = self.gallery.enroll(session_target)
        try:
            self.repository.save_person(person)
        except Exception:
            # This person and its mapping were created by this call.  Removing
            # it cannot damage a prior enrollment belonging to another target.
            self.gallery.remove(person.person_id)
            raise
        return person

    def remove(self, person_id: int) -> bool:
        """Delete a person from disk first, then remove it from memory."""

        in_memory = self.gallery.get(person_id)
        deleted = self.repository.delete_person(person_id)
        if not deleted:
            if in_memory is not None:
                raise RepositoryError(
                    f"Gallery memory/database mismatch for person_id={person_id}"
                )
            return False
        self.gallery.remove(person_id)
        return True

    def clear(self) -> None:
        """Clear persisted people first, then clear the in-memory Gallery."""

        self.repository.clear()
        self.gallery.clear()
