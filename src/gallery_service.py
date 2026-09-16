"""Application-level coordination between TargetGallery and SQLite."""

from __future__ import annotations

from collections.abc import Iterable
from logging import getLogger

from .database import GalleryRepository, RepositoryError
from .config import GalleryEnrichmentConfig
from .gallery import GalleryPerson, TargetGallery
from .gallery_reference_bank import (
    normalized_centroid,
    update_persistent_reference_bank,
)
from .models import SessionTarget, TargetState
from .target_recovery import ReferenceUpdateEvent


LOGGER = getLogger(__name__)


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
        enrichment_config: GalleryEnrichmentConfig | None = None,
    ) -> None:
        self.gallery = gallery
        self.repository = repository
        self.enrichment_config = enrichment_config or GalleryEnrichmentConfig()
        self._explicitly_enrolled_target_ids: set[int] = set()
        self._post_recovery_blocked: set[int] = set()
        self._stable_active_frames: dict[int, int] = {}

    def load(self) -> tuple[GalleryPerson, ...]:
        """Restore people only; never restore old SessionTarget mappings."""

        people = self.repository.load_all()
        next_person_id = self.repository.load_next_person_id()
        self.gallery.restore_people(people, next_person_id=next_person_id)
        self._explicitly_enrolled_target_ids.clear()
        self._post_recovery_blocked.clear()
        self._stable_active_frames.clear()
        return self.gallery.all_people()

    def enroll(self, session_target: SessionTarget) -> GalleryPerson:
        """Enroll and persist a target, rolling back only a new person on error."""

        existing = self.gallery.person_for_session_target(session_target.target_id)
        if existing is not None:
            # Idempotent enrollment must not rewrite or remove the existing
            # persisted person, even if a repository failure is injected.
            self._mark_explicit_enrollment(session_target.target_id)
            return existing

        person = self.gallery.enroll(session_target)
        try:
            self.repository.save_person(person)
        except Exception:
            # This person and its mapping were created by this call.  Removing
            # it cannot damage a prior enrollment belonging to another target.
            self.gallery.remove(person.person_id)
            raise
        self._mark_explicit_enrollment(session_target.target_id)
        return person

    def update_runtime_state(
        self,
        targets: Iterable[SessionTarget],
        frame_index: int,
    ) -> None:
        """Advance the in-memory post-recovery enrichment cooldown."""

        targets_by_id = {target.target_id: target for target in targets}
        for target_id in list(self._explicitly_enrolled_target_ids):
            if target_id not in targets_by_id:
                self._explicitly_enrolled_target_ids.discard(target_id)
                self._post_recovery_blocked.discard(target_id)
                self._stable_active_frames.pop(target_id, None)

        for target_id in self._explicitly_enrolled_target_ids:
            target = targets_by_id[target_id]
            if target.state is not TargetState.ACTIVE:
                self._post_recovery_blocked.add(target_id)
                self._stable_active_frames[target_id] = 0
                continue

            if target_id not in self._post_recovery_blocked:
                continue
            # A target can remain ACTIVE during the TargetManager grace period
            # while its current Track is temporarily missing.  Those frames
            # are not stable evidence for persistent feature enrichment.
            if target.missing_frames > 0:
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
        """Persist one accepted reference using the persistent bank policy.

        ``event.reference_embeddings`` is the complete bounded runtime bank and
        intentionally is not copied into persistent storage.  Only the newly
        accepted embedding is considered as an incremental candidate so the
        runtime FIFO policy cannot evict useful long-term Gallery references.
        """

        target_id = event.target_id
        if target_id not in self._explicitly_enrolled_target_ids:
            LOGGER.debug(
                "GALLERY_ENRICHMENT_SKIPPED target=%d reason=not_explicitly_enrolled",
                target_id,
            )
            return False
        if event.target_state is not TargetState.ACTIVE:
            LOGGER.debug(
                "GALLERY_ENRICHMENT_SKIPPED target=%d reason=target_not_active",
                target_id,
            )
            return False
        if target_id in self._post_recovery_blocked:
            LOGGER.debug(
                "GALLERY_ENRICHMENT_SKIPPED target=%d reason=post_recovery_cooldown",
                target_id,
            )
            return False

        person = self.gallery.person_for_session_target(target_id)
        if person is None:
            return False

        try:
            persistent_references, changed = update_persistent_reference_bank(
                person.reference_embeddings,
                event.accepted_embedding,
                max_reference_embeddings=(
                    self.enrichment_config.max_reference_embeddings
                ),
                duplicate_similarity_threshold=(
                    self.enrichment_config.duplicate_similarity_threshold
                ),
            )
            if not changed:
                LOGGER.debug(
                    "GALLERY_ENRICHMENT_SKIPPED target=%d person=%d "
                    "reason=duplicate_or_not_more_diverse",
                    target_id,
                    person.person_id,
                )
                return False
            snapshot = self.gallery.feature_snapshot(
                person.person_id,
                persistent_references,
                normalized_centroid(persistent_references),
            )
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.error(
                "GALLERY_ENRICHMENT_REJECTED target=%d person=%d reason=%s",
                target_id,
                person.person_id,
                exc,
            )
            return False

        try:
            self.repository.update_person_features(
                snapshot.person_id,
                snapshot.reference_embeddings,
                snapshot.centroid,
            )
        except Exception:
            LOGGER.exception(
                "GALLERY_ENRICHMENT_FAILED target=%d person=%d",
                target_id,
                person.person_id,
            )
            return False

        try:
            self.gallery.apply_feature_snapshot(snapshot)
        except Exception:
            LOGGER.exception(
                "GALLERY_ENRICHMENT_CONSISTENCY_ERROR target=%d person=%d",
                target_id,
                person.person_id,
            )
            try:
                persisted = self.repository.load_person(person.person_id)
                if persisted is None:
                    raise RepositoryError(
                        f"updated person disappeared: {person.person_id}"
                    )
                self.gallery.restore_person(persisted)
            except Exception:
                LOGGER.exception(
                    "GALLERY_ENRICHMENT_CONSISTENCY_RESTORE_FAILED person=%d",
                    person.person_id,
                )
            return False

        LOGGER.debug(
            "GALLERY_ENRICHED person=%d target=%d references=%d",
            snapshot.person_id,
            target_id,
            len(snapshot.reference_embeddings),
        )
        return True

    def enrich_reference_updates(
        self,
        events: Iterable[ReferenceUpdateEvent],
    ) -> int:
        """Apply a batch of already accepted, one-time reference events."""

        return sum(self.enrich_reference_update(event) for event in events)

    def _mark_explicit_enrollment(self, target_id: int) -> None:
        if target_id in self._explicitly_enrolled_target_ids:
            return
        self._explicitly_enrolled_target_ids.add(target_id)
        # Initial G enrollment is explicitly exempt from the post-recovery
        # cooldown.  A target that later becomes LOST will be blocked again by
        # update_runtime_state().
        self._post_recovery_blocked.discard(target_id)
        self._stable_active_frames[target_id] = (
            self.enrichment_config.post_recovery_stable_frames
        )

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
