"""In-memory session target state and Track-to-target bindings for MVP-5."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from logging import getLogger

import numpy as np

from .models import SessionTarget, TargetState, Track
from .reid import cosine_similarity, normalize_embedding


LOGGER = getLogger(__name__)


class TargetManager:
    """Manage temporary session identities independently from BoT-SORT IDs.

    ``targets`` is the authoritative state.  The ``selected_track_ids``
    property is retained as a derived compatibility view for visualization and
    older callers; recovery never mutates that set directly.
    """

    def __init__(self) -> None:
        self.targets: dict[int, SessionTarget] = {}
        self._next_target_id = 1
        self.target_lost_count = 0
        self.target_recovered_count = 0

    @property
    def selected_track_ids(self) -> set[int]:
        """Return currently visible Track IDs bound to ACTIVE session targets."""

        return {
            target.current_track_id
            for target in self.targets.values()
            if target.state is TargetState.ACTIVE
            and target.current_track_id is not None
        }

    def select(
        self,
        track: Track,
        embedding: np.ndarray,
        frame_index: int = 0,
    ) -> SessionTarget:
        """Create an in-session target from a valid initial ReID embedding.

        Selecting an already-bound Track is idempotent and does not create a
        second session target.  A target without an initial embedding is not
        allowed because it could never be recovered safely by ReID.
        """

        existing = self.target_for_track(track.track_id)
        if existing is not None:
            return existing

        reference = _normalize_single_embedding(embedding)
        target = SessionTarget(
            target_id=self._next_target_id,
            current_track_id=track.track_id,
            last_track_id=track.track_id,
            state=TargetState.ACTIVE,
            reference_embeddings=[reference.copy()],
            centroid=reference.copy(),
            missing_frames=0,
            last_reference_frame=frame_index,
        )
        self.targets[target.target_id] = target
        self._next_target_id += 1
        return target

    def select_from_reference_bank(
        self,
        track: Track,
        candidate_embedding: np.ndarray,
        reference_embeddings: Sequence[np.ndarray],
        centroid: np.ndarray,
        frame_index: int,
        max_reference_embeddings: int,
        *,
        include_candidate: bool = True,
    ) -> SessionTarget:
        """Create a target from copied Gallery references and a live crop.

        This method knows nothing about Gallery or persistence.  It is the
        generic bridge used when an existing in-memory identity is recognized:
        all incoming arrays are normalized and copied, and the bounded runtime
        bank is independent from the Gallery arrays.  Automatic Gallery
        recognition can set ``include_candidate=False`` so its initial runtime
        references contain only trusted Gallery features.  Later observations
        then enter through the normal ACTIVE reference-update policy.
        """

        if max_reference_embeddings < 1:
            raise ValueError("max_reference_embeddings must be positive")
        existing = self.target_for_track(track.track_id)
        if existing is not None:
            return existing

        normalized_centroid = _normalize_single_embedding(centroid)
        references: list[np.ndarray] = []
        for index, reference in enumerate(reference_embeddings):
            try:
                normalized = _normalize_single_embedding(reference)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"reference embedding at index {index} is invalid"
                ) from exc
            if normalized.shape != normalized_centroid.shape:
                raise ValueError("reference and centroid dimensions must match")
            references.append(normalized.copy())
        if not references:
            raise ValueError("reference_embeddings cannot be empty")
        candidate: np.ndarray | None = None
        if include_candidate:
            candidate = _normalize_single_embedding(candidate_embedding)
            if candidate.shape != normalized_centroid.shape:
                raise ValueError("candidate and centroid dimensions must match")

        # Keep the persisted references and optionally this current
        # observation. When the bank is full, retain the newest bounded suffix.
        # The centroid is recomputed from the copied runtime bank, never
        # aliased to Gallery.
        if candidate is not None:
            references.append(candidate.copy())
        if len(references) > max_reference_embeddings:
            references = references[-max_reference_embeddings:]
        runtime_centroid = _centroid(references)
        target = SessionTarget(
            target_id=self._next_target_id,
            current_track_id=track.track_id,
            last_track_id=track.track_id,
            state=TargetState.ACTIVE,
            reference_embeddings=references,
            centroid=runtime_centroid,
            missing_frames=0,
            last_reference_frame=frame_index,
        )
        self.targets[target.target_id] = target
        self._next_target_id += 1
        return target

    def deselect(self, track: Track) -> bool:
        """Delete the entire session target currently bound to ``track``."""

        return self.remove_by_track_id(track.track_id)

    def remove_by_track_id(self, track_id: int) -> bool:
        """Remove a visible target binding and its complete reference bank."""

        target = self.target_for_track(track_id)
        if target is None:
            return False
        del self.targets[target.target_id]
        return True

    def clear(self) -> None:
        """Delete all session targets and their in-memory embeddings."""

        self.targets.clear()

    def is_selected(self, track_id: int) -> bool:
        return track_id in self.selected_track_ids

    def selected_tracks(self, tracks: list[Track]) -> list[Track]:
        """Return visible Tracks bound to ACTIVE session targets."""

        selected_ids = self.selected_track_ids
        return [track for track in tracks if track.track_id in selected_ids]

    def target_for_track(self, track_id: int) -> SessionTarget | None:
        for target in self.targets.values():
            if (
                target.state is TargetState.ACTIVE
                and target.current_track_id == track_id
            ):
                return target
        return None

    def active_targets(self) -> tuple[SessionTarget, ...]:
        return tuple(
            target
            for target in self.targets.values()
            if target.state is TargetState.ACTIVE
        )

    def lost_targets(self) -> tuple[SessionTarget, ...]:
        return tuple(
            target
            for target in self.targets.values()
            if target.state is TargetState.LOST
        )

    def active_track_ids(self) -> set[int]:
        return {
            target.current_track_id
            for target in self.active_targets()
            if target.current_track_id is not None
        }

    def recovery_candidates(self, tracks: Iterable[Track]) -> list[Track]:
        """Return tracks not currently owned by an ACTIVE target."""

        active_ids = self.active_track_ids()
        return [track for track in tracks if track.track_id not in active_ids]

    def update_visibility(
        self,
        tracks: Collection[Track],
        lost_grace_frames: int,
        frame_index: int,
    ) -> None:
        """Advance ACTIVE target visibility and transition stable losses.

        A missing Track increments a target's grace counter.  Its original
        binding remains in ``last_track_id`` when the target becomes LOST.
        Returning before the grace limit keeps the same session target active.
        """

        if lost_grace_frames < 1:
            raise ValueError("lost_grace_frames must be positive")
        visible_ids = {track.track_id for track in tracks}
        for target in self.active_targets():
            current_track_id = target.current_track_id
            if current_track_id is None:
                continue
            if current_track_id in visible_ids:
                target.missing_frames = 0
                target.last_track_id = current_track_id
                continue

            target.missing_frames += 1
            if target.missing_frames < lost_grace_frames:
                continue

            old_track_id = current_track_id
            target.last_track_id = old_track_id
            target.current_track_id = None
            target.state = TargetState.LOST
            target.last_recovery_frame = None
            self.target_lost_count += 1
            LOGGER.info(
                "TARGET_LOST target=%d old_track_id=%d missing_frames=%d frame=%d",
                target.target_id,
                old_track_id,
                target.missing_frames,
                frame_index,
            )

    def reference_update_due(
        self, target: SessionTarget, frame_index: int, interval: int
    ) -> bool:
        if interval < 1:
            raise ValueError("reference update interval must be positive")
        if target.last_reference_frame is None:
            return True
        return frame_index - target.last_reference_frame >= interval

    def mark_reference_attempt(self, target: SessionTarget, frame_index: int) -> None:
        target.last_reference_frame = frame_index

    def add_reference(
        self,
        target_id: int,
        embedding: np.ndarray,
        frame_index: int,
        max_reference_embeddings: int,
        reference_update_threshold: float,
    ) -> bool:
        """Add a consistent ACTIVE reference, bounded and FIFO.

        The candidate is compared with the normalized centroid before it is
        stored.  This gate prevents a Track ID switch from poisoning the
        target's feature bank.
        """

        if max_reference_embeddings < 1:
            raise ValueError("max_reference_embeddings must be positive")
        if not 0.0 < reference_update_threshold <= 1.0:
            raise ValueError("reference_update_threshold must be in (0, 1]")
        target = self.targets.get(target_id)
        if target is None or target.state is not TargetState.ACTIVE:
            return False

        candidate = _normalize_single_embedding(embedding)
        if target.centroid.size == 0:
            consistent = True
        else:
            try:
                consistency = cosine_similarity(target.centroid, candidate)
            except ValueError:
                LOGGER.warning(
                    "REFERENCE_UPDATE_REJECTED target=%d reason=embedding_dimension",
                    target_id,
                )
                return False
            consistent = consistency >= reference_update_threshold
            if not consistent:
                LOGGER.info(
                    "REFERENCE_UPDATE_REJECTED target=%d similarity=%.4f threshold=%.4f",
                    target_id,
                    consistency,
                    reference_update_threshold,
                )
        if not consistent:
            return False

        target.reference_embeddings.append(candidate.copy())
        if len(target.reference_embeddings) > max_reference_embeddings:
            del target.reference_embeddings[
                : len(target.reference_embeddings) - max_reference_embeddings
            ]
        target.centroid = _centroid(target.reference_embeddings)
        target.last_reference_frame = frame_index
        return True

    def recovery_due(
        self, target: SessionTarget, frame_index: int, interval: int
    ) -> bool:
        if interval < 1:
            raise ValueError("recovery interval must be positive")
        if target.last_recovery_frame is None:
            return True
        return frame_index - target.last_recovery_frame >= interval

    def mark_recovery_attempt(self, target: SessionTarget, frame_index: int) -> None:
        target.last_recovery_frame = frame_index

    def recover(
        self,
        target_id: int,
        track: Track,
        embedding: np.ndarray,
        similarity: float,
        frame_index: int,
        max_reference_embeddings: int,
        reference_update_threshold: float,
    ) -> SessionTarget:
        """Bind a new Track to the existing SessionTarget after a safe match."""

        target = self.targets.get(target_id)
        if target is None:
            raise KeyError(f"unknown session target: {target_id}")
        if target.state is not TargetState.LOST:
            raise ValueError(f"target {target_id} is not LOST")
        if self.target_for_track(track.track_id) is not None:
            raise ValueError(
                f"track {track.track_id} is already active for another target"
            )

        old_track_id = target.last_track_id
        target.current_track_id = track.track_id
        target.last_track_id = track.track_id
        target.state = TargetState.ACTIVE
        target.missing_frames = 0
        target.last_recovery_frame = frame_index
        self.add_reference(
            target_id,
            embedding,
            frame_index,
            max_reference_embeddings,
            reference_update_threshold,
        )
        self.target_recovered_count += 1
        LOGGER.info(
            "TARGET_RECOVERED target=%d old_track_id=%s new_track_id=%d similarity=%.4f",
            target_id,
            old_track_id,
            track.track_id,
            similarity,
        )
        return target


def _normalize_single_embedding(embedding: np.ndarray) -> np.ndarray:
    normalized = normalize_embedding(embedding)
    if normalized.ndim != 1:
        raise ValueError("target embedding must have shape (D,)")
    return normalized.astype(np.float32, copy=True)


def _centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        return np.empty((0,), dtype=np.float32)
    return _normalize_single_embedding(np.mean(np.stack(embeddings), axis=0))
