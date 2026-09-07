"""MVP-5 session-target reference updates and conservative ReID recovery."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from logging import getLogger

import numpy as np

from .config import ReIDConfig, ReIDQualityConfig, ReIDRecoveryConfig
from .models import SessionTarget, TargetState, Track
from .reid import ReIDExtractor, cosine_similarity
from .reid_frame_cache import ReIDFrameCache
from .reid_quality import ReIDQualityResult, assess_reid_quality
from .target_manager import TargetManager


LOGGER = getLogger(__name__)


@dataclass(frozen=True)
class RecoveryCandidate:
    """A current unbound Track and its already-extracted ReID embedding."""

    track: Track
    embedding: np.ndarray


@dataclass(frozen=True)
class RecoveryMatch:
    """One accepted one-to-one LOST-target recovery assignment."""

    target_id: int
    candidate: RecoveryCandidate
    similarity: float


@dataclass
class _PendingRecovery:
    """A valid recovery proposal awaiting repeated confirmation."""

    hits: int
    last_attempt_frame: int


@dataclass(frozen=True)
class ReferenceUpdateEvent:
    """One accepted ACTIVE reference-bank update for downstream enrichment."""

    target_id: int
    frame_index: int
    reference_embeddings: tuple[np.ndarray, ...]
    centroid: np.ndarray
    target_state: TargetState = TargetState.ACTIVE


def recovery_score(target: SessionTarget, embedding: np.ndarray) -> float:
    """Score a candidate against the normalized centroid, not max bank score.

    A centroid-based score makes one anomalous historical embedding unable to
    force recovery by itself.  The reference bank remains available for
    stable centroid construction and bounded history, while the conservative
    MVP-5 decision uses only the normalized centroid similarity.
    """

    if target.centroid.size == 0:
        raise ValueError(f"target {target.target_id} has no reference centroid")
    return cosine_similarity(target.centroid, embedding)


def assign_recovery_matches(
    lost_targets: Sequence[SessionTarget],
    candidates: Sequence[RecoveryCandidate],
    recovery_threshold: float,
    recovery_margin: float,
) -> list[RecoveryMatch]:
    """Return conservative one-to-one target/candidate assignments.

    Both sides must be confident: the best score must clear the threshold and,
    when a second choice exists, beat that second-best score by the configured
    margin.  With no second candidate/target, that side's margin check passes
    automatically.  Accepted pairs are sorted by score and greedily claimed,
    which is deterministic, dependency-free, and favors avoiding false
    recovery over maximizing the number of assignments.
    """

    if not 0.0 < recovery_threshold <= 1.0:
        raise ValueError("recovery_threshold must be in (0, 1]")
    if recovery_margin < 0.0:
        raise ValueError("recovery_margin must be non-negative")

    target_list = [target for target in lost_targets if target.state is TargetState.LOST]
    candidate_list = list(candidates)
    if not target_list or not candidate_list:
        return []

    scores: dict[tuple[int, int], float] = {}
    for target_index, target in enumerate(target_list):
        for candidate_index, candidate in enumerate(candidate_list):
            try:
                score = recovery_score(target, candidate.embedding)
            except ValueError:
                continue
            scores[(target_index, candidate_index)] = score

    if not scores:
        return []

    target_rankings: dict[int, list[tuple[float, int]]] = {}
    for target_index in range(len(target_list)):
        ranking = sorted(
            (
                score,
                candidate_index,
            )
            for (row, candidate_index), score in scores.items()
            if row == target_index
        )
        target_rankings[target_index] = list(reversed(ranking))

    candidate_rankings: dict[int, list[tuple[float, int]]] = {}
    for candidate_index in range(len(candidate_list)):
        ranking = sorted(
            (
                score,
                target_index,
            )
            for (target_index, column), score in scores.items()
            if column == candidate_index
        )
        candidate_rankings[candidate_index] = list(reversed(ranking))

    eligible: list[tuple[float, int, int]] = []
    for target_index, ranking in target_rankings.items():
        if not ranking:
            continue
        best_score, best_candidate_index = ranking[0]
        if best_score < recovery_threshold:
            continue
        target_margin_ok = (
            len(ranking) == 1
            or best_score - ranking[1][0] >= recovery_margin
        )
        if not target_margin_ok:
            continue

        candidate_ranking = candidate_rankings[best_candidate_index]
        candidate_margin_ok = (
            len(candidate_ranking) == 1
            or best_score - candidate_ranking[1][0] >= recovery_margin
        )
        if candidate_margin_ok:
            eligible.append((best_score, target_index, best_candidate_index))

    eligible.sort(
        key=lambda item: (
            -item[0],
            target_list[item[1]].target_id,
            candidate_list[item[2]].track.track_id,
        )
    )
    used_targets: set[int] = set()
    used_candidates: set[int] = set()
    matches: list[RecoveryMatch] = []
    for score, target_index, candidate_index in eligible:
        if target_index in used_targets or candidate_index in used_candidates:
            continue
        used_targets.add(target_index)
        used_candidates.add(candidate_index)
        matches.append(
            RecoveryMatch(
                target_id=target_list[target_index].target_id,
                candidate=candidate_list[candidate_index],
                similarity=score,
            )
        )
    return matches


class TargetRecoveryCoordinator:
    """Coordinate conservative, quality-gated ReID recovery.

    Recovery is deliberately confirmation-based in crowded scenes.  A valid
    candidate match is a pending proposal first; only repeated valid attempts
    for the same target/Track pair can change a LOST target back to ACTIVE.
    """

    def __init__(
        self,
        target_manager: TargetManager,
        reid_extractor: ReIDExtractor,
        reid_config: ReIDConfig,
        recovery_config: ReIDRecoveryConfig,
        embedding_cache: ReIDFrameCache | None = None,
        quality_config: ReIDQualityConfig | None = None,
        person_class_id: int = 0,
    ) -> None:
        self.target_manager = target_manager
        self.reid_extractor = reid_extractor
        self.reid_config = reid_config
        self.recovery_config = recovery_config
        self.embedding_cache = embedding_cache
        self.quality_config = quality_config or ReIDQualityConfig()
        self.person_class_id = person_class_id
        self.last_recovered_track_ids: frozenset[int] = frozenset()
        self._track_ages: dict[int, int] = {}
        self._last_seen_frame: dict[int, int] = {}
        self._pending: dict[tuple[int, int], _PendingRecovery] = {}
        self.recovery_attempted_count = 0
        self.recovery_pending_count = 0
        self.recovery_accepted_count = 0
        self.quality_rejected_count = 0
        self.reid_batch_count = 0
        self._reference_updates: list[ReferenceUpdateEvent] = []

    @property
    def pending(self) -> dict[tuple[int, int], int]:
        """Return pending recovery hits as a diagnostic/test snapshot."""

        return {key: value.hits for key, value in self._pending.items()}

    @property
    def track_ages(self) -> dict[int, int]:
        return dict(self._track_ages)

    def drain_reference_updates(self) -> tuple[ReferenceUpdateEvent, ...]:
        """Consume accepted reference events exactly once."""

        events = tuple(self._reference_updates)
        self._reference_updates.clear()
        return events

    def select_from_track(
        self,
        frame: np.ndarray,
        track: Track,
        frame_index: int,
    ) -> SessionTarget | None:
        """Create a recoverable target from one frozen-frame Track crop."""

        existing = self.target_manager.target_for_track(track.track_id)
        if existing is not None:
            LOGGER.info(
                "TARGET_SELECTION_IGNORED track=%d target=%d reason=already_selected",
                track.track_id,
                existing.target_id,
            )
            return existing

        quality = self._quality(frame, track, (track,))
        if not quality.accepted or quality.crop is None:
            LOGGER.info(
                "TARGET_SELECTION_REJECTED track=%d reason=%s",
                track.track_id,
                quality.reason or "quality_rejected",
            )
            return None

        embedding = self.reid_extractor.extract(quality.crop)
        target = self.target_manager.select(track, embedding, frame_index)
        LOGGER.info(
            "TARGET_SELECTED target=%d track_id=%d reference_count=%d",
            target.target_id,
            track.track_id,
            len(target.reference_embeddings),
        )
        return target

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        frame_index: int,
    ) -> list[RecoveryMatch]:
        """Update visibility and run only due, quality-valid ReID work."""

        self.last_recovered_track_ids = frozenset()
        self._update_track_ages(tracks, frame_index)
        if self.embedding_cache is not None:
            self.embedding_cache.begin_frame(frame_index)

        self._expire_pending(frame_index)

        self.target_manager.update_visibility(
            tracks,
            lost_grace_frames=self.recovery_config.lost_grace_frames,
            frame_index=frame_index,
        )

        embedding_jobs: list[tuple[str, int, Track, np.ndarray]] = []
        visible_ids = {track.track_id for track in tracks}

        for target in self.target_manager.active_targets():
            if target.current_track_id not in visible_ids:
                continue
            if not self.target_manager.reference_update_due(
                target,
                frame_index,
                self.recovery_config.reference_update_interval_frames,
            ):
                continue
            self.target_manager.mark_reference_attempt(target, frame_index)
            track = next(
                track
                for track in tracks
                if track.track_id == target.current_track_id
            )
            quality = self._quality(frame, track, tracks)
            if not quality.accepted or quality.crop is None:
                self.quality_rejected_count += 1
                LOGGER.debug(
                    "REID_QUALITY_REJECTED kind=reference target=%d track=%d reason=%s",
                    target.target_id,
                    track.track_id,
                    quality.reason or "unknown",
                )
                continue
            embedding_jobs.append(("reference", target.target_id, track, quality.crop))

        due_lost_targets = [
            target
            for target in self.target_manager.lost_targets()
            if self.target_manager.recovery_due(
                target,
                frame_index,
                self.recovery_config.recovery_interval_frames,
            )
        ]
        for target in due_lost_targets:
            self.target_manager.mark_recovery_attempt(target, frame_index)

        candidate_tracks = [
            track
            for track in self.target_manager.recovery_candidates(tracks)
            if track.class_id == self.person_class_id
            and self._track_ages.get(track.track_id, 0)
            >= self.recovery_config.recovery_min_track_age_frames
        ]
        valid_candidate_ids: set[int] = set()
        for track in (candidate_tracks if due_lost_targets else ()):
            quality = self._quality(frame, track, tracks)
            if not quality.accepted or quality.crop is None:
                self.quality_rejected_count += 1
                LOGGER.debug(
                    "REID_QUALITY_REJECTED kind=recovery track=%d reason=%s",
                    track.track_id,
                    quality.reason or "unknown",
                )
                continue
            valid_candidate_ids.add(track.track_id)
            embedding_jobs.append(("candidate", track.track_id, track, quality.crop))

        if due_lost_targets and valid_candidate_ids:
            self.recovery_attempted_count += len(due_lost_targets)
            LOGGER.debug(
                "TARGET_RECOVERY_ATTEMPT targets=%d candidates=%d frame=%d",
                len(due_lost_targets),
                len(valid_candidate_ids),
                frame_index,
            )

        if not embedding_jobs:
            return []

        resolved_jobs = self._ensure_embeddings(embedding_jobs, frame_index)

        recovery_candidates: list[RecoveryCandidate] = []
        for job, embedding in resolved_jobs:
            kind, owner_id, track, _crop = job
            if kind == "reference":
                accepted = self.target_manager.add_reference(
                    owner_id,
                    embedding,
                    frame_index,
                    self.recovery_config.max_reference_embeddings,
                    self.recovery_config.reference_update_threshold,
                )
                if accepted:
                    target = self.target_manager.targets.get(owner_id)
                    if target is not None:
                        self._reference_updates.append(
                            ReferenceUpdateEvent(
                                target_id=target.target_id,
                                frame_index=frame_index,
                                reference_embeddings=tuple(
                                    reference.copy()
                                    for reference in target.reference_embeddings
                                ),
                                centroid=target.centroid.copy(),
                                target_state=target.state,
                            )
                        )
            elif track is not None:
                recovery_candidates.append(RecoveryCandidate(track, embedding.copy()))

        if not due_lost_targets or not recovery_candidates:
            return []

        matches = assign_recovery_matches(
            due_lost_targets,
            recovery_candidates,
            recovery_threshold=self.recovery_config.recovery_threshold,
            recovery_margin=self.recovery_config.recovery_margin,
        )
        accepted_pairs = {
            (match.target_id, match.candidate.track.track_id) for match in matches
        }
        due_target_ids = {target.target_id for target in due_lost_targets}
        for key in list(self._pending):
            if key[0] in due_target_ids and key not in accepted_pairs:
                del self._pending[key]

        recovered_matches: list[RecoveryMatch] = []
        for match in matches:
            key = (match.target_id, match.candidate.track.track_id)
            pending = self._pending.get(key)
            hits = 1 if pending is None else pending.hits + 1
            if hits < self.recovery_config.recovery_confirmation_hits:
                self._pending[key] = _PendingRecovery(
                    hits=hits,
                    last_attempt_frame=frame_index,
                )
                self.recovery_pending_count += 1
                LOGGER.debug(
                    "TARGET_RECOVERY_PENDING target=%d track=%d hits=%d/%d similarity=%.4f",
                    match.target_id,
                    match.candidate.track.track_id,
                    hits,
                    self.recovery_config.recovery_confirmation_hits,
                    match.similarity,
                )
                continue
            self.target_manager.recover(
                target_id=match.target_id,
                track=match.candidate.track,
                embedding=match.candidate.embedding,
                similarity=match.similarity,
                frame_index=frame_index,
                max_reference_embeddings=self.recovery_config.max_reference_embeddings,
                reference_update_threshold=self.recovery_config.reference_update_threshold,
            )
            self._pending.pop(key, None)
            self.recovery_accepted_count += 1
            recovered_matches.append(match)
        self.last_recovered_track_ids = frozenset(
            match.candidate.track.track_id for match in recovered_matches
        )
        return recovered_matches

    def _update_track_ages(
        self,
        tracks: Sequence[Track],
        frame_index: int,
    ) -> None:
        visible_ids = {track.track_id for track in tracks}
        for track_id in list(self._track_ages):
            if track_id not in visible_ids:
                del self._track_ages[track_id]
                self._last_seen_frame.pop(track_id, None)
                for key in [key for key in self._pending if key[1] == track_id]:
                    del self._pending[key]

        for track in tracks:
            previous_frame = self._last_seen_frame.get(track.track_id)
            if previous_frame == frame_index - 1:
                self._track_ages[track.track_id] = (
                    self._track_ages.get(track.track_id, 0) + 1
                )
            else:
                self._track_ages[track.track_id] = 1
            self._last_seen_frame[track.track_id] = frame_index

    def _expire_pending(self, frame_index: int) -> None:
        max_age = self.recovery_config.recovery_pending_max_age_frames
        for key, pending in list(self._pending.items()):
            if frame_index - pending.last_attempt_frame > max_age:
                del self._pending[key]

    def _ensure_embeddings(
        self,
        jobs: Sequence[tuple[str, int, Track, np.ndarray]],
        frame_index: int,
    ) -> list[tuple[tuple[str, int, Track, np.ndarray], np.ndarray]]:
        resolved: list[tuple[tuple[str, int, Track, np.ndarray], np.ndarray] | None] = [
            None
        ] * len(jobs)
        missing_indices: list[int] = []
        missing_crops: list[np.ndarray] = []
        for index, job in enumerate(jobs):
            track = job[2]
            cached = (
                self.embedding_cache.get(track.track_id, frame_index)
                if self.embedding_cache is not None
                else None
            )
            if cached is not None:
                resolved[index] = (job, cached)
            else:
                missing_indices.append(index)
                missing_crops.append(job[3])

        if missing_crops:
            self.reid_batch_count += 1
            embeddings = np.asarray(
                self.reid_extractor.extract_batch(missing_crops),
                dtype=np.float32,
            )
            if embeddings.ndim != 2 or embeddings.shape[0] != len(missing_crops):
                raise ValueError("ReID batch output does not match recovery jobs")
            for index, embedding in zip(missing_indices, embeddings):
                track = jobs[index][2]
                embedding_copy = np.asarray(embedding, dtype=np.float32).copy()
                if self.embedding_cache is not None:
                    self.embedding_cache.put(track.track_id, embedding_copy, frame_index)
                resolved[index] = (jobs[index], embedding_copy)

        return [item for item in resolved if item is not None]

    def _quality(
        self,
        frame: np.ndarray,
        track: Track,
        tracks: Sequence[Track],
    ) -> ReIDQualityResult:
        return assess_reid_quality(
            frame,
            track,
            tracks,
            self.reid_config,
            self.quality_config,
            person_class_id=self.person_class_id,
        )
