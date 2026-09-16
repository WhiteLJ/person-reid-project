"""MVP-5 session-target reference updates and conservative ReID recovery."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from logging import getLogger
from time import perf_counter

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
    centroid_similarity: float
    reference_support_similarity: float

    @property
    def similarity(self) -> float:
        """Backward-compatible alias for the primary centroid score."""

        return self.centroid_similarity


@dataclass(frozen=True)
class RecoveryFrameStats:
    """Per-frame Recovery workload metrics for performance diagnostics only."""

    recovery_due: bool = False
    candidate_count: int = 0
    quality_valid_count: int = 0
    reid_batch_count: int = 0
    reid_ms: float = 0.0
    sweep_started: bool = False
    sweep_completed: bool = False
    sweep_candidate_total: int = 0
    sweep_processed_this_frame: int = 0
    sweep_frames: int = 0
    sweep_reid_ms: float = 0.0


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
    accepted_embedding: np.ndarray
    target_state: TargetState = TargetState.ACTIVE


@dataclass
class _RecoverySweep:
    """A stable, incremental snapshot of one complete Recovery attempt."""

    start_frame: int
    target_ids: tuple[int, ...]
    candidate_track_ids: tuple[int, ...]
    cursor: int = 0
    frames: int = 0
    reid_ms: float = 0.0
    embeddings: dict[int, np.ndarray] = field(default_factory=dict)

    def next_track_ids(self, budget: int) -> tuple[int, ...]:
        """Consume at most ``budget`` snapshot entries exactly once."""

        if budget < 1:
            raise ValueError("Recovery sweep budget must be positive")
        end = min(self.cursor + budget, len(self.candidate_track_ids))
        selected = self.candidate_track_ids[self.cursor:end]
        self.cursor = end
        return selected

    @property
    def complete(self) -> bool:
        return self.cursor >= len(self.candidate_track_ids)


def recovery_score(target: SessionTarget, embedding: np.ndarray) -> float:
    """Return the candidate-to-target centroid cosine similarity."""

    if target.centroid.size == 0:
        raise ValueError(f"target {target.target_id} has no reference centroid")
    return cosine_similarity(target.centroid, embedding)


def recovery_reference_support_score(
    target: SessionTarget,
    embedding: np.ndarray,
    top_k: int = 3,
) -> float:
    """Return the mean of the candidate's strongest reference similarities.

    This is deliberately a top-k mean rather than a maximum.  One accidental
    high score from a single historical reference must not be sufficient to
    recover a LOST target in an open-set scene.  If the target has fewer than
    ``top_k`` references, all available references are used.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not target.reference_embeddings:
        raise ValueError(f"target {target.target_id} has no reference embeddings")
    scores = sorted(
        (
            cosine_similarity(reference, embedding)
            for reference in target.reference_embeddings
        ),
        reverse=True,
    )
    return float(np.mean(scores[: min(top_k, len(scores))]))


def recovery_evidence(
    target: SessionTarget,
    embedding: np.ndarray,
    reference_support_top_k: int = 3,
) -> tuple[float, float]:
    """Return ``(centroid_similarity, reference_support_similarity)``."""

    return (
        recovery_score(target, embedding),
        recovery_reference_support_score(
            target,
            embedding,
            top_k=reference_support_top_k,
        ),
    )


def assign_recovery_matches(
    lost_targets: Sequence[SessionTarget],
    candidates: Sequence[RecoveryCandidate],
    recovery_threshold: float,
    recovery_margin: float,
    recovery_reference_support_threshold: float = 0.80,
    recovery_reference_support_top_k: int = 3,
) -> list[RecoveryMatch]:
    """Return conservative one-to-one target/candidate assignments.

    Both absolute identity checks must pass: centroid similarity and the
    top-k reference-support score.  Then both sides must be confident: the
    best centroid score must beat a second choice by the configured margin
    when one exists.  With no second candidate/target, that side's margin
    check passes automatically.  Accepted pairs are sorted by centroid score
    and greedily claimed, which is deterministic and one-to-one.
    """

    if not 0.0 < recovery_threshold <= 1.0:
        raise ValueError("recovery_threshold must be in (0, 1]")
    if recovery_margin < 0.0:
        raise ValueError("recovery_margin must be non-negative")
    if not 0.0 < recovery_reference_support_threshold <= 1.0:
        raise ValueError(
            "recovery_reference_support_threshold must be in (0, 1]"
        )
    if recovery_reference_support_top_k < 1:
        raise ValueError("recovery_reference_support_top_k must be positive")

    target_list = [target for target in lost_targets if target.state is TargetState.LOST]
    candidate_list = list(candidates)
    if not target_list or not candidate_list:
        return []

    scores: dict[tuple[int, int], tuple[float, float]] = {}
    for target_index, target in enumerate(target_list):
        for candidate_index, candidate in enumerate(candidate_list):
            try:
                evidence = recovery_evidence(
                    target,
                    candidate.embedding,
                    reference_support_top_k=recovery_reference_support_top_k,
                )
            except ValueError:
                continue
            scores[(target_index, candidate_index)] = evidence

    if not scores:
        return []

    target_rankings: dict[int, list[tuple[float, int]]] = {}
    for target_index in range(len(target_list)):
        ranking = sorted(
            (
                evidence[0],
                candidate_index,
            )
            for (row, candidate_index), evidence in scores.items()
            if row == target_index
        )
        target_rankings[target_index] = list(reversed(ranking))

    candidate_rankings: dict[int, list[tuple[float, int]]] = {}
    for candidate_index in range(len(candidate_list)):
        ranking = sorted(
            (
                evidence[0],
                target_index,
            )
            for (target_index, column), evidence in scores.items()
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
        best_support = scores[(target_index, best_candidate_index)][1]
        if best_support < recovery_reference_support_threshold:
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
                centroid_similarity=score,
                reference_support_similarity=scores[
                    (target_index, candidate_index)
                ][1],
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
        self._recovery_sweep: _RecoverySweep | None = None
        self.recovery_attempted_count = 0
        self.recovery_pending_count = 0
        self.recovery_accepted_count = 0
        self.quality_rejected_count = 0
        self.reid_batch_count = 0
        self._reference_updates: list[ReferenceUpdateEvent] = []
        self.last_frame_recovery_stats = RecoveryFrameStats()

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
        """Update visibility and execute Recovery as bounded frame-spread sweeps.

        Candidate enumeration is cheap and happens once when a sweep starts.
        Quality checks and embedding extraction consume at most the configured
        number of snapshot entries per video frame.  Matching is deliberately
        deferred until the complete snapshot has been processed so the
        existing one-to-one and two-sided-margin semantics see the full set.
        """

        self.last_recovered_track_ids = frozenset()
        self.last_frame_recovery_stats = RecoveryFrameStats()
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

        # ACTIVE reference updates retain their existing cadence and are not
        # counted against the incremental Recovery candidate budget.
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

        sweep_started = False
        sweep = self._recovery_sweep
        if sweep is None:
            due_targets = tuple(
                target
                for target in self.target_manager.lost_targets()
                if self.target_manager.recovery_due(
                    target,
                    frame_index,
                    self.recovery_config.recovery_interval_frames,
                )
            )
            if due_targets:
                candidate_ids = tuple(
                    sorted(
                        track.track_id
                        for track in self.target_manager.recovery_candidates(tracks)
                        if track.class_id == self.person_class_id
                        and self._track_ages.get(track.track_id, 0)
                        >= self.recovery_config.recovery_min_track_age_frames
                    )
                )
                sweep = _RecoverySweep(
                    start_frame=frame_index,
                    target_ids=tuple(target.target_id for target in due_targets),
                    candidate_track_ids=candidate_ids,
                )
                self._recovery_sweep = sweep
                sweep_started = True
                for target in due_targets:
                    # The interval is measured between sweep starts, not
                    # between sub-batches inside a sweep.
                    self.target_manager.mark_recovery_attempt(target, frame_index)
                self.recovery_attempted_count += len(due_targets)
                LOGGER.debug(
                    "RECOVERY_SWEEP_STARTED frame=%d targets=%d candidates=%d budget=%d",
                    frame_index,
                    len(due_targets),
                    len(candidate_ids),
                    self.recovery_config.recovery_candidates_per_frame,
                )

        recovery_due = sweep is not None
        processed_track_ids: tuple[int, ...] = ()
        valid_candidate_ids: set[int] = set()
        if sweep is not None:
            sweep.frames += 1
            processed_track_ids = sweep.next_track_ids(
                self.recovery_config.recovery_candidates_per_frame
            )
            tracks_by_id = {track.track_id: track for track in tracks}
            for track_id in processed_track_ids:
                track = tracks_by_id.get(track_id)
                if track is None:
                    # A snapshot entry that disappeared is consumed without
                    # being allowed back into this sweep.
                    continue
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

        candidate_reid_started = perf_counter() if valid_candidate_ids else None
        batch_count_before = self.reid_batch_count
        resolved_jobs = self._ensure_embeddings(embedding_jobs, frame_index)
        candidate_reid_ms = (
            (perf_counter() - candidate_reid_started) * 1000.0
            if candidate_reid_started is not None
            else 0.0
        )
        if sweep is not None:
            sweep.reid_ms += candidate_reid_ms

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
                                accepted_embedding=embedding.copy(),
                                target_state=target.state,
                            )
                        )
            elif sweep is not None and track is not None:
                # This result belongs to the current sweep.  It is retained
                # for the final full-snapshot decision, but is not put into
                # ReIDFrameCache for another frame.
                sweep.embeddings[track.track_id] = embedding.copy()

        self.last_frame_recovery_stats = RecoveryFrameStats(
            recovery_due=recovery_due,
            candidate_count=(len(sweep.candidate_track_ids) if sweep else 0),
            quality_valid_count=len(valid_candidate_ids),
            reid_batch_count=self.reid_batch_count - batch_count_before,
            reid_ms=candidate_reid_ms,
            sweep_started=sweep_started,
            sweep_candidate_total=(len(sweep.candidate_track_ids) if sweep else 0),
            sweep_processed_this_frame=len(processed_track_ids),
            sweep_frames=(sweep.frames if sweep else 0),
            sweep_reid_ms=(sweep.reid_ms if sweep else 0.0),
        )
        if sweep is not None:
            LOGGER.debug(
                "RECOVERY_SWEEP_PROGRESS frame=%d candidate_total=%d "
                "processed_this_frame=%d frames=%d reid_ms=%.2f",
                frame_index,
                len(sweep.candidate_track_ids),
                len(processed_track_ids),
                sweep.frames,
                sweep.reid_ms,
            )

        if sweep is None or not sweep.complete:
            return []

        current_sweep = sweep
        self._recovery_sweep = None
        matches = self._complete_sweep(current_sweep, tracks, frame_index)
        self.last_frame_recovery_stats = replace(
            self.last_frame_recovery_stats,
            sweep_completed=True,
            sweep_frames=current_sweep.frames,
            sweep_reid_ms=current_sweep.reid_ms,
        )
        LOGGER.debug(
            "RECOVERY_SWEEP_COMPLETED start_frame=%d frame=%d "
            "candidates=%d frames=%d reid_ms=%.2f",
            current_sweep.start_frame,
            frame_index,
            len(current_sweep.candidate_track_ids),
            current_sweep.frames,
            current_sweep.reid_ms,
        )
        return matches

    def _complete_sweep(
        self,
        sweep: _RecoverySweep,
        tracks: Sequence[Track],
        frame_index: int,
    ) -> list[RecoveryMatch]:
        """Run the unchanged full-snapshot matching/confirmation phase."""

        lost_targets = [
            target
            for target_id in sweep.target_ids
            if (target := self.target_manager.targets.get(target_id)) is not None
            and target.state is TargetState.LOST
        ]
        current_tracks = {track.track_id: track for track in tracks}
        currently_unbound = {
            track.track_id
            for track in self.target_manager.recovery_candidates(tracks)
            if track.class_id == self.person_class_id
        }
        recovery_candidates = [
            RecoveryCandidate(current_tracks[track_id], embedding.copy())
            for track_id, embedding in sweep.embeddings.items()
            if track_id in current_tracks and track_id in currently_unbound
        ]
        if not lost_targets or not recovery_candidates:
            return []

        matches = assign_recovery_matches(
            lost_targets,
            recovery_candidates,
            recovery_threshold=self.recovery_config.recovery_threshold,
            recovery_margin=self.recovery_config.recovery_margin,
            recovery_reference_support_threshold=(
                self.recovery_config.recovery_reference_support_threshold
            ),
            recovery_reference_support_top_k=(
                self.recovery_config.recovery_reference_support_top_k
            ),
        )
        self._log_recovery_rejections(
            lost_targets,
            recovery_candidates,
            matches,
            frame_index,
        )
        accepted_pairs = {
            (match.target_id, match.candidate.track.track_id) for match in matches
        }
        lost_target_ids = {target.target_id for target in lost_targets}
        for key in list(self._pending):
            if key[0] in lost_target_ids and key not in accepted_pairs:
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
                    "TARGET_RECOVERY_PENDING frame=%d target_id=%d "
                    "candidate_track_id=%d hits=%d/%d centroid=%.4f support=%.4f",
                    frame_index,
                    match.target_id,
                    match.candidate.track.track_id,
                    hits,
                    self.recovery_config.recovery_confirmation_hits,
                    match.centroid_similarity,
                    match.reference_support_similarity,
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
                reference_support_similarity=match.reference_support_similarity,
            )
            self._pending.pop(key, None)
            self.recovery_accepted_count += 1
            recovered_matches.append(match)
        self.last_recovered_track_ids = frozenset(
            match.candidate.track.track_id for match in recovered_matches
        )
        return recovered_matches

    def _log_recovery_rejections(
        self,
        lost_targets: Sequence[SessionTarget],
        candidates: Sequence[RecoveryCandidate],
        matches: Sequence[RecoveryMatch],
        frame_index: int,
    ) -> None:
        """Log the strongest rejected evidence without changing matching.

        This is intentionally diagnostic only.  Matching remains implemented
        by ``assign_recovery_matches``; the log helps distinguish an absolute
        evidence rejection from a margin/assignment rejection in open-set
        footage without producing INFO-level per-frame noise.
        """

        matched_target_ids = {match.target_id for match in matches}
        for target in lost_targets:
            if target.target_id in matched_target_ids:
                continue
            scored: list[tuple[float, float, RecoveryCandidate]] = []
            for candidate in candidates:
                try:
                    centroid, support = recovery_evidence(
                        target,
                        candidate.embedding,
                        reference_support_top_k=(
                            self.recovery_config.recovery_reference_support_top_k
                        ),
                    )
                except ValueError:
                    continue
                scored.append((centroid, support, candidate))
            if not scored:
                continue

            centroid, support, candidate = max(
                scored,
                key=lambda item: (
                    item[0],
                    item[1],
                    -item[2].track.track_id,
                ),
            )
            if centroid < self.recovery_config.recovery_threshold:
                reason = "centroid_threshold"
            elif (
                support
                < self.recovery_config.recovery_reference_support_threshold
            ):
                reason = "reference_support"
            else:
                reason = "margin"
            LOGGER.debug(
                "TARGET_RECOVERY_REJECTED frame=%d target_id=%d "
                "candidate_track_id=%d centroid_score=%.4f "
                "reference_support_score=%.4f reason=%s",
                frame_index,
                target.target_id,
                candidate.track.track_id,
                centroid,
                support,
                reason,
            )

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
