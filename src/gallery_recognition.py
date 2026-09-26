"""Conservative automatic recognition against the in-memory TargetGallery."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field, replace
from logging import getLogger
from time import perf_counter
from typing import Any

import numpy as np

from .config import (
    GalleryRecognitionConfig,
    ReIDConfig,
    ReIDQualityConfig,
    ReIDRecoveryConfig,
)
from .gallery import TargetGallery
from .models import Track
from .reid import ReIDExtractor, cosine_similarity
from .reid_frame_cache import ReIDFrameCache
from .reid_frame_budget import ReIDFrameBudget
from .reid_quality import assess_reid_quality
from .target_manager import TargetManager


LOGGER = getLogger(__name__)


@dataclass(frozen=True)
class GalleryRecognitionCandidate:
    """A valid current Track and its normalized current-frame embedding."""

    track: Track
    embedding: np.ndarray


@dataclass(frozen=True)
class GalleryRecognitionMatch:
    """A conservative Gallery person/candidate assignment."""

    person_id: int
    candidate: GalleryRecognitionCandidate
    similarity: float

    @property
    def identity_id(self) -> int:
        """Generic alias; ``person_id`` remains for Person compatibility."""

        return self.person_id


@dataclass(frozen=True)
class GalleryRecognitionAdapter:
    """Domain callbacks shared by Person and Vehicle recognition."""

    all_identities: Callable[[], Sequence[Any]]
    attached_identity_ids: Callable[[], Collection[int]]
    get_identity: Callable[[int], Any | None]
    identity_for_session_target: Callable[[int], Any | None]
    attach_session_target: Callable[[int, int], bool]
    identity_id: Callable[[Any], int]
    identity_label: Callable[[int], str]


def person_gallery_adapter(gallery: TargetGallery) -> GalleryRecognitionAdapter:
    """Build the default adapter without changing Person behavior."""

    return GalleryRecognitionAdapter(
        all_identities=gallery.all_people,
        attached_identity_ids=gallery.attached_person_ids,
        get_identity=gallery.get,
        identity_for_session_target=gallery.person_for_session_target,
        attach_session_target=gallery.attach_session_target,
        identity_id=lambda person: person.person_id,
        identity_label=_person_label,
    )


@dataclass
class _PendingRecognition:
    hits: int


@dataclass(frozen=True)
class _PendingRecognitionRevalidation:
    match: GalleryRecognitionMatch
    snapshot_frame: int


@dataclass(frozen=True)
class GalleryRecognitionFrameStats:
    """Per-frame recognition workload metrics for diagnostics."""

    recognition_due: bool = False
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
    skipped_retry_cooldown: int = 0
    candidate_reid_ms: float = 0.0
    revalidation_reid_ms: float = 0.0
    cache_hits: int = 0
    deferred_by_budget: int = 0
    candidate_new_embeddings: int = 0
    revalidation_new_embeddings: int = 0
    total_reid_ms: float = 0.0
    total_new_embeddings: int = 0
    max_new_embeddings_per_frame: int = 0


@dataclass
class _RecognitionSweep:
    """Stable candidate/identity snapshot for one complete recognition attempt."""

    start_frame: int
    identity_ids: tuple[int, ...]
    candidate_track_ids: tuple[int, ...]
    cursor: int = 0
    frames: int = 0
    reid_ms: float = 0.0
    quality_valid_count: int = 0
    embeddings: dict[int, np.ndarray] = field(default_factory=dict)
    embedding_frame_by_track: dict[int, int] = field(default_factory=dict)
    protected_track_ids: set[int] = field(default_factory=set)

    def peek_track_ids(self, budget: int) -> tuple[int, ...]:
        if budget < 1:
            raise ValueError("Gallery recognition sweep budget must be positive")
        end = min(self.cursor + budget, len(self.candidate_track_ids))
        return self.candidate_track_ids[self.cursor:end]

    def advance(self, count: int) -> None:
        if count < 0 or self.cursor + count > len(self.candidate_track_ids):
            raise ValueError("invalid Gallery recognition sweep cursor advance")
        self.cursor += count

    @property
    def complete(self) -> bool:
        return self.cursor >= len(self.candidate_track_ids)


def gallery_recognition_score(
    person: Any,
    embedding: np.ndarray,
) -> float:
    """Score a candidate against the stable normalized Gallery centroid."""

    return cosine_similarity(person.centroid, embedding)


def assign_gallery_matches(
    people: Sequence[Any],
    candidates: Sequence[GalleryRecognitionCandidate],
    recognition_threshold: float,
    recognition_margin: float,
    *,
    identity_id_getter: Callable[[Any], int] | None = None,
) -> list[GalleryRecognitionMatch]:
    """Return conservative, deterministic one-to-one Gallery assignments.

    A person-side and candidate-side margin are both enforced.  If either
    side has no second choice, that side passes automatically.  Scores use the
    Gallery centroid rather than the maximum historical embedding score so a
    single anomalous reference cannot force recognition.
    """

    if not 0.0 < recognition_threshold <= 1.0:
        raise ValueError("recognition_threshold must be in (0, 1]")
    if recognition_margin < 0.0:
        raise ValueError("recognition_margin must be non-negative")

    person_list = list(people)
    identity_id = identity_id_getter or (lambda person: person.person_id)
    candidate_list = list(candidates)
    if not person_list or not candidate_list:
        return []

    scores: dict[tuple[int, int], float] = {}
    for person_index, person in enumerate(person_list):
        for candidate_index, candidate in enumerate(candidate_list):
            try:
                scores[(person_index, candidate_index)] = gallery_recognition_score(
                    person,
                    candidate.embedding,
                )
            except ValueError:
                continue

    person_rankings: dict[int, list[tuple[float, int]]] = {}
    for person_index in range(len(person_list)):
        ranking = [
            (score, candidate_index)
            for (row, candidate_index), score in scores.items()
            if row == person_index
        ]
        person_rankings[person_index] = sorted(
            ranking,
            key=lambda item: (-item[0], candidate_list[item[1]].track.track_id),
        )

    candidate_rankings: dict[int, list[tuple[float, int]]] = {}
    for candidate_index in range(len(candidate_list)):
        ranking = [
            (score, person_index)
            for (person_index, column), score in scores.items()
            if column == candidate_index
        ]
        candidate_rankings[candidate_index] = sorted(
            ranking,
            key=lambda item: (-item[0], identity_id(person_list[item[1]])),
        )

    eligible: list[tuple[float, int, int]] = []
    for person_index, ranking in person_rankings.items():
        if not ranking:
            continue
        best_score, candidate_index = ranking[0]
        if best_score < recognition_threshold:
            continue
        person_margin_ok = (
            len(ranking) == 1
            or best_score - ranking[1][0] >= recognition_margin
        )
        if not person_margin_ok:
            continue

        candidate_ranking = candidate_rankings[candidate_index]
        candidate_margin_ok = (
            len(candidate_ranking) == 1
            or best_score - candidate_ranking[1][0] >= recognition_margin
        )
        if candidate_margin_ok:
            eligible.append((best_score, person_index, candidate_index))

    eligible.sort(
        key=lambda item: (
            -item[0],
            identity_id(person_list[item[1]]),
            candidate_list[item[2]].track.track_id,
        )
    )
    used_people: set[int] = set()
    used_candidates: set[int] = set()
    matches: list[GalleryRecognitionMatch] = []
    for score, person_index, candidate_index in eligible:
        if person_index in used_people or candidate_index in used_candidates:
            continue
        used_people.add(person_index)
        used_candidates.add(candidate_index)
        matches.append(
            GalleryRecognitionMatch(
                person_id=identity_id(person_list[person_index]),
                candidate=candidate_list[candidate_index],
                similarity=score,
            )
        )
    return matches


class GalleryRecognitionCoordinator:
    """Incrementally recognize unbound Tracks against a loaded Gallery.

    Only neural feature extraction is spread across frames.  Gallery cosine
    scoring, one-to-one assignment, margins, and confirmation still run once
    against the complete sweep snapshot.
    """

    def __init__(
        self,
        target_manager: TargetManager,
        gallery: TargetGallery,
        reid_extractor: ReIDExtractor,
        reid_config: ReIDConfig,
        recognition_config: GalleryRecognitionConfig,
        recovery_config: ReIDRecoveryConfig,
        *,
        person_class_id: int = 0,
        embedding_cache: ReIDFrameCache | None = None,
        quality_config: ReIDQualityConfig | None = None,
        quality_assessor: Callable[..., Any] | None = None,
        track_class_id: int | None = None,
        track_class_ids: Collection[int] | None = None,
        gallery_adapter: GalleryRecognitionAdapter | None = None,
        reid_budget: ReIDFrameBudget | None = None,
    ) -> None:
        self.target_manager = target_manager
        self.gallery = gallery
        self.reid_extractor = reid_extractor
        self.reid_config = reid_config
        self.recognition_config = recognition_config
        self.recovery_config = recovery_config
        self.person_class_id = person_class_id
        if track_class_ids is None:
            resolved_class_ids = (
                person_class_id if track_class_id is None else track_class_id,
            )
        else:
            resolved_class_ids = tuple(
                sorted({int(class_id) for class_id in track_class_ids})
            )
            if track_class_id is not None and resolved_class_ids != (
                int(track_class_id),
            ):
                raise ValueError(
                    "track_class_id and track_class_ids specify different classes"
                )
        if not resolved_class_ids:
            raise ValueError("track_class_ids must not be empty")
        self.track_class_ids = frozenset(resolved_class_ids)
        self.track_class_id = (
            next(iter(self.track_class_ids))
            if len(self.track_class_ids) == 1
            else None
        )
        self.embedding_cache = embedding_cache
        self.quality_config = quality_config or ReIDQualityConfig()
        self.quality_assessor = quality_assessor
        self.gallery_adapter = gallery_adapter or person_gallery_adapter(gallery)
        # AppConfig injects the finite domain-wide budget.  A large fallback
        # keeps legacy standalone coordinator construction compatible.
        self.reid_budget = reid_budget or ReIDFrameBudget(1_000_000)
        self._track_ages: dict[int, int] = {}
        self._last_seen_frame: dict[int, int] = {}
        self._pending: dict[tuple[int, int], _PendingRecognition] = {}
        self._retry_until_frame: dict[int, int] = {}
        self._recognition_sweep: _RecognitionSweep | None = None
        self._pending_revalidations: dict[
            tuple[int, int], _PendingRecognitionRevalidation
        ] = {}
        self._last_recognition_frame: int | None = None
        self.recognized_count = 0
        self.quality_rejected_count = 0
        self.reid_batch_count = 0
        self.retry_skipped_count = 0
        self._retry_skipped_this_frame = 0
        self.last_frame_recognition_stats = GalleryRecognitionFrameStats()

    @property
    def pending(self) -> dict[tuple[int, int], int]:
        """Return a read-only-by-convention snapshot useful for diagnostics/tests."""

        return {key: value.hits for key, value in self._pending.items()}

    @property
    def track_ages(self) -> dict[int, int]:
        return dict(self._track_ages)

    @property
    def retry_until_frame(self) -> dict[int, int]:
        """Return a diagnostic snapshot of unmatched-track retry cooldowns."""

        return dict(self._retry_until_frame)

    def invalidate_retry_state(self) -> None:
        """Allow all visible Tracks to be reconsidered after Gallery changes."""

        self._retry_until_frame.clear()

    def notify_gallery_changed(self) -> None:
        """Invalidate retry state after an in-memory Gallery mutation.

        Recognition runs only against the in-memory Gallery, so explicit
        enroll/remove/clear operations must invalidate cooldowns without
        querying SQLite.  Pending pairs for identities that no longer exist
        are also stale and are discarded.
        """

        self.invalidate_retry_state()
        attached = self.gallery_adapter.attached_identity_ids()
        for key in list(self._pending):
            if (
                self.gallery_adapter.get_identity(key[0]) is None
                or key[0] in attached
            ):
                del self._pending[key]
        for key in list(self._pending_revalidations):
            if (
                self.gallery_adapter.get_identity(key[0]) is None
                or key[0] in attached
            ):
                del self._pending_revalidations[key]

    def _recognition_budget(self, candidate_count: int) -> int:
        configured = int(
            getattr(self.recognition_config, "recognition_candidates_per_frame", 0)
        )
        # Legacy direct construction of GalleryRecognitionConfig used no budget;
        # keep that API behavior while YAML-loaded configs always set one.
        return candidate_count if configured < 1 else configured

    def _retry_interval(self) -> int:
        return max(
            1,
            int(
                getattr(
                    self.recognition_config,
                    "unmatched_retry_interval_frames",
                    15,
                )
            ),
        )

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        frame_index: int,
        *,
        protected_track_ids: Sequence[int] = (),
    ) -> list[GalleryRecognitionMatch]:
        """Run due recognition work and bind existing Gallery identities.

        ``protected_track_ids`` is intentionally a narrow hand-off from
        Recovery: it contains only Tracks successfully claimed by Recovery,
        not every Track that Recovery inspected.  Unmatched Recovery
        candidates remain eligible and can reuse this frame's cached feature.
        """

        self.reid_budget.begin_frame(frame_index)
        self._update_track_ages(tracks, frame_index)
        if self.embedding_cache is not None:
            self.embedding_cache.begin_frame(frame_index)

        self.last_frame_recognition_stats = GalleryRecognitionFrameStats()
        self._retry_skipped_this_frame = 0
        protected_ids = set(protected_track_ids)
        sweep_started = False
        candidate_reid_ms = 0.0
        revalidation_reid_ms = 0.0
        candidate_new = revalidation_new = 0
        cache_hits = 0
        deferred = 0
        processed_count = 0
        batch_count_before = self.reid_batch_count

        if not self.recognition_config.enabled:
            return []
        recognized, revalidation_metrics = self._process_pending_revalidations(
            frame, tracks, frame_index, protected_ids
        )
        revalidation_reid_ms += revalidation_metrics[0]
        revalidation_new += revalidation_metrics[1]
        cache_hits += revalidation_metrics[2]
        deferred += revalidation_metrics[3]

        sweep = self._recognition_sweep
        if (
            sweep is None
            and not self._pending_revalidations
            and self._recognition_due(frame_index)
        ):
            sweep = self._start_sweep(tracks, frame_index, protected_ids)
            if sweep is not None:
                self._recognition_sweep = sweep
                sweep_started = True

        if sweep is not None:
            sweep.protected_track_ids.update(protected_ids)
            sweep.frames += 1
            tracks_by_id = {track.track_id: track for track in tracks}
            selected = sweep.peek_track_ids(
                self._recognition_budget(len(sweep.candidate_track_ids))
            )
            jobs: list[tuple[Track, np.ndarray]] = []
            consumed_without_inference = 0
            for track_id in selected:
                track = tracks_by_id.get(track_id)
                if (
                    track is None
                    or track.class_id not in self.track_class_ids
                    or track_id in sweep.protected_track_ids
                ):
                    sweep.advance(1)
                    consumed_without_inference += 1
                    continue
                active_target = self.target_manager.target_for_track(track_id)
                if (
                    active_target is not None
                    and self.gallery_adapter.identity_for_session_target(
                        active_target.target_id
                    )
                    is not None
                ):
                    sweep.advance(1)
                    consumed_without_inference += 1
                    continue
                quality = self._assess_quality(frame, track, tracks)
                if not quality.accepted or quality.crop is None:
                    self.quality_rejected_count += 1
                    sweep.advance(1)
                    consumed_without_inference += 1
                    LOGGER.debug(
                        "REID_QUALITY_REJECTED kind=gallery track=%d reason=%s",
                        track_id,
                        quality.reason or "unknown",
                    )
                    continue
                sweep.quality_valid_count += 1
                cached = (
                    self.embedding_cache.get(track_id, frame_index)
                    if self.embedding_cache is not None
                    else None
                )
                if cached is not None:
                    sweep.embeddings[track_id] = cached.copy()
                    sweep.embedding_frame_by_track[track_id] = frame_index
                    sweep.advance(1)
                    consumed_without_inference += 1
                    cache_hits += 1
                    continue
                if not self.reid_budget.can_consume(len(jobs) + 1):
                    deferred += 1
                    break
                jobs.append((track, quality.crop))

            if jobs:
                self.reid_budget.consume(len(jobs))
                started = perf_counter()
                embeddings = np.asarray(
                    self.reid_extractor.extract_batch([job[1] for job in jobs]),
                    dtype=np.float32,
                )
                candidate_reid_ms += (perf_counter() - started) * 1000.0
                self.reid_batch_count += 1
                if embeddings.ndim != 2 or embeddings.shape[0] != len(jobs):
                    raise ValueError("ReID batch output does not match recognition candidates")
                for (track, _crop), embedding in zip(jobs, embeddings):
                    value = np.asarray(embedding, dtype=np.float32).copy()
                    if self.embedding_cache is not None:
                        self.embedding_cache.put(track.track_id, value, frame_index)
                    sweep.embeddings[track.track_id] = value
                    sweep.embedding_frame_by_track[track.track_id] = frame_index
                sweep.advance(len(jobs))
                processed_count += len(jobs)
                candidate_new += len(jobs)
                sweep.reid_ms += candidate_reid_ms
            processed_count += consumed_without_inference

            LOGGER.debug(
                "GALLERY_RECOGNITION_SWEEP_PROGRESS frame=%d candidate_total=%d "
                "processed_this_frame=%d frames=%d reid_ms=%.2f",
                frame_index,
                len(sweep.candidate_track_ids),
                processed_count,
                sweep.frames,
                sweep.reid_ms,
            )

            if sweep.complete:
                current_sweep = sweep
                self._recognition_sweep = None
                self._complete_sweep(current_sweep, tracks, frame_index)
                LOGGER.debug(
                    "GALLERY_RECOGNITION_SWEEP_COMPLETED start_frame=%d frame=%d candidates=%d frames=%d reid_ms=%.2f",
                    current_sweep.start_frame,
                    frame_index,
                    len(current_sweep.candidate_track_ids),
                    current_sweep.frames,
                    current_sweep.reid_ms,
                )
                more, revalidation_metrics = self._process_pending_revalidations(
                    frame, tracks, frame_index, protected_ids
                )
                recognized.extend(more)
                revalidation_reid_ms += revalidation_metrics[0]
                revalidation_new += revalidation_metrics[1]
                cache_hits += revalidation_metrics[2]
                deferred += revalidation_metrics[3]

        total_reid_ms = candidate_reid_ms + revalidation_reid_ms
        total_new = candidate_new + revalidation_new
        self.last_frame_recognition_stats = GalleryRecognitionFrameStats(
            recognition_due=(sweep is not None),
            candidate_count=(len(sweep.candidate_track_ids) if sweep else 0),
            quality_valid_count=(sweep.quality_valid_count if sweep else 0),
            reid_batch_count=self.reid_batch_count - batch_count_before,
            reid_ms=total_reid_ms,
            sweep_started=sweep_started,
            sweep_completed=(sweep is not None and sweep.complete),
            sweep_candidate_total=(len(sweep.candidate_track_ids) if sweep else 0),
            sweep_processed_this_frame=processed_count,
            sweep_frames=(sweep.frames if sweep else 0),
            sweep_reid_ms=(sweep.reid_ms if sweep else 0.0),
            skipped_retry_cooldown=self._retry_skipped_this_frame,
            candidate_reid_ms=candidate_reid_ms,
            revalidation_reid_ms=revalidation_reid_ms,
            cache_hits=cache_hits,
            deferred_by_budget=deferred,
            candidate_new_embeddings=candidate_new,
            revalidation_new_embeddings=revalidation_new,
            total_reid_ms=total_reid_ms,
            total_new_embeddings=total_new,
            max_new_embeddings_per_frame=self.reid_budget.capacity,
        )
        return recognized

    def _assess_quality(
        self,
        frame: np.ndarray,
        track: Track,
        tracks: Sequence[Track],
    ) -> Any:
        if self.quality_assessor is not None:
            return self.quality_assessor(frame, track, tracks)
        return assess_reid_quality(
            frame,
            track,
            tracks,
            self.reid_config,
            self.quality_config,
            person_class_id=self.person_class_id,
        )

    def _recognition_due(self, frame_index: int) -> bool:
        return self._last_recognition_frame is None or (
            frame_index - self._last_recognition_frame
            >= self.recognition_config.recognition_interval_frames
        )

    def _free_identity_ids(self) -> tuple[int, ...]:
        attached = self.gallery_adapter.attached_identity_ids()
        return tuple(
            self.gallery_adapter.identity_id(identity)
            for identity in self.gallery_adapter.all_identities()
            if self.gallery_adapter.identity_id(identity) not in attached
        )

    def _start_sweep(
        self,
        tracks: Sequence[Track],
        frame_index: int,
        protected_track_ids: set[int],
    ) -> _RecognitionSweep | None:
        identity_ids = self._free_identity_ids()
        if not identity_ids:
            return None

        candidate_track_ids: list[int] = []
        skipped_retry = 0
        for track in tracks:
            if track.class_id not in self.track_class_ids:
                continue
            if self._track_ages.get(track.track_id, 0) < (
                self.recognition_config.min_track_age_frames
            ):
                continue
            if track.track_id in protected_track_ids:
                continue
            active_target = self.target_manager.target_for_track(track.track_id)
            if (
                active_target is not None
                and self.gallery_adapter.identity_for_session_target(
                    active_target.target_id
                )
                is not None
            ):
                continue
            next_eligible_frame = self._retry_until_frame.get(track.track_id)
            if next_eligible_frame is not None and frame_index < next_eligible_frame:
                skipped_retry += 1
                continue
            candidate_track_ids.append(track.track_id)

        self.retry_skipped_count += skipped_retry
        self._retry_skipped_this_frame = skipped_retry
        if not candidate_track_ids:
            return None

        self._last_recognition_frame = frame_index
        sweep = _RecognitionSweep(
            start_frame=frame_index,
            identity_ids=identity_ids,
            candidate_track_ids=tuple(candidate_track_ids),
        )
        LOGGER.debug(
            "GALLERY_RECOGNITION_SWEEP_STARTED frame=%d identities=%d "
            "candidates=%d budget=%d skipped_retry=%d",
            frame_index,
            len(identity_ids),
            len(candidate_track_ids),
            self._recognition_budget(len(candidate_track_ids)),
            skipped_retry,
        )
        return sweep

    def _complete_sweep(
        self,
        sweep: _RecognitionSweep,
        tracks: Sequence[Track],
        frame_index: int,
    ) -> list[GalleryRecognitionMatch]:
        current_tracks = {track.track_id: track for track in tracks}
        attached = self.gallery_adapter.attached_identity_ids()
        identities = [
            identity
            for identity_id in sweep.identity_ids
            if identity_id not in attached
            and (identity := self.gallery_adapter.get_identity(identity_id)) is not None
        ]
        candidates: list[GalleryRecognitionCandidate] = []
        for track_id, embedding in sweep.embeddings.items():
            track = current_tracks.get(track_id)
            if track is None or track.class_id not in self.track_class_ids:
                continue
            if track_id in sweep.protected_track_ids:
                continue
            active_target = self.target_manager.target_for_track(track_id)
            if (
                active_target is not None
                and self.gallery_adapter.identity_for_session_target(
                    active_target.target_id
                )
                is not None
            ):
                continue
            candidates.append(
                GalleryRecognitionCandidate(track, embedding.copy())
            )

        if not identities or not candidates:
            return []

        matches = assign_gallery_matches(
            identities,
            candidates,
            recognition_threshold=self.recognition_config.recognition_threshold,
            recognition_margin=self.recognition_config.recognition_margin,
            identity_id_getter=self.gallery_adapter.identity_id,
        )
        accepted_pairs = {
            (match.person_id, match.candidate.track.track_id) for match in matches
        }
        attempted_track_ids = {candidate.track.track_id for candidate in candidates}
        self._clear_failed_pending(accepted_pairs, attempted_track_ids)
        matched_track_ids = {
            match.candidate.track.track_id for match in matches
        }
        for track_id in attempted_track_ids - matched_track_ids:
            self._retry_until_frame[track_id] = (
                frame_index + self._retry_interval()
            )

        for match in matches:
            key = (match.person_id, match.candidate.track.track_id)
            snapshot_frame = sweep.embedding_frame_by_track.get(
                match.candidate.track.track_id,
                sweep.start_frame,
            )
            self._pending_revalidations[key] = _PendingRecognitionRevalidation(
                match=match,
                snapshot_frame=snapshot_frame,
            )
            LOGGER.debug(
                "RECOGNITION_PROVISIONAL frame=%d identity_id=%s track_id=%d "
                "snapshot_embedding_frame=%d snapshot_embedding_age=%d "
                "snapshot_similarity=%.4f",
                frame_index,
                self.gallery_adapter.identity_label(match.person_id),
                match.candidate.track.track_id,
                snapshot_frame,
                frame_index - snapshot_frame,
                match.similarity,
            )

        return []

    def _process_pending_revalidations(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        frame_index: int,
        protected_track_ids: set[int],
    ) -> tuple[list[GalleryRecognitionMatch], tuple[float, int, int, int]]:
        """Freshly verify provisional Gallery pairs before confirmation."""

        recognized: list[GalleryRecognitionMatch] = []
        reid_ms = 0.0
        new_embeddings = 0
        cache_hits = 0
        deferred = 0
        tracks_by_id = {track.track_id: track for track in tracks}
        for key, proposal in list(self._pending_revalidations.items()):
            identity_id, track_id = key
            identity = self.gallery_adapter.get_identity(identity_id)
            track = tracks_by_id.get(track_id)
            reason: str | None = None
            if identity is None:
                reason = "identity_missing"
            elif track is None:
                reason = "track_missing"
            elif track.class_id not in self.track_class_ids:
                reason = "class_changed"
            elif track_id in protected_track_ids:
                reason = "recovery_claimed"
            else:
                active_target = self.target_manager.target_for_track(track_id)
                if (
                    active_target is not None
                    and self.gallery_adapter.identity_for_session_target(
                        active_target.target_id
                    )
                    is not None
                ):
                    reason = "track_bound"
            if reason is not None:
                self._pending_revalidations.pop(key, None)
                self._pending.pop(key, None)
                LOGGER.debug(
                    "RECOGNITION_REVALIDATION_REJECTED frame=%d identity_id=%s "
                    "track_id=%d reason=%s",
                    frame_index,
                    self.gallery_adapter.identity_label(identity_id),
                    track_id,
                    reason,
                )
                continue

            quality = self._assess_quality(frame, track, tracks)
            if not quality.accepted or quality.crop is None:
                self.quality_rejected_count += 1
                self._pending_revalidations.pop(key, None)
                self._pending.pop(key, None)
                LOGGER.debug(
                    "RECOGNITION_REVALIDATION_REJECTED frame=%d identity_id=%s "
                    "track_id=%d reason=quality",
                    frame_index,
                    self.gallery_adapter.identity_label(identity_id),
                    track_id,
                )
                continue

            embedding = None
            if proposal.snapshot_frame == frame_index:
                embedding = proposal.match.candidate.embedding.copy()
            elif self.embedding_cache is not None:
                embedding = self.embedding_cache.get(track_id, frame_index)
            if embedding is not None:
                cache_hits += 1
            else:
                if not self.reid_budget.can_consume(1):
                    deferred += 1
                    continue
                self.reid_budget.consume(1)
                started = perf_counter()
                values = np.asarray(
                    self.reid_extractor.extract_batch([quality.crop]),
                    dtype=np.float32,
                )
                reid_ms += (perf_counter() - started) * 1000.0
                self.reid_batch_count += 1
                if values.ndim != 2 or values.shape[0] != 1:
                    raise ValueError("ReID batch output does not match revalidation")
                embedding = values[0].copy()
                new_embeddings += 1
                if self.embedding_cache is not None:
                    self.embedding_cache.put(track_id, embedding, frame_index)

            try:
                fresh_similarity = gallery_recognition_score(identity, embedding)
            except ValueError:
                fresh_similarity = 0.0
            if fresh_similarity < self.recognition_config.recognition_threshold:
                self._pending_revalidations.pop(key, None)
                self._pending.pop(key, None)
                self._retry_until_frame[track_id] = frame_index + self._retry_interval()
                LOGGER.debug(
                    "RECOGNITION_REVALIDATION_REJECTED frame=%d identity_id=%s "
                    "track_id=%d snapshot_similarity=%.4f fresh_similarity=%.4f reason=threshold",
                    frame_index,
                    self.gallery_adapter.identity_label(identity_id),
                    track_id,
                    proposal.match.similarity,
                    fresh_similarity,
                )
                continue

            age = frame_index - proposal.snapshot_frame
            LOGGER.debug(
                "RECOGNITION_REVALIDATED frame=%d identity_id=%s track_id=%d "
                "snapshot_embedding_age=%d snapshot_similarity=%.4f fresh_similarity=%.4f",
                frame_index,
                self.gallery_adapter.identity_label(identity_id),
                track_id,
                age,
                proposal.match.similarity,
                fresh_similarity,
            )
            self._pending_revalidations.pop(key, None)
            hits = self._pending.get(key, _PendingRecognition(0)).hits + 1
            fresh_match = replace(
                proposal.match,
                candidate=GalleryRecognitionCandidate(track, np.asarray(embedding).copy()),
                similarity=fresh_similarity,
            )
            if hits < self.recognition_config.confirmation_hits:
                self._pending[key] = _PendingRecognition(hits=hits)
                LOGGER.debug(
                    "GALLERY_RECOGNITION_PENDING identity=%s track=%d hits=%d/%d similarity=%.4f",
                    self.gallery_adapter.identity_label(identity_id),
                    track_id,
                    hits,
                    self.recognition_config.confirmation_hits,
                    fresh_similarity,
                )
                continue
            if self._bind_match(fresh_match, frame_index):
                recognized.append(fresh_match)
                self.recognized_count += 1
            self._pending.pop(key, None)
        return recognized, (reid_ms, new_embeddings, cache_hits, deferred)

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
                self._retry_until_frame.pop(track_id, None)
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

    def _build_candidates(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        protected_track_ids: set[int],
    ) -> list[tuple[Track, np.ndarray]]:
        candidates: list[tuple[Track, np.ndarray]] = []
        for track in tracks:
            if track.class_id not in self.track_class_ids:
                continue
            if self._track_ages.get(track.track_id, 0) < self.recognition_config.min_track_age_frames:
                continue
            if track.track_id in protected_track_ids:
                continue

            # An ACTIVE manually-selected, Gallery-unbound SessionTarget is
            # eligible for an existing Gallery attachment.  An already
            # Gallery-bound target is not eligible for a second identity.
            active_target = self.target_manager.target_for_track(track.track_id)
            if (
                active_target is not None
                and self.gallery_adapter.identity_for_session_target(
                    active_target.target_id
                )
                is not None
            ):
                continue
            if self.quality_assessor is None:
                quality = assess_reid_quality(
                    frame,
                    track,
                    tracks,
                    self.reid_config,
                    self.quality_config,
                    person_class_id=self.person_class_id,
                )
            else:
                quality = self.quality_assessor(frame, track, tracks)
            if not quality.accepted or quality.crop is None:
                self.quality_rejected_count += 1
                LOGGER.debug(
                    "REID_QUALITY_REJECTED kind=gallery track=%d reason=%s",
                    track.track_id,
                    quality.reason or "unknown",
                )
                continue
            candidates.append((track, quality.crop))
        return candidates

    def _ensure_embeddings(
        self,
        candidates: Sequence[tuple[Track, np.ndarray]],
        frame_index: int,
    ) -> list[GalleryRecognitionCandidate]:
        resolved: list[GalleryRecognitionCandidate | None] = [None] * len(candidates)
        missing_indices: list[int] = []
        missing_crops: list[np.ndarray] = []
        for index, (track, crop) in enumerate(candidates):
            if self.embedding_cache is not None:
                cached = self.embedding_cache.get(track.track_id, frame_index)
            else:
                cached = None
            if cached is not None:
                resolved[index] = GalleryRecognitionCandidate(track, cached)
                continue
            missing_indices.append(index)
            missing_crops.append(crop)

        if missing_crops:
            self.reid_batch_count += 1
            embeddings = np.asarray(
                self.reid_extractor.extract_batch(missing_crops),
                dtype=np.float32,
            )
            if embeddings.ndim != 2 or embeddings.shape[0] != len(missing_crops):
                raise ValueError("ReID batch output does not match recognition candidates")
            for index, embedding in zip(missing_indices, embeddings):
                track = candidates[index][0]
                embedding_copy = np.asarray(embedding, dtype=np.float32).copy()
                if self.embedding_cache is not None:
                    self.embedding_cache.put(track.track_id, embedding_copy, frame_index)
                resolved[index] = GalleryRecognitionCandidate(track, embedding_copy)

        return [candidate for candidate in resolved if candidate is not None]

    def _clear_failed_pending(
        self,
        accepted_pairs: set[tuple[int, int]],
        attempted_track_ids: set[int],
    ) -> None:
        for key in list(self._pending):
            if key[1] in attempted_track_ids and key not in accepted_pairs:
                del self._pending[key]

    def _bind_match(
        self,
        match: GalleryRecognitionMatch,
        frame_index: int,
    ) -> bool:
        person = self.gallery_adapter.get_identity(match.person_id)
        if person is None:
            return False
        track = match.candidate.track
        existing_target = self.target_manager.target_for_track(track.track_id)
        created_target = False
        if existing_target is None:
            try:
                # Gallery features are only cold-start recognition evidence.
                # Once recognized, initialize the runtime target exactly like
                # manual S selection: one live reference and its centroid.
                existing_target = self.target_manager.select(
                    track,
                    match.candidate.embedding,
                    frame_index,
                )
                created_target = True
            except (TypeError, ValueError):
                LOGGER.warning(
                    "GALLERY_RECOGNITION_REJECTED identity=%s track=%d reason=invalid_reference",
                    self.gallery_adapter.identity_label(match.person_id),
                    track.track_id,
                )
                return False

        try:
            self.gallery_adapter.attach_session_target(
                existing_target.target_id,
                match.person_id,
            )
        except (KeyError, ValueError):
            if created_target:
                self.target_manager.remove_by_track_id(track.track_id)
            LOGGER.info(
                "GALLERY_RECOGNITION_REJECTED identity=%s track=%d reason=association_conflict",
                self.gallery_adapter.identity_label(match.person_id),
                track.track_id,
            )
            return False

        LOGGER.info(
            "GALLERY_RECOGNIZED frame=%d identity_id=%s target_id=%d "
            "track_id=%d similarity=%.4f",
            frame_index,
            self.gallery_adapter.identity_label(match.person_id),
            existing_target.target_id,
            track.track_id,
            match.similarity,
        )
        return True

    def _crop(self, frame: np.ndarray, track: Track) -> np.ndarray | None:
        return crop_person(
            frame,
            track.bbox,
            min_crop_width=self.reid_config.min_crop_width,
            min_crop_height=self.reid_config.min_crop_height,
        )


def _person_label(person_id: int) -> str:
    return f"P{person_id:03d}"
