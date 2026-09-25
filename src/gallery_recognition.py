"""Conservative automatic recognition against the in-memory TargetGallery."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from logging import getLogger
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
    """Periodically recognize unbound Tracks against loaded Gallery people."""

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
        gallery_adapter: GalleryRecognitionAdapter | None = None,
    ) -> None:
        self.target_manager = target_manager
        self.gallery = gallery
        self.reid_extractor = reid_extractor
        self.reid_config = reid_config
        self.recognition_config = recognition_config
        self.recovery_config = recovery_config
        self.person_class_id = person_class_id
        self.track_class_id = (
            person_class_id if track_class_id is None else track_class_id
        )
        self.embedding_cache = embedding_cache
        self.quality_config = quality_config or ReIDQualityConfig()
        self.quality_assessor = quality_assessor
        self.gallery_adapter = gallery_adapter or person_gallery_adapter(gallery)
        self._track_ages: dict[int, int] = {}
        self._last_seen_frame: dict[int, int] = {}
        self._pending: dict[tuple[int, int], _PendingRecognition] = {}
        self._last_recognition_frame: int | None = None
        self.recognized_count = 0
        self.quality_rejected_count = 0
        self.reid_batch_count = 0

    @property
    def pending(self) -> dict[tuple[int, int], int]:
        """Return a read-only-by-convention snapshot useful for diagnostics/tests."""

        return {key: value.hits for key, value in self._pending.items()}

    @property
    def track_ages(self) -> dict[int, int]:
        return dict(self._track_ages)

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

        self._update_track_ages(tracks, frame_index)
        if self.embedding_cache is not None:
            self.embedding_cache.begin_frame(frame_index)

        if not self.recognition_config.enabled:
            return []
        if self._last_recognition_frame is not None and (
            frame_index - self._last_recognition_frame
            < self.recognition_config.recognition_interval_frames
        ):
            # In-between frames do not represent a recognition result and
            # therefore must not clear confirmation pending state.
            return []

        people = [
            person
            for person in self.gallery_adapter.all_identities()
            if self.gallery_adapter.identity_id(person)
            not in self.gallery_adapter.attached_identity_ids()
        ]
        if not people:
            return []

        candidates = self._build_candidates(
            frame,
            tracks,
            set(protected_track_ids),
        )
        if not candidates:
            return []

        candidates = self._ensure_embeddings(candidates, frame_index)
        if not candidates:
            return []

        # This is a real recognition attempt.  Only now may old pending
        # proposals be invalidated by the current result.
        self._last_recognition_frame = frame_index
        matches = assign_gallery_matches(
            people,
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

        recognized: list[GalleryRecognitionMatch] = []
        for match in matches:
            key = (match.person_id, match.candidate.track.track_id)
            pending = self._pending.get(key)
            hits = 1 if pending is None else pending.hits + 1
            if hits < self.recognition_config.confirmation_hits:
                self._pending[key] = _PendingRecognition(hits=hits)
                LOGGER.debug(
                    "GALLERY_RECOGNITION_PENDING identity=%s track=%d hits=%d/%d similarity=%.4f",
                    self.gallery_adapter.identity_label(match.person_id),
                    match.candidate.track.track_id,
                    hits,
                    self.recognition_config.confirmation_hits,
                    match.similarity,
                )
                continue

            if self._bind_match(match, frame_index):
                recognized.append(match)
                self.recognized_count += 1
            self._pending.pop(key, None)

        return recognized

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

    def _build_candidates(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        protected_track_ids: set[int],
    ) -> list[tuple[Track, np.ndarray]]:
        candidates: list[tuple[Track, np.ndarray]] = []
        for track in tracks:
            if track.class_id != self.track_class_id:
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
                    person_class_id=self.track_class_id,
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
