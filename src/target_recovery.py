"""MVP-5 session-target reference updates and conservative ReID recovery."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from logging import getLogger

import numpy as np

from .config import ReIDConfig, ReIDRecoveryConfig
from .models import SessionTarget, TargetState, Track
from .reid import ReIDExtractor, cosine_similarity, crop_person
from .reid_frame_cache import ReIDFrameCache
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
    """Coordinate target state, limited crop extraction, and recovery matching."""

    def __init__(
        self,
        target_manager: TargetManager,
        reid_extractor: ReIDExtractor,
        reid_config: ReIDConfig,
        recovery_config: ReIDRecoveryConfig,
        embedding_cache: ReIDFrameCache | None = None,
    ) -> None:
        self.target_manager = target_manager
        self.reid_extractor = reid_extractor
        self.reid_config = reid_config
        self.recovery_config = recovery_config
        self.embedding_cache = embedding_cache
        self.last_recovered_track_ids: frozenset[int] = frozenset()

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

        crop = self._crop(frame, track)
        if crop is None:
            LOGGER.info(
                "TARGET_SELECTION_REJECTED track=%d reason=invalid_or_small_crop",
                track.track_id,
            )
            return None

        embedding = self.reid_extractor.extract(crop)
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
        """Update visibility and run only due, eligible ReID work for a frame."""

        self.last_recovered_track_ids = frozenset()
        if self.embedding_cache is not None:
            self.embedding_cache.begin_frame(frame_index)

        self.target_manager.update_visibility(
            tracks,
            lost_grace_frames=self.recovery_config.lost_grace_frames,
            frame_index=frame_index,
        )

        embedding_jobs: list[tuple[str, int, Track | None, np.ndarray]] = []
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
            crop = self._crop(frame, track)
            if crop is not None:
                embedding_jobs.append(("reference", target.target_id, track, crop))

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

        candidate_tracks = self.target_manager.recovery_candidates(tracks)
        for track in (candidate_tracks if due_lost_targets else ()):
            crop = self._crop(frame, track)
            if crop is not None:
                embedding_jobs.append(("candidate", track.track_id, track, crop))

        if not embedding_jobs:
            return []

        embeddings = np.asarray(
            self.reid_extractor.extract_batch([job[3] for job in embedding_jobs]),
            dtype=np.float32,
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(embedding_jobs):
            raise ValueError("ReID batch output does not match requested jobs")

        recovery_candidates: list[RecoveryCandidate] = []
        for job, embedding in zip(embedding_jobs, embeddings):
            kind, owner_id, track, _crop = job
            if self.embedding_cache is not None and track is not None:
                self.embedding_cache.put(
                    track.track_id,
                    embedding,
                    frame_index,
                )
            if kind == "reference":
                self.target_manager.add_reference(
                    owner_id,
                    embedding,
                    frame_index,
                    self.recovery_config.max_reference_embeddings,
                    self.recovery_config.reference_update_threshold,
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
        for match in matches:
            self.target_manager.recover(
                target_id=match.target_id,
                track=match.candidate.track,
                embedding=match.candidate.embedding,
                similarity=match.similarity,
                frame_index=frame_index,
                max_reference_embeddings=self.recovery_config.max_reference_embeddings,
                reference_update_threshold=self.recovery_config.reference_update_threshold,
            )
        self.last_recovered_track_ids = frozenset(
            match.candidate.track.track_id for match in matches
        )
        return matches

    def _crop(self, frame: np.ndarray, track: Track) -> np.ndarray | None:
        return crop_person(
            frame,
            track.bbox,
            min_crop_width=self.reid_config.min_crop_width,
            min_crop_height=self.reid_config.min_crop_height,
        )
