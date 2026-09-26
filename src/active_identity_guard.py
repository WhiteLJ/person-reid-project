"""Protect ACTIVE Person bindings from local bbox/physical-identity drift."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from logging import getLogger
from time import perf_counter

import numpy as np

from .config import (
    ActiveIdentityGuardConfig,
    ReIDConfig,
    ReIDQualityConfig,
    ReIDRecoveryConfig,
)
from .models import SessionTarget, Track
from .reid import ReIDExtractor
from .reid_frame_cache import ReIDFrameCache
from .reid_quality import ReIDQualityResult, assess_reid_quality
from .target_manager import TargetManager
from .target_recovery import recovery_evidence


LOGGER = getLogger(__name__)


@dataclass(frozen=True)
class BboxGeometry:
    """Positive geometry metrics used by the guard's stable history."""

    bbox: tuple[float, float, float, float]
    center_x: float
    center_y: float
    width: float
    height: float
    area: float
    aspect_ratio: float


@dataclass
class _GuardState:
    track_id: int
    history: list[BboxGeometry] = field(default_factory=list)
    ambiguous: bool = False
    overlap_track_ids: tuple[int, ...] = ()
    verification_track_id: int | None = None
    verification_hits: int = 0
    failed_clean_frames: int = 0


@dataclass(frozen=True)
class ActiveIdentityGuardResult:
    """Per-frame guard result consumed by the Person recovery coordinator."""

    blocked_reference_update_target_ids: frozenset[int] = frozenset()
    corrected_target_ids: frozenset[int] = frozenset()
    identity_lost_target_ids: frozenset[int] = frozenset()
    reid_ms: float = 0.0


def bbox_geometry(bbox: Sequence[float]) -> BboxGeometry:
    """Convert an xyxy bbox into finite geometry metrics."""

    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    values = np.asarray((x1, y1, x2, y2), dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("bbox must be finite")
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    if width <= 0.0 or height <= 0.0 or area <= 0.0:
        raise ValueError("bbox must have positive area")
    return BboxGeometry(
        bbox=(x1, y1, x2, y2),
        center_x=(x1 + x2) / 2.0,
        center_y=(y1 + y2) / 2.0,
        width=width,
        height=height,
        area=area,
        aspect_ratio=width / height,
    )


def min_area_overlap_ratio(first: Sequence[float], second: Sequence[float]) -> float:
    """Return intersection / min(area(first), area(second))."""

    left = bbox_geometry(first)
    right = bbox_geometry(second)
    intersection = max(
        0.0,
        min(left.bbox[2], right.bbox[2]) - max(left.bbox[0], right.bbox[0]),
    ) * max(
        0.0,
        min(left.bbox[3], right.bbox[3]) - max(left.bbox[1], right.bbox[1]),
    )
    denominator = min(left.area, right.area)
    return intersection / denominator if denominator > 0.0 else 0.0


class ActiveIdentityGuard:
    """Detect and locally correct suspicious ACTIVE Person bindings.

    The guard performs no ReID work on ordinary frames.  It only extracts
    embeddings after a selected target has both a geometry anomaly and a
    meaningful min-area overlap with another Person.
    """

    def __init__(
        self,
        target_manager: TargetManager,
        reid_extractor: ReIDExtractor,
        reid_config: ReIDConfig,
        recovery_config: ReIDRecoveryConfig,
        quality_config: ReIDQualityConfig,
        guard_config: ActiveIdentityGuardConfig,
        *,
        embedding_cache: ReIDFrameCache | None = None,
        person_class_id: int = 0,
        quality_assessor: object | None = None,
    ) -> None:
        self.target_manager = target_manager
        self.reid_extractor = reid_extractor
        self.reid_config = reid_config
        self.recovery_config = recovery_config
        self.quality_config = quality_config
        self.guard_config = guard_config
        self.embedding_cache = embedding_cache
        self.person_class_id = person_class_id
        self.quality_assessor = quality_assessor
        self._states: dict[int, _GuardState] = {}
        self._blocked_target_ids: frozenset[int] = frozenset()
        self.last_result = ActiveIdentityGuardResult()

    @property
    def blocked_reference_update_target_ids(self) -> frozenset[int]:
        return self._blocked_target_ids

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        frame_index: int,
    ) -> ActiveIdentityGuardResult:
        """Inspect active Person targets and return reference-update blocks."""

        self._blocked_target_ids = frozenset()
        self.last_result = ActiveIdentityGuardResult()
        if not self.guard_config.enabled:
            return self.last_result
        if self.embedding_cache is not None:
            self.embedding_cache.begin_frame(frame_index)

        active_targets = tuple(self.target_manager.active_targets())
        active_ids = {target.target_id for target in active_targets}
        for target_id in list(self._states):
            if target_id not in active_ids:
                del self._states[target_id]

        tracks_by_id = {track.track_id: track for track in tracks}
        blocked: set[int] = set()
        corrected: set[int] = set()
        identity_lost: set[int] = set()
        reid_ms = 0.0

        for target in active_targets:
            track_id = target.current_track_id
            if track_id is None:
                continue
            track = tracks_by_id.get(track_id)
            if track is None or track.class_id != self.person_class_id:
                continue

            try:
                current_geometry = bbox_geometry(track.bbox)
            except ValueError:
                continue
            state = self._states.get(target.target_id)
            if state is None or state.track_id != track_id:
                state = _GuardState(track_id=track_id)
                self._states[target.target_id] = state

            if state.ambiguous:
                blocked.add(target.target_id)
                still_ambiguous = self._max_overlap_with_people(track, tracks) >= (
                    self.guard_config.clear_overlap_ratio
                )
                if still_ambiguous:
                    continue

                verification_candidates = self._verification_candidates(
                    target,
                    state,
                    tracks_by_id,
                )
                valid_candidates: list[tuple[Track, np.ndarray]] = []
                for candidate in verification_candidates:
                    quality = self._quality(frame, candidate, tracks)
                    if not quality.accepted or quality.crop is None:
                        LOGGER.debug(
                            "IDENTITY_GUARD_REJECTED frame=%d target_id=%d "
                            "track_id=%d reason=%s",
                            frame_index,
                            target.target_id,
                            candidate.track_id,
                            quality.reason or "quality_rejected",
                        )
                        continue
                    valid_candidates.append((candidate, quality.crop))

                if not valid_candidates:
                    state.verification_track_id = None
                    state.verification_hits = 0
                    state.failed_clean_frames = 0
                    continue

                embeddings_started = perf_counter()
                resolved = self._embeddings_for_candidates(
                    valid_candidates,
                    frame_index,
                )
                reid_ms += (perf_counter() - embeddings_started) * 1000.0
                scored: list[tuple[float, float, Track, np.ndarray]] = []
                for candidate, embedding in resolved:
                    try:
                        centroid, support = recovery_evidence(
                            target,
                            embedding,
                            reference_support_top_k=(
                                self.recovery_config.recovery_reference_support_top_k
                            ),
                        )
                    except ValueError:
                        continue
                    scored.append((centroid, support, candidate, embedding))

                if not scored:
                    state.verification_track_id = None
                    state.verification_hits = 0
                    state.failed_clean_frames = 0
                    continue
                scored.sort(key=lambda item: (-item[0], item[2].track_id))
                best = scored[0]
                second_score = scored[1][0] if len(scored) > 1 else None
                margin_ok = (
                    second_score is None
                    or best[0] - second_score >= self.recovery_config.recovery_margin
                )
                evidence_ok = (
                    best[0] >= self.recovery_config.recovery_threshold
                    and best[1]
                    >= self.recovery_config.recovery_reference_support_threshold
                    and margin_ok
                )
                if evidence_ok:
                    state.failed_clean_frames = 0
                    if state.verification_track_id == best[2].track_id:
                        state.verification_hits += 1
                    else:
                        state.verification_track_id = best[2].track_id
                        state.verification_hits = 1
                    LOGGER.debug(
                        "IDENTITY_GUARD_PENDING frame=%d target_id=%d "
                        "track_id=%d centroid=%.4f support=%.4f margin_ok=%s "
                        "hits=%d/%d",
                        frame_index,
                        target.target_id,
                        best[2].track_id,
                        best[0],
                        best[1],
                        margin_ok,
                        state.verification_hits,
                        self.guard_config.confirmation_hits,
                    )
                    if state.verification_hits >= self.guard_config.confirmation_hits:
                        old_track_id = target.current_track_id
                        if old_track_id != best[2].track_id:
                            old_candidate_centroid = next(
                                (
                                    item[0]
                                    for item in scored
                                    if item[2].track_id == old_track_id
                                ),
                                float("nan"),
                            )
                            self.target_manager.correct_active_binding(
                                target.target_id,
                                old_track_id=old_track_id,  # type: ignore[arg-type]
                                new_track=best[2],
                                expected_class_id=self.person_class_id,
                            )
                            LOGGER.info(
                                "TARGET_TRACK_CORRECTED frame=%d target_id=%d "
                                "old_track_id=%d new_track_id=%d "
                                "old_candidate_centroid=%.4f "
                                "new_candidate_centroid=%.4f "
                                "new_candidate_support=%.4f margin=%.4f",
                                frame_index,
                                target.target_id,
                                old_track_id,
                                best[2].track_id,
                                old_candidate_centroid,
                                best[0],
                                best[1],
                                (
                                    best[0] - second_score
                                    if second_score is not None
                                    else 0.0
                                ),
                            )
                            corrected.add(target.target_id)
                        self._reset_after_verification(
                            state,
                            best[2].track_id,
                            current_geometry if best[2].track_id == track_id else None,
                        )
                        blocked.add(target.target_id)
                    continue

                state.verification_track_id = None
                state.verification_hits = 0
                state.failed_clean_frames += 1
                LOGGER.debug(
                    "IDENTITY_GUARD_REJECTED frame=%d target_id=%d "
                    "track_id=%d centroid=%.4f support=%.4f margin_ok=%s "
                    "clean_failures=%d/%d",
                    frame_index,
                    target.target_id,
                    best[2].track_id,
                    best[0],
                    best[1],
                    margin_ok,
                    state.failed_clean_frames,
                    self.guard_config.confirmation_hits,
                )
                if state.failed_clean_frames >= self.guard_config.confirmation_hits:
                    old_track_id = target.current_track_id
                    self.target_manager.invalidate_active_binding(
                        target.target_id,
                        expected_track_id=old_track_id,
                    )
                    LOGGER.info(
                        "TARGET_IDENTITY_LOST frame=%d target_id=%d "
                        "old_track_id=%d reason=post_occlusion_identity_mismatch",
                        frame_index,
                        target.target_id,
                        old_track_id,
                    )
                    identity_lost.add(target.target_id)
                    self._states.pop(target.target_id, None)
                continue

            try:
                quality = self._quality(frame, track, tracks)
            except (TypeError, ValueError):
                continue
            geometry_anomalies = self._geometry_anomalies(
                current_geometry,
                state.history,
            )
            overlap_partners = self._overlap_partners(track, tracks)
            max_overlap = overlap_partners[0][1] if overlap_partners else 0.0
            geometry_anomalous = any(geometry_anomalies.values())
            if geometry_anomalous and max_overlap >= self.guard_config.overlap_trigger_ratio:
                state.ambiguous = True
                state.overlap_track_ids = self._select_overlap_candidates(
                    track.track_id,
                    overlap_partners,
                )
                state.verification_track_id = None
                state.verification_hits = 0
                state.failed_clean_frames = 0
                blocked.add(target.target_id)
                LOGGER.info(
                    "IDENTITY_GUARD_AMBIGUOUS frame=%d target_id=%d track_id=%d "
                    "width_ratio=%.3f area_ratio=%.3f center_shift=%.3f "
                    "min_area_overlap=%.3f overlap_tracks=%s",
                    frame_index,
                    target.target_id,
                    track.track_id,
                    geometry_anomalies["width_ratio"],
                    geometry_anomalies["area_ratio"],
                    geometry_anomalies["center_shift"],
                    max_overlap,
                    state.overlap_track_ids,
                )
                continue

            if quality.accepted and not geometry_anomalous:
                state.history.append(current_geometry)
                del state.history[: -self.guard_config.bbox_history_frames]

        self._blocked_target_ids = frozenset(blocked)
        self.last_result = ActiveIdentityGuardResult(
            blocked_reference_update_target_ids=self._blocked_target_ids,
            corrected_target_ids=frozenset(corrected),
            identity_lost_target_ids=frozenset(identity_lost),
            reid_ms=reid_ms,
        )
        return self.last_result

    def _quality(
        self,
        frame: np.ndarray,
        track: Track,
        tracks: Sequence[Track],
    ) -> ReIDQualityResult:
        if self.quality_assessor is not None:
            return self.quality_assessor(frame, track, tracks)  # type: ignore[operator]
        return assess_reid_quality(
            frame,
            track,
            tracks,
            self.reid_config,
            self.quality_config,
            person_class_id=self.person_class_id,
        )

    def _geometry_anomalies(
        self,
        current: BboxGeometry,
        history: Sequence[BboxGeometry],
    ) -> dict[str, float | bool]:
        if not history:
            return {"width_ratio": 1.0, "area_ratio": 1.0, "center_shift": 0.0}
        widths = np.asarray([item.width for item in history], dtype=np.float64)
        areas = np.asarray([item.area for item in history], dtype=np.float64)
        centers = np.asarray(
            [(item.center_x, item.center_y) for item in history],
            dtype=np.float64,
        )
        diagonals = np.asarray(
            [np.hypot(item.width, item.height) for item in history],
            dtype=np.float64,
        )
        median_width = float(np.median(widths))
        median_area = float(np.median(areas))
        median_center = np.median(centers, axis=0)
        median_diagonal = max(float(np.median(diagonals)), 1e-6)
        width_ratio = current.width / max(median_width, 1e-6)
        area_ratio = current.area / max(median_area, 1e-6)
        center_shift = float(
            np.hypot(
                current.center_x - median_center[0],
                current.center_y - median_center[1],
            )
            / median_diagonal
        )
        return {
            "width_ratio": width_ratio,
            "area_ratio": area_ratio,
            "center_shift": center_shift,
            "width_anomaly": width_ratio > self.guard_config.max_width_growth_ratio,
            "area_anomaly": area_ratio > self.guard_config.max_area_growth_ratio,
            "center_anomaly": center_shift > self.guard_config.max_center_shift_ratio,
        }

    def _max_overlap_with_people(
        self,
        track: Track,
        tracks: Sequence[Track],
    ) -> float:
        return max(
            (
                min_area_overlap_ratio(track.bbox, other.bbox)
                for other in tracks
                if other.track_id != track.track_id
                and other.class_id == self.person_class_id
            ),
            default=0.0,
        )

    def _overlap_partners(
        self,
        track: Track,
        tracks: Sequence[Track],
    ) -> list[tuple[int, float]]:
        partners = [
            (other.track_id, min_area_overlap_ratio(track.bbox, other.bbox))
            for other in tracks
            if other.track_id != track.track_id
            and other.class_id == self.person_class_id
        ]
        return sorted(partners, key=lambda item: (-item[1], item[0]))

    def _select_overlap_candidates(
        self,
        current_track_id: int,
        partners: Sequence[tuple[int, float]],
    ) -> tuple[int, ...]:
        limit = max(1, self.guard_config.max_candidates)
        return tuple(
            [current_track_id]
            + [track_id for track_id, _ratio in partners[: max(0, limit - 1)]]
        )

    def _verification_candidates(
        self,
        target: SessionTarget,
        state: _GuardState,
        tracks_by_id: dict[int, Track],
    ) -> tuple[Track, ...]:
        candidates: list[Track] = []
        for track_id in state.overlap_track_ids:
            candidate = tracks_by_id.get(track_id)
            if candidate is None or candidate.class_id != self.person_class_id:
                continue
            owner = self.target_manager.target_for_track(candidate.track_id)
            if owner is not None and owner.target_id != target.target_id:
                LOGGER.debug(
                    "IDENTITY_GUARD_CANDIDATE_REJECTED target=%d track=%d "
                    "reason=occupied",
                    target.target_id,
                    candidate.track_id,
                )
                continue
            candidates.append(candidate)
        return tuple(candidates)

    def _embeddings_for_candidates(
        self,
        candidates: Sequence[tuple[Track, np.ndarray]],
        frame_index: int,
    ) -> list[tuple[Track, np.ndarray]]:
        resolved: list[tuple[Track, np.ndarray] | None] = [None] * len(candidates)
        missing_indices: list[int] = []
        missing_crops: list[np.ndarray] = []
        for index, (track, crop) in enumerate(candidates):
            cached = (
                self.embedding_cache.get(track.track_id, frame_index)
                if self.embedding_cache is not None
                else None
            )
            if cached is not None:
                resolved[index] = (track, cached)
            else:
                missing_indices.append(index)
                missing_crops.append(crop)
        if missing_crops:
            embeddings = np.asarray(
                self.reid_extractor.extract_batch(missing_crops),
                dtype=np.float32,
            )
            if embeddings.ndim != 2 or embeddings.shape[0] != len(missing_crops):
                raise ValueError("Identity Guard ReID output does not match candidates")
            for index, embedding in zip(missing_indices, embeddings):
                copy = np.asarray(embedding, dtype=np.float32).copy()
                track = candidates[index][0]
                if self.embedding_cache is not None:
                    self.embedding_cache.put(track.track_id, copy, frame_index)
                resolved[index] = (track, copy)
        return [item for item in resolved if item is not None]

    @staticmethod
    def _reset_after_verification(
        state: _GuardState,
        track_id: int,
        current_geometry: BboxGeometry | None,
    ) -> None:
        state.track_id = track_id
        state.ambiguous = False
        state.overlap_track_ids = ()
        state.verification_track_id = None
        state.verification_hits = 0
        state.failed_clean_frames = 0
        state.history = [current_geometry] if current_geometry is not None else []
