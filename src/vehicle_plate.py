"""Optional vehicle-plate evidence for Vehicle identity decisions.

The provider boundary is deliberately small: OCR/detection implementations return
``PlateObservation`` objects and the Vehicle business layer only consumes stable,
quality-gated plate strings.  This module is safe to import without HyperLPR3;
the optional provider is imported only when it is explicitly enabled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import logging
import math
from numbers import Real
from typing import Any, Protocol

import numpy as np

try:  # Keep Gallery/SQLite imports usable in lightweight tooling environments.
    import cv2
except ImportError:  # pragma: no cover - exercised only without OpenCV installed
    cv2 = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


def normalize_plate_number(value: str | None) -> str | None:
    """Normalize OCR text without imposing a country-specific plate format."""

    if value is None:
        return None
    normalized = "".join(str(value).strip().split()).upper()
    return normalized or None


@dataclass(frozen=True)
class PlateObservation:
    """Provider-independent plate result.

    ``bbox`` uses vehicle-crop-local ``(x1, y1, x2, y2)`` coordinates.
    """

    plate_number: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass
class VehiclePlateState:
    """Track-scoped pending state and its confirmed plate, if any."""

    stable_plate: str | None = None
    pending_counts: dict[str, int] = field(default_factory=dict)
    pending_last_frame: int | None = None
    last_inference_frame: int | None = None
    last_observation_frame: int | None = None
    last_confidence: float = 0.0
    last_sharpness: float = 0.0


@dataclass(frozen=True)
class PlateConfirmation:
    track_id: int
    plate_number: str
    frame_index: int
    confidence: float
    sharpness: float


class PlateRecognitionProvider(Protocol):
    """Small replaceable interface for plate detection/OCR providers."""

    def recognize(self, vehicle_crop: np.ndarray) -> Sequence[PlateObservation]:
        ...


class PlateProviderError(RuntimeError):
    """Raised when an explicitly enabled provider cannot be initialized."""


class HyperLPR3PlateProvider:
    """Adapter for the optional HyperLPR3 ``LicensePlateCatcher`` API."""

    def __init__(self) -> None:
        try:
            import hyperlpr3 as lpr3
        except ImportError as exc:
            raise PlateProviderError(
                "vehicle_plate provider 'hyperlpr3' is enabled but HyperLPR3 "
                "is not installed"
            ) from exc
        try:
            self._catcher = lpr3.LicensePlateCatcher()
        except Exception as exc:
            raise PlateProviderError(
                "failed to initialize HyperLPR3 LicensePlateCatcher"
            ) from exc

    def recognize(self, vehicle_crop: np.ndarray) -> tuple[PlateObservation, ...]:
        raw_results = self._catcher(vehicle_crop)
        if raw_results is None:
            return ()
        if (
            isinstance(raw_results, Mapping)
            or isinstance(raw_results, (str, bytes))
            or hasattr(raw_results, "plate_number")
        ):
            raw_results = (raw_results,)
        elif _looks_like_single_result(raw_results):
            raw_results = (raw_results,)
        observations: list[PlateObservation] = []
        for raw in raw_results:
            observation = _adapt_provider_result(raw)
            if observation is not None:
                observations.append(observation)
        return tuple(observations)


def create_plate_provider(provider_name: str) -> PlateRecognitionProvider:
    """Create an explicitly requested provider, failing loudly if unsupported."""

    normalized = str(provider_name).strip().lower()
    if normalized == "hyperlpr3":
        return HyperLPR3PlateProvider()
    raise PlateProviderError(f"unsupported vehicle plate provider: {provider_name}")


class VehiclePlateCoordinator:
    """Quality-gate and confirm plate evidence for Vehicle tracks and targets."""

    def __init__(
        self,
        config: Any,
        provider: PlateRecognitionProvider | None = None,
        *,
        vehicle_class_ids: Sequence[int] = (2, 5, 7),
    ) -> None:
        self.config = config
        self.provider = provider
        self.vehicle_class_ids = frozenset(int(value) for value in vehicle_class_ids)
        self._track_states: dict[int, VehiclePlateState] = {}
        self._target_plates: dict[int, str] = {}
        self._track_to_target: dict[int, int] = {}
        self._last_seen_frame: dict[int, int] = {}
        self._track_ages: dict[int, int] = {}
        self._round_robin_cursor = 0

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "enabled", False))

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: Sequence[Any],
        frame_index: int,
    ) -> tuple[PlateConfirmation, ...]:
        """Run at most the configured number of new plate inferences."""

        if not self.enabled or self.provider is None:
            return ()
        visible = [
            track
            for track in tracks
            if int(track.class_id) in self.vehicle_class_ids
        ]
        for track in visible:
            track_id = int(track.track_id)
            previous = self._last_seen_frame.get(track_id)
            state = self._track_states.setdefault(track_id, VehiclePlateState())
            self._last_seen_frame[track_id] = frame_index
            self._track_ages[track_id] = (
                self._track_ages.get(track_id, 0) + 1
                if previous is not None and frame_index == previous + 1
                else 1
            )
            if previous is None or frame_index != previous + 1:
                state.pending_counts.clear()
                state.pending_last_frame = None
                if track_id not in self._track_to_target:
                    state.stable_plate = None

        self._prune_states(frame_index)
        candidates = [
            track
            for track in visible
            if self._eligible_for_inference(track, frame_index)
        ]
        candidates.sort(key=lambda track: int(track.track_id))
        if candidates:
            offset = self._round_robin_cursor % len(candidates)
            candidates = candidates[offset:] + candidates[:offset]

        budget = max(0, int(getattr(self.config, "max_new_plate_inferences_per_frame", 1)))
        confirmations: list[PlateConfirmation] = []
        for track in candidates[:budget]:
            track_id = int(track.track_id)
            self._round_robin_cursor += 1
            state = self._track_states[track_id]
            state.last_inference_frame = frame_index
            crop = _crop_from_track(frame, track)
            if crop is None:
                continue
            observations = self.provider.recognize(crop)
            accepted: list[tuple[PlateObservation, float]] = []
            for observation in observations:
                quality = self._quality(crop, observation)
                if quality is None:
                    continue
                accepted.append((observation, quality))
            if not accepted:
                self._expire_pending(state, frame_index)
                continue
            observation, sharpness = max(accepted, key=lambda item: item[0].confidence)
            confirmation = self._accept_observation(
                track_id, state, observation, sharpness, frame_index
            )
            if confirmation is not None:
                confirmations.append(confirmation)
                LOGGER.info(
                    "VEHICLE_PLATE_CONFIRMED frame=%d track_id=%d target_id=%s plate=%s",
                    frame_index,
                    track_id,
                    self._track_to_target.get(track_id),
                    confirmation.plate_number,
                )
            else:
                LOGGER.debug(
                    "VEHICLE_PLATE_OBSERVED frame=%d track_id=%d plate=%s "
                    "confidence=%.4f sharpness=%.2f",
                    frame_index,
                    track_id,
                    normalize_plate_number(observation.plate_number),
                    observation.confidence,
                    sharpness,
                )
        return tuple(confirmations)

    def track_plate(self, track_id: int) -> str | None:
        state = self._track_states.get(int(track_id))
        return None if state is None else state.stable_plate

    def target_plate(self, target_id: int) -> str | None:
        return self._target_plates.get(int(target_id))

    def set_target_plate(self, target_id: int, plate_number: str | None) -> None:
        normalized = normalize_plate_number(plate_number)
        if normalized is None:
            return
        current = self._target_plates.get(int(target_id))
        if current is None:
            self._target_plates[int(target_id)] = normalized

    def bind_target_to_track(self, target_id: int, track_id: int) -> str | None:
        """Keep target plate identity across manual selection/recovery/re-entry."""

        target_id = int(target_id)
        track_id = int(track_id)
        self._track_to_target[track_id] = target_id
        current = self._target_plates.get(target_id)
        track_plate = self.track_plate(track_id)
        if current is None and track_plate is not None:
            self._target_plates[target_id] = track_plate
            current = track_plate
        elif current is not None:
            state = self._track_states.setdefault(track_id, VehiclePlateState())
            if state.stable_plate is None:
                state.stable_plate = current
        return current

    def forget_target(self, target_id: int) -> None:
        target_id = int(target_id)
        self._target_plates.pop(target_id, None)
        for track_id, mapped_target_id in list(self._track_to_target.items()):
            if mapped_target_id == target_id:
                del self._track_to_target[track_id]

    def forget_track(self, track_id: int) -> None:
        track_id = int(track_id)
        self._track_states.pop(track_id, None)
        self._last_seen_frame.pop(track_id, None)
        self._track_ages.pop(track_id, None)
        self._track_to_target.pop(track_id, None)

    def _eligible_for_inference(self, track: Any, frame_index: int) -> bool:
        track_id = int(track.track_id)
        state = self._track_states.setdefault(track_id, VehiclePlateState())
        if state.stable_plate is not None:
            return False
        age = self._track_ages.get(track_id, 0)
        if age < int(getattr(self.config, "min_track_age_frames", 3)):
            return False
        last = state.last_inference_frame
        interval = max(1, int(getattr(self.config, "inference_interval_frames", 5)))
        return last is None or frame_index - last >= interval

    def _quality(
        self, vehicle_crop: np.ndarray, observation: PlateObservation
    ) -> float | None:
        plate = normalize_plate_number(observation.plate_number)
        if plate is None or not math.isfinite(float(observation.confidence)):
            return None
        if float(observation.confidence) < float(self.config.min_confidence):
            return None
        try:
            x1, y1, x2, y2 = (int(value) for value in observation.bbox)
        except (TypeError, ValueError):
            return None
        height, width = vehicle_crop.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        if x2 - x1 < int(self.config.min_plate_width):
            return None
        if y2 - y1 < int(self.config.min_plate_height):
            return None
        plate_crop = vehicle_crop[y1:y2, x1:x2]
        if plate_crop.size == 0:
            return None
        if cv2 is None:
            raise PlateProviderError(
                "OpenCV is required for Vehicle plate sharpness validation"
            )
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if plate_crop.ndim == 3 else plate_crop
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if not math.isfinite(sharpness) or sharpness < float(self.config.min_sharpness):
            return None
        return sharpness

    def _accept_observation(
        self,
        track_id: int,
        state: VehiclePlateState,
        observation: PlateObservation,
        sharpness: float,
        frame_index: int,
    ) -> PlateConfirmation | None:
        plate = normalize_plate_number(observation.plate_number)
        if plate is None:
            return None
        self._expire_pending(state, frame_index)
        state.pending_counts[plate] = state.pending_counts.get(plate, 0) + 1
        state.pending_last_frame = frame_index
        state.last_observation_frame = frame_index
        state.last_confidence = float(observation.confidence)
        state.last_sharpness = sharpness
        best_plate, best_hits = max(
            state.pending_counts.items(), key=lambda item: (item[1], item[0])
        )
        if best_hits < int(self.config.confirmation_hits):
            return None
        if state.stable_plate == best_plate:
            state.pending_counts.clear()
            return None
        state.stable_plate = best_plate
        state.pending_counts.clear()
        return PlateConfirmation(
            track_id=track_id,
            plate_number=best_plate,
            frame_index=frame_index,
            confidence=float(observation.confidence),
            sharpness=sharpness,
        )

    def _expire_pending(self, state: VehiclePlateState, frame_index: int) -> None:
        last = state.pending_last_frame
        if last is not None and frame_index - last > int(self.config.pending_max_age_frames):
            state.pending_counts.clear()
            state.pending_last_frame = None

    def _prune_states(self, frame_index: int) -> None:
        max_age = max(1, int(self.config.pending_max_age_frames))
        for track_id, last_seen in list(self._last_seen_frame.items()):
            if frame_index - last_seen > max_age:
                self.forget_track(track_id)


def _crop_from_track(frame: np.ndarray, track: Any) -> np.ndarray | None:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    try:
        x1, y1, x2, y2 = (int(round(value)) for value in track.bbox)
    except (TypeError, ValueError):
        return None
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop.copy() if crop.size else None


def _adapt_provider_result(raw: Any) -> PlateObservation | None:
    """Convert common HyperLPR3 result shapes without exposing them upstream."""

    plate: Any = None
    confidence: Any = None
    bbox: Any = None
    if isinstance(raw, Mapping):
        plate = raw.get("plate_number", raw.get("text", raw.get("code")))
        confidence = raw.get("confidence", raw.get("score"))
        bbox = raw.get("bbox", raw.get("box"))
    else:
        plate = getattr(raw, "plate_number", getattr(raw, "text", None))
        confidence = getattr(raw, "confidence", getattr(raw, "score", None))
        bbox = getattr(raw, "bbox", getattr(raw, "box", None))
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values = list(raw)
            string_values = [value for value in values if isinstance(value, str)]
            if string_values:
                plate = string_values[0]
            numeric_values = [value for value in values if isinstance(value, Real)]
            if numeric_values:
                confidence = numeric_values[0]
                if bbox is None and len(numeric_values) >= 5:
                    bbox = numeric_values[1:5]
            for value in values:
                if isinstance(value, np.ndarray):
                    flattened = value.reshape(-1).tolist()
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    flattened = list(value)
                else:
                    continue
                if len(flattened) == 4 and all(
                    isinstance(item, Real) for item in flattened
                ):
                    bbox = flattened
                    break
    normalized = normalize_plate_number(plate if isinstance(plate, str) else None)
    if normalized is None or bbox is None:
        return None
    try:
        confidence_value = float(confidence)
        coordinates = tuple(int(round(float(value))) for value in bbox)
    except (TypeError, ValueError):
        return None
    if len(coordinates) != 4:
        return None
    return PlateObservation(normalized, confidence_value, coordinates)


def _looks_like_single_result(raw: Any) -> bool:
    """Recognize ``(plate, confidence, bbox)`` before iterating its fields."""

    if isinstance(raw, np.ndarray):
        values = raw.tolist()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        return False
    return (
        any(isinstance(value, str) for value in values)
        and any(isinstance(value, Real) for value in values)
        and any(
            isinstance(value, np.ndarray)
            or (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes))
                and len(value) == 4
            )
            for value in values
        )
    )
