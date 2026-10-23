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
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic, perf_counter
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


@dataclass(frozen=True)
class PlateOCRTask:
    """Immutable crop snapshot submitted to the background OCR worker."""

    task_id: int
    track_id: int
    frame_index: int
    track_epoch: int
    class_id: int
    bbox: tuple[float, float, float, float]
    target_id: int | None
    stable_plate_at_submit: str | None
    crop: np.ndarray


@dataclass(frozen=True)
class PlateOCRResult:
    task: PlateOCRTask
    observations: tuple[PlateObservation, ...]
    elapsed_ms: float
    error: str | None = None


@dataclass(frozen=True)
class VehiclePlateOCRStats:
    submitted: int
    completed: int
    failed: int
    dropped_tasks: int
    dropped_results: int
    total_elapsed_ms: float
    max_elapsed_ms: float


class VehiclePlateOCRWorker:
    """Run the provider outside the Qt/runtime processing thread.

    The input queue is deliberately size one.  When a newer task arrives while
    another task is waiting, the waiting task is discarded in favor of the new
    crop.  A task already inside the provider cannot be cancelled, but its
    result is later validated by ``VehiclePlateCoordinator`` before use.
    """

    def __init__(
        self,
        provider: PlateRecognitionProvider,
        *,
        max_pending_tasks: int = 1,
        result_queue_size: int = 8,
    ) -> None:
        if max_pending_tasks != 1:
            raise ValueError("VehiclePlateOCRWorker requires one pending task")
        self.provider = provider
        self._tasks: Queue[PlateOCRTask | None] = Queue(maxsize=max_pending_tasks)
        self._results: Queue[PlateOCRResult] = Queue(maxsize=result_queue_size)
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._next_task_id = 1
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._dropped_tasks = 0
        self._dropped_results = 0
        self._total_elapsed_ms = 0.0
        self._max_elapsed_ms = 0.0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = Thread(
                target=self._run,
                name="vehicle-plate-ocr",
                daemon=True,
            )
            self._thread.start()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def new_task(
        self,
        *,
        track_id: int,
        frame_index: int,
        track_epoch: int,
        class_id: int,
        bbox: tuple[float, float, float, float],
        target_id: int | None,
        stable_plate_at_submit: str | None,
        crop: np.ndarray,
    ) -> PlateOCRTask:
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
        return PlateOCRTask(
            task_id=task_id,
            track_id=int(track_id),
            frame_index=int(frame_index),
            track_epoch=int(track_epoch),
            class_id=int(class_id),
            bbox=tuple(float(value) for value in bbox),
            target_id=None if target_id is None else int(target_id),
            stable_plate_at_submit=stable_plate_at_submit,
            crop=np.asarray(crop).copy(),
        )

    def submit(self, task: PlateOCRTask) -> PlateOCRTask | None:
        """Submit newest task and return a replaced waiting task, if any."""

        if not self.running:
            raise RuntimeError("VehiclePlateOCRWorker is not running")
        dropped: PlateOCRTask | None = None
        try:
            dropped = self._tasks.get_nowait()
        except Empty:
            pass
        try:
            self._tasks.put_nowait(task)
        except Full:  # A worker can win the queue race; keep the newest task.
            try:
                dropped = self._tasks.get_nowait()
                self._tasks.put_nowait(task)
            except (Empty, Full):
                self._increment_dropped_tasks()
                return task
        with self._lock:
            self._submitted += 1
            if dropped is not None:
                self._dropped_tasks += 1
        return dropped

    def drain_results(self) -> tuple[PlateOCRResult, ...]:
        results: list[PlateOCRResult] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except Empty:
                return tuple(results)

    def stats(self) -> VehiclePlateOCRStats:
        with self._lock:
            return VehiclePlateOCRStats(
                submitted=self._submitted,
                completed=self._completed,
                failed=self._failed,
                dropped_tasks=self._dropped_tasks,
                dropped_results=self._dropped_results,
                total_elapsed_ms=self._total_elapsed_ms,
                max_elapsed_ms=self._max_elapsed_ms,
            )

    def close(self, timeout: float = 2.0) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        while True:
            try:
                self._tasks.get_nowait()
            except Empty:
                break
        try:
            self._tasks.put_nowait(None)
        except Full:
            pass
        thread.join(timeout=max(0.0, timeout))
        if not thread.is_alive():
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                task = self._tasks.get(timeout=0.1)
            except Empty:
                continue
            if task is None:
                return
            started = perf_counter()
            error: str | None = None
            observations: tuple[PlateObservation, ...] = ()
            try:
                observations = tuple(self.provider.recognize(task.crop))
            except Exception as exc:  # Provider errors must not kill runtime thread.
                error = f"{type(exc).__name__}: {exc}"
            elapsed_ms = (perf_counter() - started) * 1000.0
            with self._lock:
                self._completed += 1
                self._total_elapsed_ms += elapsed_ms
                self._max_elapsed_ms = max(self._max_elapsed_ms, elapsed_ms)
                if error is not None:
                    self._failed += 1
            result = PlateOCRResult(task, observations, elapsed_ms, error)
            try:
                self._results.put_nowait(result)
            except Full:
                try:
                    self._results.get_nowait()
                    self._results.put_nowait(result)
                except (Empty, Full):
                    with self._lock:
                        self._dropped_results += 1

    def _increment_dropped_tasks(self) -> None:
        with self._lock:
            self._dropped_tasks += 1


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
        ocr_worker: VehiclePlateOCRWorker | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.ocr_worker = ocr_worker
        self.vehicle_class_ids = frozenset(int(value) for value in vehicle_class_ids)
        self._track_states: dict[int, VehiclePlateState] = {}
        self._target_plates: dict[int, str] = {}
        self._track_to_target: dict[int, int] = {}
        self._last_seen_frame: dict[int, int] = {}
        self._track_ages: dict[int, int] = {}
        self._track_epochs: dict[int, int] = {}
        self._pending_tasks: dict[int, PlateOCRTask] = {}
        self._round_robin_cursor = 0
        self._last_global_inference_at: float | None = None
        self._first_ocr_at: float | None = None
        self._last_ocr_stats_log_at: float | None = None
        self._expired_results = 0

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "enabled", False))

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: Sequence[Any],
        frame_index: int,
    ) -> tuple[PlateConfirmation, ...]:
        """Advance plate state without blocking on the OCR provider."""

        if not self.enabled:
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
            if track_id not in self._track_epochs:
                self._track_epochs[track_id] = 1
            elif previous is None or frame_index != previous + 1:
                self._track_epochs[track_id] += 1
            if previous is None or frame_index != previous + 1:
                state.pending_counts.clear()
                state.pending_last_frame = None
                if track_id not in self._track_to_target:
                    state.stable_plate = None

        self._prune_states(frame_index)
        visible_by_id = {int(track.track_id): track for track in visible}
        if self.ocr_worker is not None:
            confirmations = self._consume_async_results(
                visible_by_id, frame_index
            )
            self._schedule_async_task(frame, visible, frame_index)
            self._maybe_log_ocr_stats()
            return tuple(confirmations)

        if self.provider is None:
            return ()
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

    def ocr_stats(self) -> dict[str, float | int]:
        """Return diagnostic counters without exposing mutable worker state."""

        if self.ocr_worker is None:
            return {
                "submitted": 0,
                "completed": 0,
                "failed": 0,
                "dropped_tasks": 0,
                "dropped_results": 0,
                "expired_results": self._expired_results,
                "actual_hz": 0.0,
                "total_elapsed_ms": 0.0,
                "max_elapsed_ms": 0.0,
            }
        stats = self.ocr_worker.stats()
        elapsed = (
            monotonic() - self._first_ocr_at
            if self._first_ocr_at is not None
            else 0.0
        )
        return {
            "submitted": stats.submitted,
            "completed": stats.completed,
            "failed": stats.failed,
            "dropped_tasks": stats.dropped_tasks,
            "dropped_results": stats.dropped_results,
            "expired_results": self._expired_results,
            "actual_hz": stats.completed / elapsed if elapsed > 0.0 else 0.0,
            "total_elapsed_ms": stats.total_elapsed_ms,
            "max_elapsed_ms": stats.max_elapsed_ms,
        }

    def _schedule_async_task(
        self, frame: np.ndarray, visible: Sequence[Any], frame_index: int
    ) -> None:
        if self.ocr_worker is None or not self.ocr_worker.running:
            return
        if int(getattr(self.config, "max_new_plate_inferences_per_frame", 1)) < 1:
            return
        now = monotonic()
        minimum_interval = max(
            0.0,
            float(getattr(self.config, "min_inference_interval_ms", 500)) / 1000.0,
        )
        if (
            self._last_global_inference_at is not None
            and now - self._last_global_inference_at < minimum_interval
        ):
            return
        candidates = [
            track
            for track in visible
            if int(track.track_id) not in self._pending_track_ids()
            and self._eligible_for_inference(track, frame_index)
        ]
        if not candidates:
            return
        candidates.sort(key=lambda track: int(track.track_id))
        offset = self._round_robin_cursor % len(candidates)
        candidates = candidates[offset:] + candidates[:offset]
        selected = candidates[0]
        crop = _crop_from_track(frame, selected)
        if crop is None:
            return
        track_id = int(selected.track_id)
        state = self._track_states[track_id]
        task = self.ocr_worker.new_task(
            track_id=track_id,
            frame_index=frame_index,
            track_epoch=self._track_epochs[track_id],
            class_id=int(selected.class_id),
            bbox=tuple(float(value) for value in selected.bbox),
            target_id=self._track_to_target.get(track_id),
            stable_plate_at_submit=state.stable_plate,
            crop=crop,
        )
        dropped = self.ocr_worker.submit(task)
        if dropped is not None:
            self._pending_tasks.pop(dropped.task_id, None)
        self._pending_tasks[task.task_id] = task
        state.last_inference_frame = frame_index
        self._round_robin_cursor += 1
        self._last_global_inference_at = now
        self._first_ocr_at = self._first_ocr_at or now

    def _consume_async_results(
        self, visible_by_id: dict[int, Any], frame_index: int
    ) -> list[PlateConfirmation]:
        if self.ocr_worker is None:
            return []
        confirmations: list[PlateConfirmation] = []
        for result in self.ocr_worker.drain_results():
            task = self._pending_tasks.pop(result.task.task_id, None)
            if task is None:
                self._expired_results += 1
                LOGGER.debug(
                    "VEHICLE_PLATE_OCR_RESULT_DROPPED task_id=%d reason=unknown_task",
                    result.task.task_id,
                )
                continue
            if result.error is not None:
                LOGGER.warning(
                    "VEHICLE_PLATE_OCR_FAILED task_id=%d track_id=%d "
                    "elapsed_ms=%.2f error=%s",
                    task.task_id,
                    task.track_id,
                    result.elapsed_ms,
                    result.error,
                )
                continue
            reason = self._stale_result_reason(
                task, visible_by_id.get(task.track_id), frame_index
            )
            if reason is not None:
                self._expired_results += 1
                LOGGER.info(
                    "VEHICLE_PLATE_OCR_RESULT_DROPPED task_id=%d track_id=%d "
                    "task_frame=%d current_frame=%d reason=%s elapsed_ms=%.2f",
                    task.task_id,
                    task.track_id,
                    task.frame_index,
                    frame_index,
                    reason,
                    result.elapsed_ms,
                )
                continue
            confirmations.extend(
                self._apply_observations(task, result.observations, result.elapsed_ms)
            )
        return confirmations

    def _apply_observations(
        self,
        task: PlateOCRTask,
        observations: Sequence[PlateObservation],
        elapsed_ms: float,
    ) -> list[PlateConfirmation]:
        state = self._track_states.get(task.track_id)
        if state is None:
            return []
        accepted: list[tuple[PlateObservation, float]] = []
        for observation in observations:
            sharpness = self._quality(task.crop, observation)
            if sharpness is not None:
                accepted.append((observation, sharpness))
        if not accepted:
            self._expire_pending(state, task.frame_index)
            return []
        observation, sharpness = max(accepted, key=lambda item: item[0].confidence)
        confirmation = self._accept_observation(
            task.track_id, state, observation, sharpness, task.frame_index
        )
        if confirmation is not None:
            LOGGER.info(
                "VEHICLE_PLATE_CONFIRMED frame=%d track_id=%d target_id=%s "
                "plate=%s ocr_ms=%.2f",
                task.frame_index,
                task.track_id,
                self._track_to_target.get(task.track_id),
                confirmation.plate_number,
                elapsed_ms,
            )
        else:
            LOGGER.debug(
                "VEHICLE_PLATE_OBSERVED frame=%d track_id=%d plate=%s "
                "confidence=%.4f sharpness=%.2f ocr_ms=%.2f",
                task.frame_index,
                task.track_id,
                normalize_plate_number(observation.plate_number),
                observation.confidence,
                sharpness,
                elapsed_ms,
            )
        return [] if confirmation is None else [confirmation]

    def _stale_result_reason(
        self, task: PlateOCRTask, current_track: Any | None, frame_index: int
    ) -> str | None:
        if current_track is None:
            return "track_missing"
        if int(current_track.class_id) not in self.vehicle_class_ids:
            return "class_changed"
        if frame_index - task.frame_index > int(
            getattr(self.config, "max_result_age_frames", 15)
        ):
            return "result_expired"
        if self._track_epochs.get(task.track_id) != task.track_epoch:
            return "track_epoch_changed"
        current_target = self._track_to_target.get(task.track_id)
        if current_target != task.target_id and task.target_id is not None:
            return "target_changed"
        state = self._track_states.get(task.track_id)
        if (
            state is not None
            and state.stable_plate is not None
            and state.stable_plate != task.stable_plate_at_submit
        ):
            return "plate_state_changed"
        return None

    def _pending_track_ids(self) -> frozenset[int]:
        return frozenset(task.track_id for task in self._pending_tasks.values())

    def _maybe_log_ocr_stats(self) -> None:
        if self.ocr_worker is None:
            return
        now = monotonic()
        if (
            self._last_ocr_stats_log_at is not None
            and now - self._last_ocr_stats_log_at < 5.0
        ):
            return
        stats = self.ocr_stats()
        if int(stats["submitted"]) == 0:
            return
        self._last_ocr_stats_log_at = now
        LOGGER.info(
            "VEHICLE_PLATE_OCR_STATS submitted=%d completed=%d actual_hz=%.3f "
            "dropped_tasks=%d expired_results=%d avg_ms=%.2f max_ms=%.2f",
            stats["submitted"],
            stats["completed"],
            stats["actual_hz"],
            stats["dropped_tasks"],
            stats["expired_results"],
            stats["total_elapsed_ms"] / max(1, int(stats["completed"])),
            stats["max_elapsed_ms"],
        )

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
        self._track_epochs.pop(track_id, None)
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
