"""CPU BoT-SORT adapter for detections produced by an Ascend YOLO OM."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Collection, Sequence

import numpy as np
import yaml

from .models import Detection, Track


class _DetectionResults:
    """Minimal Results-like object required by Ultralytics BOTSORT.update."""

    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)
        if not (len(self.xyxy) == len(self.conf) == len(self.cls)):
            raise ValueError("detection adapter arrays must have matching lengths")

    @property
    def xywh(self) -> np.ndarray:
        result = np.empty_like(self.xyxy)
        result[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        result[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
        result[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        result[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return result

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, index: Any) -> "_DetectionResults":
        return _DetectionResults(self.xyxy[index], self.conf[index], self.cls[index])


def _resolve_tracker_file(value: str | Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    try:
        from ultralytics.utils.checks import check_yaml

        return Path(check_yaml(str(value)))
    except Exception as exc:
        raise FileNotFoundError(f"BoT-SORT profile not found: {value}") from exc


def _load_tracker_args(path: str | Path) -> dict[str, Any]:
    profile_path = _resolve_tracker_file(path)
    with profile_path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError(f"BoT-SORT profile must be a mapping: {profile_path}")
    if values.get("tracker_type", "botsort") != "botsort":
        raise ValueError("Ascend backend requires a BoT-SORT tracker profile")
    values.setdefault("track_high_thresh", 0.25)
    values.setdefault("track_low_thresh", 0.1)
    values.setdefault("new_track_thresh", 0.25)
    values.setdefault("track_buffer", 30)
    values.setdefault("match_thresh", 0.8)
    values.setdefault("fuse_score", True)
    values.setdefault("gmc_method", "sparseOptFlow")
    values.setdefault("proximity_thresh", 0.5)
    values.setdefault("appearance_thresh", 0.8)
    values.setdefault("with_reid", False)
    values.setdefault("model", "auto")
    if bool(values["with_reid"]):
        raise ValueError(
            "Ascend backend requires with_reid: false; "
            "Ultralytics BoT-SORT appearance ReID is not allowed on this path"
        )
    values["device"] = "cpu"
    return values


class AscendBotSortTracker:
    """Instantiate one installed Ultralytics BOTSORT over external detections.

    The default remains the historical Person-only class-0 adapter.  Atlas
    multiclass tracking creates a separate instance with its own class set.
    """

    def __init__(
        self,
        tracker_config: str | Path,
        *,
        persist: bool = True,
        class_ids: Collection[int] = (0,),
    ) -> None:
        try:
            from ultralytics.trackers.bot_sort import BOTSORT
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics 8.4.138 is required for the CPU BoT-SORT adapter"
            ) from exc

        values = _load_tracker_args(tracker_config)
        self.tracker = BOTSORT(SimpleNamespace(**values))
        self.persist = persist
        self.class_ids = tuple(sorted({int(class_id) for class_id in class_ids}))
        if not self.class_ids or any(class_id < 0 for class_id in self.class_ids):
            raise ValueError("class_ids must contain at least one non-negative class ID")
        self.last_output_count = 0

    def update(
        self,
        detections: Sequence[Detection],
        frame: np.ndarray,
        class_ids: Collection[int] | None = None,
    ) -> list[Track]:
        if not self.persist:
            self.tracker.reset()
        accepted_class_ids = self.class_ids if class_ids is None else tuple(
            sorted({int(class_id) for class_id in class_ids})
        )
        if not accepted_class_ids or any(class_id < 0 for class_id in accepted_class_ids):
            raise ValueError("class_ids must contain at least one non-negative class ID")
        selected_detections = [
            detection
            for detection in detections
            if detection.class_id in accepted_class_ids
        ]
        results = _DetectionResults(
            np.asarray([detection.bbox for detection in selected_detections], dtype=np.float32),
            np.asarray([detection.confidence for detection in selected_detections], dtype=np.float32),
            np.asarray([detection.class_id for detection in selected_detections], dtype=np.float32),
        )
        tracked = np.asarray(self.tracker.update(results, img=frame), dtype=np.float32)
        self.last_output_count = 0 if tracked.size == 0 else len(tracked)
        if tracked.size == 0:
            return []
        tracked = np.atleast_2d(tracked)
        tracks: list[Track] = []
        for row in tracked:
            if len(row) < 7:
                continue
            bbox = tuple(float(value) for value in row[:4])
            track_id = float(row[4])
            confidence = float(row[5])
            class_id = float(row[6])
            if not np.isfinite((*bbox, track_id, confidence, class_id)).all():
                continue
            if int(class_id) not in accepted_class_ids:
                continue
            tracks.append(
                Track(
                    track_id=int(track_id),
                    bbox=bbox,
                    confidence=confidence,
                    class_id=int(class_id),
                )
            )
        return tracks

    def reset(self) -> None:
        self.tracker.reset()
