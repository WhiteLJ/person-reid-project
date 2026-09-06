"""Configuration loading and runtime device selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass(frozen=True)
class VideoConfig:
    source: int | str


@dataclass(frozen=True)
class ModelConfig:
    yolo_weight: Path
    device: str
    person_class_id: int
    conf_threshold: float
    iou_threshold: float
    image_size: int


@dataclass(frozen=True)
class RuntimeConfig:
    num_workers: int
    log_level: str


@dataclass(frozen=True)
class TrackingConfig:
    tracker: str
    persist: bool
    show_track_id: bool


@dataclass(frozen=True)
class SelectionConfig:
    min_iou: float


@dataclass(frozen=True)
class ReIDConfig:
    model_name: str
    weight: Path
    image_height: int
    image_width: int
    min_crop_width: int
    min_crop_height: int


@dataclass(frozen=True)
class ReIDRecoveryConfig:
    """Runtime controls for in-memory target recovery in MVP-5."""

    lost_grace_frames: int
    reference_update_interval_frames: int
    recovery_interval_frames: int
    max_reference_embeddings: int
    recovery_threshold: float
    recovery_margin: float
    reference_update_threshold: float


@dataclass(frozen=True)
class UIConfig:
    window_name: str
    wait_key_ms: int
    show_class_name: bool
    show_confidence: bool
    show_unselected_tracks: bool = False


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    video: VideoConfig
    model: ModelConfig
    runtime: RuntimeConfig
    tracking: TrackingConfig
    selection: SelectionConfig
    reid: ReIDConfig
    reid_recovery: ReIDRecoveryConfig
    ui: UIConfig


def parse_source(value: Any) -> int | str:
    """Convert a camera index or path-like source to the runtime representation."""

    if isinstance(value, bool):
        raise ValueError("video.source must be a camera index or a path, not a boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("camera index must be non-negative")
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        if not stripped:
            raise ValueError("video.source cannot be empty")
        return stripped
    raise TypeError("video.source must be an integer camera index or string path")


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` and safely fall back when CUDA was requested but unavailable."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if normalized == "cpu" or normalized.startswith("cuda"):
        return normalized
    raise ValueError("model.device must be 'auto', 'cpu', or a CUDA device such as 'cuda:0'")


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"configuration section '{name}' must be a mapping")
    return value


def _resolve_project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name.lower() == "config":
        return resolved.parent.parent
    return resolved.parent


def load_config(config_path: str | Path = "config/config.yaml") -> AppConfig:
    """Load and validate the project YAML configuration."""

    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    project_root = _resolve_project_root(path)
    video = _section(raw, "video")
    model = _section(raw, "model")
    runtime = _section(raw, "runtime")
    tracking = _section(raw, "tracking")
    selection = _section(raw, "selection")
    reid = _section(raw, "reid")
    reid_recovery = _section(raw, "reid_recovery")
    ui = _section(raw, "ui")

    source = parse_source(video.get("source", 0))
    weight_value = model.get("yolo_weight", "weights/yolo/yolov8n.pt")
    weight_path = Path(weight_value)
    if not weight_path.is_absolute():
        weight_path = project_root / weight_path

    requested_workers = int(runtime.get("num_workers", 0))
    if requested_workers < 0:
        raise ValueError("runtime.num_workers must be non-negative")

    min_iou = float(selection.get("min_iou", 0.20))
    if not 0.0 < min_iou <= 1.0:
        raise ValueError("selection.min_iou must be greater than 0 and at most 1")

    reid_weight_value = reid.get(
        "weight", "weights/reid/osnet_x0_25_msmt17.pth"
    )
    reid_weight_path = Path(reid_weight_value)
    if not reid_weight_path.is_absolute():
        reid_weight_path = project_root / reid_weight_path

    reid_model_name = str(reid.get("model_name", "osnet_x0_25")).strip()
    if not reid_model_name:
        raise ValueError("reid.model_name cannot be empty")

    image_height = int(reid.get("image_height", 256))
    image_width = int(reid.get("image_width", 128))
    min_crop_width = int(reid.get("min_crop_width", 40))
    min_crop_height = int(reid.get("min_crop_height", 100))
    if image_height <= 0 or image_width <= 0:
        raise ValueError("reid image dimensions must be positive")
    if min_crop_width <= 0 or min_crop_height <= 0:
        raise ValueError("reid minimum crop dimensions must be positive")

    lost_grace_frames = int(reid_recovery.get("lost_grace_frames", 10))
    reference_update_interval_frames = int(
        reid_recovery.get("reference_update_interval_frames", 15)
    )
    recovery_interval_frames = int(
        reid_recovery.get("recovery_interval_frames", 10)
    )
    max_reference_embeddings = int(
        reid_recovery.get("max_reference_embeddings", 8)
    )
    recovery_threshold = float(
        reid_recovery.get("recovery_threshold", 0.75)
    )
    recovery_margin = float(reid_recovery.get("recovery_margin", 0.05))
    reference_update_threshold = float(
        reid_recovery.get("reference_update_threshold", 0.80)
    )
    if lost_grace_frames < 1:
        raise ValueError("reid_recovery.lost_grace_frames must be positive")
    if reference_update_interval_frames < 1:
        raise ValueError(
            "reid_recovery.reference_update_interval_frames must be positive"
        )
    if recovery_interval_frames < 1:
        raise ValueError("reid_recovery.recovery_interval_frames must be positive")
    if max_reference_embeddings < 1:
        raise ValueError("reid_recovery.max_reference_embeddings must be positive")
    for name, value in (
        ("recovery_threshold", recovery_threshold),
        ("reference_update_threshold", reference_update_threshold),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"reid_recovery.{name} must be greater than 0 and at most 1")
    if recovery_margin < 0.0:
        raise ValueError("reid_recovery.recovery_margin must be non-negative")

    return AppConfig(
        project_root=project_root,
        video=VideoConfig(source=source),
        model=ModelConfig(
            yolo_weight=weight_path,
            device=resolve_device(str(model.get("device", "auto"))),
            person_class_id=int(model.get("person_class_id", 0)),
            conf_threshold=float(model.get("conf_threshold", 0.35)),
            iou_threshold=float(model.get("iou_threshold", 0.50)),
            image_size=int(model.get("image_size", 640)),
        ),
        runtime=RuntimeConfig(
            num_workers=requested_workers,
            log_level=str(runtime.get("log_level", "INFO")).upper(),
        ),
        tracking=TrackingConfig(
            tracker=str(tracking.get("tracker", "botsort.yaml")),
            persist=bool(tracking.get("persist", True)),
            show_track_id=bool(tracking.get("show_track_id", True)),
        ),
        selection=SelectionConfig(min_iou=min_iou),
        reid=ReIDConfig(
            model_name=reid_model_name,
            weight=reid_weight_path,
            image_height=image_height,
            image_width=image_width,
            min_crop_width=min_crop_width,
            min_crop_height=min_crop_height,
        ),
        reid_recovery=ReIDRecoveryConfig(
            lost_grace_frames=lost_grace_frames,
            reference_update_interval_frames=reference_update_interval_frames,
            recovery_interval_frames=recovery_interval_frames,
            max_reference_embeddings=max_reference_embeddings,
            recovery_threshold=recovery_threshold,
            recovery_margin=recovery_margin,
            reference_update_threshold=reference_update_threshold,
        ),
        ui=UIConfig(
            window_name=str(ui.get("window_name", "Person Tracking - MVP-5")),
            wait_key_ms=max(1, int(ui.get("wait_key_ms", 1))),
            show_class_name=bool(ui.get("show_class_name", True)),
            show_confidence=bool(ui.get("show_confidence", True)),
            show_unselected_tracks=bool(ui.get("show_unselected_tracks", False)),
        ),
    )
