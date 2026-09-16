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
class InferenceConfig:
    """Select the neural-network inference backend."""

    backend: str = "torch"


@dataclass(frozen=True)
class AscendConfig:
    """AscendCL model and dynamic-batch settings."""

    device_id: int = 0
    yolo_model: Path = Path("weights/atlas/yolov8n.om")
    reid_model: Path = Path("weights/atlas/osnet_x0_25.om")
    reid_dynamic_batches: tuple[int, ...] = (1, 2, 4, 8)


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
    recovery_min_track_age_frames: int = 3
    recovery_confirmation_hits: int = 2
    recovery_pending_max_age_frames: int = 60


@dataclass(frozen=True)
class GalleryEnrichmentConfig:
    """Runtime policy for explicitly enrolled Gallery feature enrichment."""

    post_recovery_stable_frames: int = 30


@dataclass(frozen=True)
class ReIDQualityConfig:
    """Quality gates for ReID decisions in crowded scenes.

    These are engineering starting values, not universal operating points.
    """

    min_track_confidence: float = 0.35
    max_edge_truncation_ratio: float = 0.30
    max_person_overlap_ratio: float = 0.50


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Lightweight runtime statistics controls."""

    enabled: bool = True
    log_interval_frames: int = 300


@dataclass(frozen=True)
class GalleryRecognitionConfig:
    """Runtime controls for automatic recognition of persisted Gallery people."""

    enabled: bool
    recognition_interval_frames: int
    min_track_age_frames: int
    recognition_threshold: float
    recognition_margin: float
    confirmation_hits: int


@dataclass(frozen=True)
class DatabaseConfig:
    """SQLite persistence settings."""

    path: Path


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
    inference: InferenceConfig
    ascend: AscendConfig
    runtime: RuntimeConfig
    tracking: TrackingConfig
    selection: SelectionConfig
    reid: ReIDConfig
    reid_recovery: ReIDRecoveryConfig
    gallery_enrichment: GalleryEnrichmentConfig
    reid_quality: ReIDQualityConfig
    gallery_recognition: GalleryRecognitionConfig
    database: DatabaseConfig
    ui: UIConfig
    diagnostics: DiagnosticsConfig


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
    inference = _section(raw, "inference")
    ascend = _section(raw, "ascend")
    runtime = _section(raw, "runtime")
    tracking = _section(raw, "tracking")
    selection = _section(raw, "selection")
    reid = _section(raw, "reid")
    reid_recovery = _section(raw, "reid_recovery")
    gallery_enrichment = _section(raw, "gallery_enrichment")
    reid_quality = _section(raw, "reid_quality")
    gallery_recognition = _section(raw, "gallery_recognition")
    database = _section(raw, "database")
    ui = _section(raw, "ui")
    diagnostics = _section(raw, "diagnostics")

    source = parse_source(video.get("source", 0))
    inference_backend = str(inference.get("backend", "torch")).strip().lower()
    if inference_backend not in {"torch", "ascend"}:
        raise ValueError("inference.backend must be 'torch' or 'ascend'")

    ascend_device_id = int(ascend.get("device_id", 0))
    if ascend_device_id < 0:
        raise ValueError("ascend.device_id must be non-negative")

    def resolve_project_path(value: Any, default: str) -> Path:
        resolved = Path(value if value is not None else default)
        return resolved if resolved.is_absolute() else project_root / resolved

    ascend_yolo_model = resolve_project_path(
        ascend.get("yolo_model"), "weights/atlas/yolov8n.om"
    )
    ascend_reid_model = resolve_project_path(
        ascend.get("reid_model"), "weights/atlas/osnet_x0_25.om"
    )
    dynamic_batch_values = ascend.get("reid_dynamic_batches", [1, 2, 4, 8])
    if not isinstance(dynamic_batch_values, (list, tuple)):
        raise ValueError("ascend.reid_dynamic_batches must be a list")
    reid_dynamic_batches = tuple(sorted({int(value) for value in dynamic_batch_values}))
    if not reid_dynamic_batches or reid_dynamic_batches[0] != 1:
        raise ValueError("ascend.reid_dynamic_batches must include batch size 1")
    if any(value < 1 for value in reid_dynamic_batches):
        raise ValueError("ascend.reid_dynamic_batches must be positive")

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
    recovery_min_track_age_frames = int(
        reid_recovery.get("recovery_min_track_age_frames", 3)
    )
    recovery_confirmation_hits = int(
        reid_recovery.get("recovery_confirmation_hits", 2)
    )
    recovery_pending_max_age_frames = int(
        reid_recovery.get("recovery_pending_max_age_frames", 60)
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
    if recovery_min_track_age_frames < 1:
        raise ValueError(
            "reid_recovery.recovery_min_track_age_frames must be positive"
        )
    if recovery_confirmation_hits < 1:
        raise ValueError(
            "reid_recovery.recovery_confirmation_hits must be positive"
        )
    if recovery_pending_max_age_frames < 1:
        raise ValueError(
            "reid_recovery.recovery_pending_max_age_frames must be positive"
        )

    post_recovery_stable_frames = int(
        gallery_enrichment.get("post_recovery_stable_frames", 30)
    )
    if post_recovery_stable_frames < 0:
        raise ValueError(
            "gallery_enrichment.post_recovery_stable_frames must be non-negative"
        )

    min_track_confidence = float(
        reid_quality.get("min_track_confidence", 0.35)
    )
    max_edge_truncation_ratio = float(
        reid_quality.get("max_edge_truncation_ratio", 0.30)
    )
    max_person_overlap_ratio = float(
        reid_quality.get("max_person_overlap_ratio", 0.50)
    )
    if not 0.0 <= min_track_confidence <= 1.0:
        raise ValueError("reid_quality.min_track_confidence must be in [0, 1]")
    for name, value in (
        ("max_edge_truncation_ratio", max_edge_truncation_ratio),
        ("max_person_overlap_ratio", max_person_overlap_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"reid_quality.{name} must be in [0, 1]")

    recognition_interval_frames = int(
        gallery_recognition.get("recognition_interval_frames", 10)
    )
    min_track_age_frames = int(
        gallery_recognition.get("min_track_age_frames", 5)
    )
    recognition_threshold = float(
        gallery_recognition.get("recognition_threshold", 0.80)
    )
    recognition_margin = float(
        gallery_recognition.get("recognition_margin", 0.05)
    )
    confirmation_hits = int(gallery_recognition.get("confirmation_hits", 2))
    if recognition_interval_frames < 1:
        raise ValueError(
            "gallery_recognition.recognition_interval_frames must be positive"
        )
    if min_track_age_frames < 1:
        raise ValueError(
            "gallery_recognition.min_track_age_frames must be positive"
        )
    if not 0.0 < recognition_threshold <= 1.0:
        raise ValueError(
            "gallery_recognition.recognition_threshold must be greater than 0 and at most 1"
        )
    if recognition_margin < 0.0:
        raise ValueError(
            "gallery_recognition.recognition_margin must be non-negative"
        )
    if confirmation_hits < 1:
        raise ValueError("gallery_recognition.confirmation_hits must be positive")

    database_value = database.get("path", "database/person_reid.db")
    database_path = Path(database_value)
    if not str(database_path).strip():
        raise ValueError("database.path cannot be empty")
    if not database_path.is_absolute():
        database_path = project_root / database_path

    tracker_value = str(tracking.get("tracker", "botsort.yaml"))
    tracker_path = Path(tracker_value)
    if not tracker_path.is_absolute():
        project_tracker_path = project_root / tracker_path
        if project_tracker_path.is_file():
            tracker_value = str(project_tracker_path)

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
        inference=InferenceConfig(backend=inference_backend),
        ascend=AscendConfig(
            device_id=ascend_device_id,
            yolo_model=ascend_yolo_model,
            reid_model=ascend_reid_model,
            reid_dynamic_batches=reid_dynamic_batches,
        ),
        runtime=RuntimeConfig(
            num_workers=requested_workers,
            log_level=str(runtime.get("log_level", "INFO")).upper(),
        ),
        tracking=TrackingConfig(
            tracker=tracker_value,
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
            recovery_min_track_age_frames=recovery_min_track_age_frames,
            recovery_confirmation_hits=recovery_confirmation_hits,
            recovery_pending_max_age_frames=recovery_pending_max_age_frames,
        ),
        gallery_enrichment=GalleryEnrichmentConfig(
            post_recovery_stable_frames=post_recovery_stable_frames,
        ),
        reid_quality=ReIDQualityConfig(
            min_track_confidence=min_track_confidence,
            max_edge_truncation_ratio=max_edge_truncation_ratio,
            max_person_overlap_ratio=max_person_overlap_ratio,
        ),
        gallery_recognition=GalleryRecognitionConfig(
            enabled=bool(gallery_recognition.get("enabled", True)),
            recognition_interval_frames=recognition_interval_frames,
            min_track_age_frames=min_track_age_frames,
            recognition_threshold=recognition_threshold,
            recognition_margin=recognition_margin,
            confirmation_hits=confirmation_hits,
        ),
        database=DatabaseConfig(path=database_path),
        ui=UIConfig(
            window_name=str(ui.get("window_name", "Person Tracking - MVP-8.2")),
            wait_key_ms=max(1, int(ui.get("wait_key_ms", 1))),
            show_class_name=bool(ui.get("show_class_name", True)),
            show_confidence=bool(ui.get("show_confidence", True)),
            show_unselected_tracks=bool(ui.get("show_unselected_tracks", False)),
        ),
        diagnostics=DiagnosticsConfig(
            enabled=bool(diagnostics.get("enabled", True)),
            log_interval_frames=max(
                1, int(diagnostics.get("log_interval_frames", 300))
            ),
        ),
    )
