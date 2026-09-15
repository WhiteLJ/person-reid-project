"""Torchreid OSNet feature extraction for MVP-4.

This module deliberately has no knowledge of Track, Person ID, TargetGallery,
or SQLite.  It converts OpenCV crops into normalized ReID embeddings only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from logging import getLogger
from math import ceil, floor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import AscendConfig, ReIDConfig, resolve_device


LOGGER = getLogger(__name__)
EMBEDDING_DIM = 512


def normalize_embedding(embedding: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return one or more L2-normalized float32 embeddings."""

    array = np.asarray(embedding, dtype=np.float32)
    if array.ndim not in (1, 2):
        raise ValueError("embedding must have shape (D,) or (N, D)")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("embedding must contain finite, non-empty values")

    axis = 0 if array.ndim == 1 else 1
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("embedding must not contain a zero vector")
    return (array / norms).astype(np.float32, copy=False)


def cosine_similarity(
    embedding_a: np.ndarray | Sequence[float],
    embedding_b: np.ndarray | Sequence[float],
) -> float:
    """Return cosine similarity for two single embeddings."""

    normalized_a = normalize_embedding(embedding_a)
    normalized_b = normalize_embedding(embedding_b)
    if normalized_a.ndim != 1 or normalized_b.ndim != 1:
        raise ValueError("cosine_similarity accepts single embeddings only")
    if normalized_a.shape != normalized_b.shape:
        raise ValueError("embeddings must have the same shape")
    return float(np.dot(normalized_a, normalized_b))


def crop_person(
    frame: np.ndarray,
    bbox: Sequence[float],
    *,
    min_crop_width: int,
    min_crop_height: int,
) -> np.ndarray | None:
    """Clamp an ``xyxy`` person box and return a valid crop or ``None``."""

    if not isinstance(frame, np.ndarray) or frame.ndim < 2:
        raise ValueError("frame must be an image NumPy array")
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four values")
    if min_crop_width <= 0 or min_crop_height <= 0:
        raise ValueError("minimum crop dimensions must be positive")

    coordinates = tuple(float(value) for value in bbox)
    if not np.isfinite(coordinates).all():
        return None

    height, width = frame.shape[:2]
    raw_x1, raw_y1, raw_x2, raw_y2 = coordinates
    x1 = max(0, min(width, floor(raw_x1)))
    y1 = max(0, min(height, floor(raw_y1)))
    x2 = max(0, min(width, ceil(raw_x2)))
    y2 = max(0, min(height, ceil(raw_y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.shape[1] < min_crop_width or crop.shape[0] < min_crop_height:
        return None
    return crop.copy()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("ReID checkpoint does not contain a state_dict mapping")
    return checkpoint


def _without_module_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


class ReIDExtractor:
    """Load one official Torchreid FeatureExtractor and reuse it for inference."""

    def __init__(
        self,
        config: ReIDConfig,
        device: str,
        feature_extractor: Any | None = None,
        *,
        backend: str = "torch",
        ascend_config: AscendConfig | None = None,
        ascend_runtime: Any | None = None,
    ) -> None:
        self.config = config
        self.backend = backend.strip().lower()
        if self.backend not in {"torch", "ascend"}:
            raise ValueError("ReID backend must be 'torch' or 'ascend'")
        self.device = resolve_device(device) if self.backend == "torch" else device
        self._feature_extractor = feature_extractor
        self._ascend_extractor: Any | None = None
        self._ascend_runtime: Any | None = ascend_runtime
        self._owns_ascend_runtime = False

        if self.backend == "ascend":
            if feature_extractor is not None:
                raise ValueError("feature_extractor is only valid for the Torch backend")
            from .ascend_reid import AscendReIDExtractor

            deployment = ascend_config or AscendConfig()
            if ascend_runtime is None:
                from .ascend_runtime import AscendRuntime

                ascend_runtime = AscendRuntime(deployment.device_id)
                self._owns_ascend_runtime = True
            self._ascend_runtime = ascend_runtime
            self.device = f"ascend:{deployment.device_id}"
            self._ascend_extractor = AscendReIDExtractor(
                config,
                deployment.reid_model,
                deployment.reid_dynamic_batches,
                ascend_runtime,
            )
            return

        if self._feature_extractor is not None:
            if not callable(self._feature_extractor):
                raise TypeError("feature_extractor must be callable")
            return

        weight_path = Path(config.weight)
        if not weight_path.is_file():
            raise FileNotFoundError(
                f"ReID checkpoint file not found: {weight_path}. "
                "Place the official MSMT17 checkpoint at this path."
            )

        self._feature_extractor = self._create_official_extractor(weight_path)
        self._audit_loaded_checkpoint(weight_path)

    def _create_official_extractor(self, weight_path: Path) -> Any:
        try:
            import torchreid
            from torchreid.utils import FeatureExtractor
        except ImportError as exc:
            raise RuntimeError(
                "Torchreid 1.4.0 from the official deep-person-reid Git "
                "repository is required for MVP-4"
            ) from exc

        version = getattr(torchreid, "__version__", None)
        if version != "1.4.0":
            raise RuntimeError(
                f"Torchreid 1.4.0 is required, but version {version!r} is installed"
            )

        try:
            extractor = FeatureExtractor(
                model_name=self.config.model_name,
                model_path=str(weight_path),
                image_size=(self.config.image_height, self.config.image_width),
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ReID checkpoint: {weight_path}"
            ) from exc

        model = getattr(extractor, "model", None)
        if model is None or not hasattr(model, "eval"):
            raise RuntimeError("Torchreid FeatureExtractor did not expose a model")
        model.eval()
        return extractor

    def _audit_loaded_checkpoint(self, weight_path: Path) -> None:
        """Reject a random/incorrect backbone while allowing classifier mismatch."""

        try:
            from torchreid.utils import load_checkpoint

            checkpoint = load_checkpoint(str(weight_path))
            checkpoint_state = _checkpoint_state_dict(checkpoint)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to inspect ReID checkpoint: {weight_path}"
            ) from exc

        model = getattr(self._feature_extractor, "model", None)
        if model is None or not hasattr(model, "state_dict"):
            raise RuntimeError("Cannot verify loaded ReID model weights")
        model_state = model.state_dict()

        matched_backbone: set[str] = set()
        discarded_classifier: list[str] = []
        discarded_backbone: list[str] = []
        for raw_key, value in checkpoint_state.items():
            key = _without_module_prefix(str(raw_key))
            if key.startswith("classifier."):
                if key not in model_state or tuple(model_state[key].size()) != tuple(value.size()):
                    discarded_classifier.append(key)
                continue
            if key in model_state and tuple(model_state[key].size()) == tuple(value.size()):
                matched_backbone.add(key)
            else:
                discarded_backbone.append(key)

        expected_backbone = {
            key for key in model_state if not key.startswith("classifier.")
        }
        missing_backbone = sorted(expected_backbone - matched_backbone)
        if not matched_backbone or missing_backbone or discarded_backbone:
            raise RuntimeError(
                "ReID checkpoint backbone weights failed validation: "
                f"matched={len(matched_backbone)}, "
                f"missing={len(missing_backbone)}, "
                f"incompatible={len(discarded_backbone)}"
            )

        LOGGER.info(
            "REID_CHECKPOINT_LOADED path=%s matched_backbone=%d "
            "classifier_layers_discarded=%d (expected num_classes mismatch may be normal)",
            weight_path,
            len(matched_backbone),
            len(discarded_classifier),
        )

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Extract one normalized 512-D embedding from a BGR crop."""

        return self.extract_batch([crop])[0]

    def extract_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Extract normalized embeddings while preserving crop order."""

        if self.backend == "ascend":
            if self._ascend_extractor is None:
                raise RuntimeError("Ascend ReID extractor is not initialized")
            return self._ascend_extractor.extract_batch(crops)

        crop_list = list(crops)
        if not crop_list:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        rgb_crops: list[np.ndarray] = []
        for crop in crop_list:
            if not isinstance(crop, np.ndarray) or crop.ndim != 3:
                raise ValueError("each crop must be a BGR image with shape (H, W, 3)")
            if crop.shape[2] != 3 or crop.shape[0] == 0 or crop.shape[1] == 0:
                raise ValueError("each crop must be a non-empty 3-channel image")
            rgb_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        raw_features = _as_numpy(self._feature_extractor(rgb_crops))
        if raw_features.ndim == 1:
            raw_features = raw_features.reshape(1, -1)
        if raw_features.ndim != 2 or raw_features.shape[0] != len(crop_list):
            raise ValueError(
                "Torchreid output must have shape (N, 512) matching the input batch"
            )
        if raw_features.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Torchreid output dimension must be {EMBEDDING_DIM}, "
                f"got {raw_features.shape[1]}"
            )
        return normalize_embedding(raw_features)

    def drain_inference_diagnostics(
        self,
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[dict[str, float], ...]]:
        """Consume Atlas timing/batch diagnostics without affecting inference.

        The application drains this once per frame so optional benchmark
        telemetry cannot accumulate in memory during a long-running session.
        The Torch backend has no Atlas events and returns empty tuples.
        """

        if self.backend != "ascend" or self._ascend_extractor is None:
            return (), ()
        return (
            self._ascend_extractor.drain_batch_events(),
            self._ascend_extractor.drain_timing_events(),
        )

    def close(self) -> None:
        """Close an Ascend runtime created by this extractor, if any."""

        if self._owns_ascend_runtime and self._ascend_runtime is not None:
            self._ascend_runtime.close()
            self._ascend_runtime = None
            self._owns_ascend_runtime = False
