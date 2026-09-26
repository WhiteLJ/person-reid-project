"""Torch-free Vehicle ReID preprocessing and feature normalization helpers."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


VEHICLE_EMBEDDING_DIM = 2048
_IMAGENET_MEAN_255 = np.asarray(
    (0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0),
    dtype=np.float32,
).reshape(1, 3, 1, 1)
_IMAGENET_STD_255 = np.asarray(
    (0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0),
    dtype=np.float32,
).reshape(1, 3, 1, 1)


def preprocess_vehicle_crop(
    crop: np.ndarray,
    *,
    image_height: int = 256,
    image_width: int = 256,
) -> np.ndarray:
    """Convert one OpenCV BGR crop to a normalized contiguous CHW tensor."""

    if not isinstance(crop, np.ndarray) or crop.ndim != 3:
        raise ValueError("vehicle crop must have shape (H, W, 3) in BGR order")
    if crop.shape[2] != 3 or crop.shape[0] <= 0 or crop.shape[1] <= 0:
        raise ValueError("vehicle crop must be a non-empty 3-channel image")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("vehicle image dimensions must be positive")
    try:
        finite = np.isfinite(crop).all()
    except TypeError as exc:
        raise ValueError("vehicle crop must contain numeric values") from exc
    if not finite:
        raise ValueError("vehicle crop contains non-finite values")

    # Keep the same ordering as the PC extractor: BGR -> RGB, then cubic resize.
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        (image_width, image_height),
        interpolation=cv2.INTER_CUBIC,
    )
    chw = np.ascontiguousarray(resized.astype(np.float32).transpose(2, 0, 1))
    normalized = (chw[None, ...] - _IMAGENET_MEAN_255) / _IMAGENET_STD_255
    return np.ascontiguousarray(normalized[0], dtype=np.float32)


def prepare_vehicle_batch(
    crops: Sequence[np.ndarray],
    *,
    image_height: int = 256,
    image_width: int = 256,
) -> np.ndarray:
    """Prepare an ordered BGR crop sequence as contiguous NCHW float32 data."""

    crop_list = list(crops)
    if not crop_list:
        return np.empty(
            (0, 3, image_height, image_width),
            dtype=np.float32,
        )
    return np.ascontiguousarray(
        np.stack(
            [
                preprocess_vehicle_crop(
                    crop,
                    image_height=image_height,
                    image_width=image_width,
                )
                for crop in crop_list
            ],
            axis=0,
        ),
        dtype=np.float32,
    )


def normalize_vehicle_features(
    features: np.ndarray,
    *,
    expected_dim: int | None = None,
) -> np.ndarray:
    """Validate and L2-normalize Vehicle features as float32."""

    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("Vehicle features must have shape (N, D)")
    if expected_dim is not None and array.shape[1] != expected_dim:
        raise ValueError(
            f"Vehicle features must have dimension {expected_dim}, "
            f"found {array.shape[1]}"
        )
    if array.shape[0] == 0:
        dimension = expected_dim if expected_dim is not None else array.shape[1]
        return np.empty((0, dimension), dtype=np.float32)
    if array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError("Vehicle features must be finite and non-empty")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("Vehicle features must not contain a zero vector")
    return np.ascontiguousarray(array / norms, dtype=np.float32)
