"""Ascend OM OSNet preprocessing and dynamic-batch feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .ascend_runtime import AscendModel, AscendRuntime
from .config import ReIDConfig


EMBEDDING_DIM = 512
_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def preprocess_reid_crop(
    crop: np.ndarray,
    *,
    image_height: int = 256,
    image_width: int = 128,
) -> np.ndarray:
    """Convert one OpenCV BGR crop to a normalized NCHW-ready image."""

    if not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError("crop must have shape (H, W, 3) in BGR order")
    if crop.shape[0] < 1 or crop.shape[1] < 1:
        raise ValueError("crop must not be empty")
    if image_height < 1 or image_width < 1:
        raise ValueError("ReID image dimensions must be positive")

    resized = cv2.resize(crop, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)


def dynamic_batch_chunks(
    count: int,
    supported_batches: Sequence[int] = (1, 2, 4, 8),
) -> tuple[int, ...]:
    """Split a request into supported dynamic batches without fake samples."""

    if count < 0:
        raise ValueError("count must be non-negative")
    batches = tuple(sorted({int(value) for value in supported_batches}))
    if count == 0:
        return ()
    if not batches or batches[0] != 1 or any(value < 1 for value in batches):
        raise ValueError("supported_batches must contain positive batch size 1")

    chunks: list[int] = []
    remaining = count
    while remaining:
        choices = [value for value in batches if value <= remaining]
        if choices:
            chosen = max(choices)
        else:
            # This branch is unreachable when batch size 1 is supported.
            raise ValueError(f"cannot represent remaining batch size {remaining}")
        chunks.append(chosen)
        remaining -= chosen
    return tuple(chunks)


class AscendReIDExtractor:
    """Run one OSNet OM and preserve the project's ReIDExtractor contract."""

    def __init__(
        self,
        config: ReIDConfig,
        model_path: str | Path,
        dynamic_batch_sizes: Sequence[int],
        runtime: AscendRuntime,
        *,
        model: AscendModel | Any | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.model_path = Path(model_path)
        self.dynamic_batch_sizes = tuple(sorted({int(value) for value in dynamic_batch_sizes}))
        if not self.dynamic_batch_sizes or self.dynamic_batch_sizes[0] != 1:
            raise ValueError("dynamic_batch_sizes must include 1")
        self.model = (
            model
            if model is not None
            else runtime.load_model(
                self.model_path,
                dynamic_batch_sizes=self.dynamic_batch_sizes,
            )
        )
        self.inference_count = 0

    def extract(self, crop: np.ndarray) -> np.ndarray:
        return self.extract_batch([crop])[0]

    def extract_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        crop_list = list(crops)
        if not crop_list:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        inputs = np.stack(
            [
                preprocess_reid_crop(
                    crop,
                    image_height=self.config.image_height,
                    image_width=self.config.image_width,
                )
                for crop in crop_list
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        outputs: list[np.ndarray] = []
        offset = 0
        for chunk_size in dynamic_batch_chunks(
            len(crop_list), self.dynamic_batch_sizes
        ):
            chunk = np.ascontiguousarray(inputs[offset : offset + chunk_size])
            raw_outputs = self.runtime.execute(
                self.model,
                [chunk],
                dynamic_batch=chunk_size,
            )
            if not raw_outputs:
                raise ValueError("OSNet OM returned no outputs")
            output = np.asarray(raw_outputs[0], dtype=np.float32)
            output = output.reshape(-1, EMBEDDING_DIM)
            if len(output) < chunk_size:
                raise ValueError(
                    f"OSNet output contains {len(output)} embeddings for batch {chunk_size}"
                )
            outputs.append(output[:chunk_size].copy())
            offset += chunk_size
            self.inference_count += 1

        result = np.concatenate(outputs, axis=0)
        if result.shape != (len(crop_list), EMBEDDING_DIM):
            raise ValueError(f"OSNet output shape is invalid: {result.shape}")
        from .reid import normalize_embedding

        return normalize_embedding(result)

