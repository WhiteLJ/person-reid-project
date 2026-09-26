"""Atlas Vehicle ReID OM extractor without a PyTorch dependency."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .ascend_reid import dynamic_batch_chunks
from .ascend_runtime import AscendModel, AscendRuntime, AscendRuntimeError
from .config import VehicleReIDConfig
from .vehicle_reid_common import (
    VEHICLE_EMBEDDING_DIM,
    normalize_vehicle_features,
    prepare_vehicle_batch,
)


class AscendVehicleReIDExtractor:
    """Run the Vehicle SBS(R50-IBN) OM through a shared AscendRuntime."""

    def __init__(
        self,
        config: VehicleReIDConfig,
        model_path: str | Path,
        dynamic_batch_sizes: Sequence[int],
        runtime: AscendRuntime,
        *,
        model: AscendModel | Any | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.model_path = Path(model_path).expanduser().resolve()
        self.dynamic_batch_sizes = tuple(
            sorted({int(value) for value in dynamic_batch_sizes})
        )
        if not self.dynamic_batch_sizes or self.dynamic_batch_sizes[0] != 1:
            raise ValueError("dynamic_batch_sizes must include batch size 1")
        if (config.image_height, config.image_width) != (256, 256):
            raise ValueError(
                "Atlas Vehicle OM expects Vehicle input size (256, 256), "
                f"got ({config.image_height}, {config.image_width})"
            )
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
            return np.empty((0, VEHICLE_EMBEDDING_DIM), dtype=np.float32)

        batch = prepare_vehicle_batch(
            crop_list,
            image_height=self.config.image_height,
            image_width=self.config.image_width,
        )
        chunks = dynamic_batch_chunks(
            len(crop_list),
            self.dynamic_batch_sizes,
        )
        outputs: list[np.ndarray] = []
        offset = 0
        for chunk_size in chunks:
            chunk = np.ascontiguousarray(batch[offset : offset + chunk_size])
            context = (
                f"model_path={self.model_path} batch_size={chunk_size} "
                f"input_shape={chunk.shape} input_dtype={chunk.dtype}"
            )
            try:
                raw_outputs = self.runtime.execute(
                    self.model,
                    [chunk],
                    dynamic_batch=chunk_size,
                )
            except Exception as exc:
                raise AscendRuntimeError(
                    "Vehicle OM inference failed: "
                    f"{context}; cause={exc}"
                ) from exc

            if not raw_outputs:
                raise ValueError(
                    f"Vehicle OM returned no output: {context}"
                )
            output = np.asarray(raw_outputs[0], dtype=np.float32)
            if output.size % VEHICLE_EMBEDDING_DIM != 0:
                raise ValueError(
                    "Vehicle OM output size is not divisible by 2048: "
                    f"{context} output_shape={output.shape} "
                    f"output_size={output.size}"
                )
            reshaped = output.reshape(-1, VEHICLE_EMBEDDING_DIM)
            if reshaped.shape[0] < chunk_size:
                raise ValueError(
                    "Vehicle OM output contains fewer rows than requested: "
                    f"{context} output_shape={output.shape}"
                )
            if not np.isfinite(reshaped[:chunk_size]).all():
                raise ValueError(
                    f"Vehicle OM output contains NaN/Inf: {context}"
                )
            outputs.append(reshaped[:chunk_size].copy())
            offset += chunk_size
            self.inference_count += 1

        raw_features = np.concatenate(outputs, axis=0)
        if raw_features.shape != (len(crop_list), VEHICLE_EMBEDDING_DIM):
            raise ValueError(
                "Vehicle OM output shape is invalid: "
                f"model_path={self.model_path} expected="
                f"({len(crop_list)}, {VEHICLE_EMBEDDING_DIM}) "
                f"actual={raw_features.shape}"
            )
        try:
            return normalize_vehicle_features(
                raw_features,
                expected_dim=VEHICLE_EMBEDDING_DIM,
            )
        except ValueError as exc:
            raise ValueError(
                "Vehicle OM feature normalization failed: "
                f"model_path={self.model_path} input_shape={batch.shape} "
                f"input_dtype={batch.dtype}; cause={exc}"
            ) from exc
