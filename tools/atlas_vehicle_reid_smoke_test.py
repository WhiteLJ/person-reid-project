"""Validate Vehicle SBS(R50-IBN) OM inference on an Atlas board."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.ascend_runtime import AscendRuntime
from src.ascend_vehicle_reid import AscendVehicleReIDExtractor
from src.config import load_config


def _read_images(paths: Sequence[Path]) -> tuple[list[np.ndarray], list[str]]:
    images: list[np.ndarray] = []
    names: list[str] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read Vehicle image: {path}")
        images.append(image)
        names.append(path.name)
    return images, names


def _percentiles(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean()),
        float(array.min()),
        float(np.percentile(array, 50)),
        float(np.percentile(array, 95)),
        float(array.max()),
    )


def _load_reference(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as data:
        if "embeddings" not in data or "image_names" not in data:
            raise ValueError("reference NPZ must contain embeddings and image_names")
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        names = [str(value) for value in data["image_names"].tolist()]
    if embeddings.ndim != 2 or embeddings.shape[1] != 2048:
        raise ValueError(f"reference embeddings must have shape (N,2048), got {embeddings.shape}")
    if len(names) != embeddings.shape[0]:
        raise ValueError("reference image_names and embeddings lengths differ")
    return embeddings, names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config_atlas.yaml")
    parser.add_argument("--image", type=Path, nargs="+", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--model", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:
    if any(value < 1 for value in args.batches):
        raise ValueError("--batches values must be positive")
    if args.warmup < 0 or args.repeat < 1:
        raise ValueError("--warmup must be non-negative and --repeat must be positive")

    config = load_config(args.config)
    images, image_names = _read_images(args.image)
    model_path = args.model or config.ascend.vehicle_reid_model
    dynamic_batches = config.ascend.vehicle_reid_dynamic_batches

    runtime = AscendRuntime(config.ascend.device_id)
    try:
        extractor = AscendVehicleReIDExtractor(
            config.vehicle_reid,
            model_path,
            dynamic_batches,
            runtime,
        )
        print(
            "ATLAS_VEHICLE_REID_MODEL_LOADED "
            f"path={extractor.model_path} "
            f"dynamic_batches={','.join(str(value) for value in dynamic_batches)}"
        )
        for batch_size in args.batches:
            batch_images = [images[index % len(images)] for index in range(batch_size)]
            for _ in range(args.warmup):
                extractor.extract_batch(batch_images)
            timings: list[float] = []
            result: np.ndarray | None = None
            for _ in range(args.repeat):
                started = time.perf_counter()
                result = extractor.extract_batch(batch_images)
                timings.append((time.perf_counter() - started) * 1000.0)
            assert result is not None
            if result.shape != (batch_size, 2048):
                raise RuntimeError(f"Vehicle embedding shape mismatch: {result.shape}")
            norms = np.linalg.norm(result, axis=1)
            if not np.isfinite(result).all() or not np.allclose(norms, 1.0, atol=1e-4):
                raise RuntimeError("Vehicle embedding finite/L2 contract failed")
            average, minimum, p50, p95, maximum = _percentiles(timings)
            print(
                "ATLAS_VEHICLE_REID_BATCH "
                f"batch={batch_size} shape={result.shape} "
                f"norm_min={norms.min():.6f} norm_mean={norms.mean():.6f} "
                f"norm_max={norms.max():.6f} "
                f"avg_ms={average:.3f} min_ms={minimum:.3f} "
                f"p50_ms={p50:.3f} p95_ms={p95:.3f} max_ms={maximum:.3f}"
            )

        if args.reference is not None:
            reference_embeddings, reference_names = _load_reference(args.reference)
            if reference_names != image_names:
                raise ValueError(
                    "reference image_names do not match current image order: "
                    f"reference={reference_names} current={image_names}"
                )
            atlas_embeddings = extractor.extract_batch(images)
            cosines = np.sum(reference_embeddings * atlas_embeddings, axis=1)
            for name, cosine in zip(image_names, cosines):
                print(f"ATLAS_VEHICLE_REID_PARITY image={name} cosine={cosine:.6f}")
            print(
                "ATLAS_VEHICLE_REID_PARITY_SUMMARY "
                f"min={cosines.min():.6f} mean={cosines.mean():.6f} "
                f"max={cosines.max():.6f}"
            )

        print("ATLAS_VEHICLE_REID_SMOKE_OK")
        return 0
    finally:
        runtime.close()


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
