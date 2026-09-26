"""Create a PC Vehicle ReID reference NPZ for Atlas parity checks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.config import load_config
from src.vehicle_reid import VehicleReIDExtractor


def _read_images(image_paths: Sequence[str]) -> tuple[list[np.ndarray], list[str]]:
    images: list[np.ndarray] = []
    names: list[str] = []
    for value in image_paths:
        path = Path(value)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read Vehicle image: {path}")
        images.append(image)
        names.append(path.name)
    return images, names


def create_reference(
    config_path: str | Path,
    image_paths: Sequence[str],
    output_path: str | Path,
) -> Path:
    if not image_paths:
        raise ValueError("at least one --image is required")
    config = load_config(config_path)
    images, names = _read_images(image_paths)
    extractor = VehicleReIDExtractor(config.vehicle_reid, device="cpu")
    embeddings = extractor.extract_batch(images)
    if embeddings.ndim != 2 or embeddings.shape[1] != 2048:
        raise ValueError(f"Vehicle reference embedding shape is invalid: {embeddings.shape}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        embeddings=np.ascontiguousarray(embeddings, dtype=np.float32),
        image_names=np.asarray(names),
        image_shapes=np.asarray([image.shape for image in images], dtype=np.int32),
    )
    print(
        f"PC_VEHICLE_ATLAS_REFERENCE path={destination} "
        f"images={len(images)} shape={embeddings.shape}"
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--image", action="append", required=True, help="Vehicle crop; repeat for multiple images")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    create_reference(args.config, args.image, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
