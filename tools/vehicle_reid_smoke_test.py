"""PC-only Vehicle ReID smoke test for MVP-8.3-PC1."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config
from src.reid import cosine_similarity
from src.vehicle_reid import VehicleReIDExtractor


def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--image-a", required=True, help="vehicle crop/frame A")
    parser.add_argument("--image-b", required=True, help="same vehicle crop/frame B")
    parser.add_argument("--image-c", required=True, help="different vehicle crop/frame")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    config = load_config(Path(args.config))
    vehicle_config = config.vehicle_reid
    if not vehicle_config.enabled:
        raise RuntimeError("vehicle_reid.enabled is false in the selected config")

    print(f"model={vehicle_config.model_name}")
    print(f"checkpoint={vehicle_config.weight}")
    print(
        f"input_size=({vehicle_config.image_height}, {vehicle_config.image_width})"
    )
    extractor = VehicleReIDExtractor(vehicle_config, device=vehicle_config.device)
    crops = [_read_image(args.image_a), _read_image(args.image_b), _read_image(args.image_c)]
    embeddings = extractor.extract_batch(crops)
    for name, embedding in zip(("A", "B", "C"), embeddings):
        print(
            f"embedding_{name}: shape={embedding.shape} "
            f"dtype={embedding.dtype} norm={np.linalg.norm(embedding):.6f}"
        )

    single_a = extractor.extract(crops[0])
    batch_ab = extractor.extract_batch(crops[:2])
    print(f"cos(A,B)={cosine_similarity(embeddings[0], embeddings[1]):.6f}")
    print(f"cos(A,C)={cosine_similarity(embeddings[0], embeddings[2]):.6f}")
    print(
        "single_vs_batch_A_cos="
        f"{cosine_similarity(single_a, batch_ab[0]):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
