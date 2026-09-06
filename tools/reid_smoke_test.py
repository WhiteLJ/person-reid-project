"""Validate local OSNet ReID embeddings for three image files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.config import load_config
from src.reid import ReIDExtractor, cosine_similarity
from src.logging_utils import configure_logging


LOGGER = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare normalized OSNet embeddings for A1, A2, and B1 images"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="path to the YAML configuration file",
    )
    parser.add_argument(
        "images",
        nargs=3,
        metavar="IMAGE",
        help="three image paths: same-person A1/A2, comparison-person B1",
    )
    return parser


def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"unable to read image: {Path(path).resolve()}")
    return image


def run(config_path: str, image_paths: Sequence[str]) -> int:
    config = load_config(config_path)
    images = [_read_image(path) for path in image_paths]
    extractor = ReIDExtractor(config.reid, device=config.model.device)
    embeddings = extractor.extract_batch(images)

    labels = ("A1", "A2", "B1")
    for label, embedding in zip(labels, embeddings):
        print(
            f"{label}: shape={embedding.shape} "
            f"dtype={embedding.dtype} norm={np.linalg.norm(embedding):.6f}"
        )

    same_similarity = cosine_similarity(embeddings[0], embeddings[1])
    different_similarity = cosine_similarity(embeddings[0], embeddings[2])
    print(f"cos(A1, A2)={same_similarity:.6f}")
    print(f"cos(A1, B1)={different_similarity:.6f}")
    if same_similarity <= different_similarity:
        LOGGER.warning(
            "same-person similarity is not higher than comparison similarity; "
            "inspect the image crops and camera conditions"
        )
    else:
        LOGGER.info("same-person similarity is higher for this sample")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_argument_parser().parse_args(argv)
    try:
        return run(args.config, args.images)
    except Exception:
        LOGGER.exception("REID_SMOKE_TEST_FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
