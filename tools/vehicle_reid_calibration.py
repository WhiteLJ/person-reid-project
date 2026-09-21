"""Measure same-vehicle and different-vehicle ReID cosine distributions."""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config
from src.reid import cosine_similarity
from src.vehicle_reid import VehicleReIDExtractor


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _format_stats(values: list[float]) -> str:
    if not values:
        return "no_pairs"
    array = np.asarray(values, dtype=np.float32)
    return (
        f"min={array.min():.6f} p10={np.percentile(array, 10):.6f} "
        f"median={np.median(array):.6f} mean={array.mean():.6f} "
        f"p90={np.percentile(array, 90):.6f} "
        f"p95={np.percentile(array, 95):.6f} max={array.max():.6f}"
    )


def _pairwise_scores(
    embeddings_by_vehicle: dict[str, list[tuple[str, np.ndarray]]],
) -> tuple[list[tuple[float, str, str]], list[tuple[float, str, str]]]:
    positive: list[tuple[float, str, str]] = []
    negative: list[tuple[float, str, str]] = []
    names = sorted(embeddings_by_vehicle)
    for vehicle_name in names:
        samples = embeddings_by_vehicle[vehicle_name]
        for (name_a, embedding_a), (name_b, embedding_b) in itertools.combinations(
            samples, 2
        ):
            positive.append(
                (cosine_similarity(embedding_a, embedding_b), name_a, name_b)
            )
    for index, first_name in enumerate(names):
        for second_name in names[index + 1 :]:
            for name_a, embedding_a in embeddings_by_vehicle[first_name]:
                for name_b, embedding_b in embeddings_by_vehicle[second_name]:
                    negative.append(
                        (
                            cosine_similarity(embedding_a, embedding_b),
                            name_a,
                            name_b,
                        )
                    )
    return positive, negative


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate Vehicle ReID similarity distributions"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--root",
        required=True,
        help="directory containing one subdirectory per vehicle",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(f"calibration directory not found: {root}")

    paths_by_vehicle: dict[str, list[Path]] = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if paths:
            paths_by_vehicle[directory.name] = paths
    if not paths_by_vehicle:
        raise ValueError(f"no vehicle image subdirectories found under {root}")

    extractor = VehicleReIDExtractor(config.vehicle_reid)
    embeddings_by_vehicle: dict[str, list[tuple[str, np.ndarray]]] = {}
    for vehicle_name, paths in paths_by_vehicle.items():
        crops: list[np.ndarray] = []
        valid_paths: list[Path] = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                LOGGER.warning("CALIBRATION_IMAGE_SKIPPED path=%s", path)
                continue
            crops.append(image)
            valid_paths.append(path)
        if not crops:
            continue
        embeddings = extractor.extract_batch(crops)
        embeddings_by_vehicle[vehicle_name] = [
            (str(path), embedding)
            for path, embedding in zip(valid_paths, embeddings)
        ]

    positive, negative = _pairwise_scores(embeddings_by_vehicle)
    print(f"vehicles={len(embeddings_by_vehicle)} samples={sum(len(v) for v in embeddings_by_vehicle.values())}")
    print(f"same-vehicle cosine: {_format_stats([score for score, _, _ in positive])}")
    print(f"different-vehicle cosine: {_format_stats([score for score, _, _ in negative])}")
    if positive:
        hardest_positive = min(positive, key=lambda item: item[0])
        print(
            "hardest_positive="
            f"{hardest_positive[0]:.6f} "
            f"a={hardest_positive[1]} b={hardest_positive[2]}"
        )
    else:
        print("hardest_positive=no_pair")
    if negative:
        hardest_negative = max(negative, key=lambda item: item[0])
        print(
            "hardest_negative="
            f"{hardest_negative[0]:.6f} "
            f"a={hardest_negative[1]} b={hardest_negative[2]}"
        )
    else:
        print("hardest_negative=no_pair")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main())
