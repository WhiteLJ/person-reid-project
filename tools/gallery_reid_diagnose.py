"""Read-only Gallery feature and optional candidate-crop diagnostic tool."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.config import load_config
from src.database import GalleryRepository
from src.gallery import TargetGallery, format_person_id
from src.gallery_diagnostics import (
    person_feature_diagnostics,
    rank_candidate_against_gallery,
)
from src.gallery_service import GalleryPersistenceService
from src.reid import ReIDExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Gallery reference-bank and candidate diagnostic"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--candidate-crop",
        nargs="+",
        type=Path,
        help="optional BGR image crops to rank against Gallery centroids",
    )
    parser.add_argument(
        "--track-ids",
        nargs="+",
        type=int,
        help="optional labels matching --candidate-crop order",
    )
    return parser


def _print_value(value: float | None) -> str:
    return "none" if value is None else f"{value:.6f}"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.track_ids is not None and args.candidate_crop is None:
        raise ValueError("--track-ids requires --candidate-crop")
    if args.track_ids is not None and len(args.track_ids) != len(args.candidate_crop):
        raise ValueError("--track-ids must have one value per candidate crop")

    config = load_config(args.config)
    repository = GalleryRepository(config.database.path)
    gallery = TargetGallery()
    GalleryPersistenceService(gallery, repository).load()
    print(f"database={repository.path} people={len(gallery.all_people())}")
    for person in gallery.all_people():
        diagnostics = person_feature_diagnostics(
            person,
            duplicate_threshold=config.gallery_enrichment.reference_duplicate_threshold,
        )
        pairwise = diagnostics["reference_pairwise"]
        centroid_stats = diagnostics["reference_to_centroid"]
        assert isinstance(pairwise, dict)
        assert isinstance(centroid_stats, dict)
        print(
            f"{format_person_id(person.person_id)} label={person.label!r} "
            f"references={diagnostics['reference_count']} "
            f"centroid_norm={diagnostics['centroid_norm']:.6f}"
        )
        print(
            "  pairwise_cosine="
            f"min={_print_value(pairwise['min'])} "
            f"mean={_print_value(pairwise['mean'])} "
            f"max={_print_value(pairwise['max'])} "
            f"duplicate_pairs={pairwise['duplicate_pair_count']}"
        )
        print(
            "  reference_to_centroid="
            f"min={_print_value(centroid_stats['min'])} "
            f"mean={_print_value(centroid_stats['mean'])} "
            f"max={_print_value(centroid_stats['max'])}"
        )

    if not args.candidate_crop:
        return 0
    import cv2

    crops = []
    for path in args.candidate_crop:
        crop = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if crop is None:
            raise FileNotFoundError(f"cannot read candidate crop: {path}")
        crops.append(crop)

    runtime = None
    extractor = None
    try:
        if config.inference.backend == "ascend":
            from src.ascend_runtime import AscendRuntime

            runtime = AscendRuntime(config.ascend.device_id)
        extractor = ReIDExtractor(
            config.reid,
            config.model.device,
            backend=config.inference.backend,
            ascend_config=config.ascend,
            ascend_runtime=runtime,
        )
        embeddings = extractor.extract_batch(crops)
        people = gallery.all_people()
        for index, (path, embedding) in enumerate(zip(args.candidate_crop, embeddings)):
            ranking = rank_candidate_against_gallery(people, embedding)
            track_id = (
                args.track_ids[index]
                if args.track_ids is not None
                else None
            )
            prefix = f"candidate={path}"
            if track_id is not None:
                prefix += f" track={track_id}"
            if not ranking:
                print(prefix + " gallery_ranking=empty")
                continue
            top1_id, top1_score = ranking[0]
            top2_id, top2_score = ranking[1] if len(ranking) > 1 else (None, None)
            margin = top1_score - top2_score if top2_score is not None else None
            print(
                f"{prefix} top1={format_person_id(top1_id)} "
                f"score={top1_score:.6f} "
                f"top2={format_person_id(top2_id) if top2_id is not None else 'none'} "
                f"top2_score={_print_value(top2_score)} "
                f"margin={_print_value(margin)}"
            )
    finally:
        if extractor is not None:
            extractor.close()
        if runtime is not None:
            runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
