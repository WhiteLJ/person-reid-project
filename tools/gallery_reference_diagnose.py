"""Print persistent Gallery reference-bank diversity diagnostics.

This is an offline diagnostic only.  It reads SQLite and performs no ReID
inference or database mutation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import load_config
from src.database import GalleryRepository
from src.gallery import format_person_id
from src.gallery_reference_bank import pairwise_cosine_matrix
from src.reid import cosine_similarity


def _person_id(value: str) -> int:
    text = value.strip().upper()
    if text.startswith("P"):
        text = text[1:]
    try:
        person_id = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "person must be an integer or formatted like P001"
        ) from exc
    if person_id < 1:
        raise argparse.ArgumentTypeError("person ID must be positive")
    return person_id


def _print_person(person) -> None:
    references = person.reference_embeddings
    matrix = pairwise_cosine_matrix(references)
    centroid_similarities = np.asarray(
        [cosine_similarity(person.centroid, reference) for reference in references],
        dtype=np.float32,
    )
    print(
        f"{format_person_id(person.person_id)} label={person.label!r} "
        f"reference_count={len(references)} "
        f"centroid_norm={np.linalg.norm(person.centroid):.6f}"
    )
    print("centroid_to_reference_cosine=")
    print(np.array2string(centroid_similarities, precision=6, suppress_small=True))
    print("reference_pairwise_cosine=")
    print(np.array2string(matrix, precision=6, suppress_small=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect persistent Gallery reference diversity"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="project YAML configuration (default: config/config.yaml)",
    )
    parser.add_argument(
        "--person",
        type=_person_id,
        help="only inspect one person, for example P001",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    repository = GalleryRepository(config.database.path)
    people = repository.load_all()
    if args.person is not None:
        people = tuple(
            person for person in people if person.person_id == args.person
        )
        if not people:
            raise SystemExit(f"Gallery person not found: {format_person_id(args.person)}")
    for index, person in enumerate(people):
        if index:
            print()
        _print_person(person)
    if not people:
        print("Gallery is empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
