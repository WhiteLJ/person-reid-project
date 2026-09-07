"""Offline command-line management for the MVP-7 SQLite Gallery."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.config import load_config
from src.database import GalleryRepository, RepositoryError
from src.gallery import format_person_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline SQLite Gallery administration for MVP-7"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="configuration file used to resolve the default database path",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="optional SQLite path override, useful for development and tests",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list persisted Gallery people")
    remove = commands.add_parser("remove", help="remove one persisted Gallery person")
    remove.add_argument("person_id", help="person ID such as P001 or 1")
    commands.add_parser("clear", help="remove all persisted Gallery people")
    return parser


def _parse_person_id(value: str) -> int:
    normalized = value.strip().upper()
    if normalized.startswith("P"):
        normalized = normalized[1:]
    if not normalized.isdigit() or int(normalized) < 1:
        raise ValueError(f"invalid person ID: {value}")
    return int(normalized)


def _database_path(config_path: str, override: Path | None) -> Path:
    if override is not None:
        return override
    return load_config(config_path).database.path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        repository = GalleryRepository(_database_path(args.config, args.db))
        repository.initialize()
        if args.command == "list":
            for person in repository.load_all():
                print(
                    f"{format_person_id(person.person_id)}\t"
                    f"{person.label}\t"
                    f"references={len(person.reference_embeddings)}"
                )
            return 0

        if args.command == "remove":
            person_id = _parse_person_id(args.person_id)
            if not repository.delete_person(person_id):
                print(f"Gallery person not found: {format_person_id(person_id)}", file=sys.stderr)
                return 1
            print(f"Removed {format_person_id(person_id)}")
            return 0

        if args.command == "clear":
            repository.clear()
            print("Gallery cleared")
            return 0
    except (RepositoryError, ValueError) as exc:
        print(f"gallery_admin error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
