"""Offline administration tool for the MVP-8.3-PC5 Vehicle Gallery.

Close the running application before ``remove`` or ``clear`` so this offline
tool cannot diverge from an in-memory VehicleTargetGallery in another process.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.config import load_config
from src.vehicle_database import VehicleGalleryRepository, VehicleRepositoryError
from src.vehicle_gallery import format_vehicle_id, parse_vehicle_id


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline SQLite administration for the Vehicle Gallery"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="configuration file used to resolve vehicle_database.path",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="optional SQLite path override, useful for development and tests",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list persisted Vehicle identities")
    remove = commands.add_parser("remove", help="remove one Vehicle identity")
    remove.add_argument("vehicle_id", help="Vehicle ID such as V001 or 1")
    commands.add_parser("clear", help="remove all persisted Vehicle identities")
    return parser


def _database_path(config_path: str, override: Path | None) -> Path:
    if override is not None:
        return override
    return load_config(config_path).vehicle_database.path


def _parse_vehicle_id(value: str) -> int:
    return parse_vehicle_id(value)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        repository = VehicleGalleryRepository(_database_path(args.config, args.db))
        repository.initialize()

        if args.command == "list":
            for vehicle in repository.load_all():
                print(
                    f"{format_vehicle_id(vehicle.vehicle_id)}\t"
                    f"{vehicle.label}\t"
                    f"references={len(vehicle.reference_embeddings)}"
                )
            return 0

        if args.command == "remove":
            vehicle_id = _parse_vehicle_id(args.vehicle_id)
            if not repository.delete_vehicle(vehicle_id):
                print(
                    f"Vehicle not found: {format_vehicle_id(vehicle_id)}",
                    file=sys.stderr,
                )
                return 1
            print(f"Removed {format_vehicle_id(vehicle_id)}")
            return 0

        if args.command == "clear":
            repository.clear()
            print("Vehicle Gallery cleared")
            return 0
    except (VehicleRepositoryError, ValueError) as exc:
        print(f"vehicle_gallery_admin error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
