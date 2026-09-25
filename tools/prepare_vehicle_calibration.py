"""Prepare a Vehicle ReID calibration subset from AIC21 Track2 labels.

The source dataset is never modified.  Images are copied into the directory
layout consumed by ``tools.vehicle_reid_calibration``::

    vehicle_<vehicle-id>/0001.jpg
    vehicle_<vehicle-id>/0002.jpg

Vehicle identities and camera IDs come from ``train_label.xml``; image names
are never parsed to infer identity.
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ImageValidator = Callable[[Path], bool]


@dataclass(frozen=True)
class TrainingImage:
    """One XML-labelled training image."""

    image_name: str
    vehicle_id: str
    camera_id: str | None
    source_path: Path


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _safe_source_path(image_dir: Path, image_name: str) -> Path:
    """Resolve an XML image name while preventing paths outside image_train."""

    relative = Path(image_name)
    if relative.is_absolute() or relative.drive or relative.anchor:
        raise ValueError(f"XML imageName must be relative: {image_name!r}")
    if ".." in relative.parts:
        raise ValueError(
            f"XML imageName points outside image_train: {image_name!r}"
        )
    return image_dir / relative


def parse_training_labels(xml_path: Path, image_dir: Path) -> dict[str, list[TrainingImage]]:
    """Parse AIC21 ``Item`` records grouped by XML ``vehicleID``.

    The AIC21 file currently has the following structure::

        <TrainingImages>
          <Item imageName="000001.jpg" vehicleID="0269" cameraID="c026" />
        </TrainingImages>

    ``vehicleID`` is the only source of identity.  Missing required XML
    attributes fail loudly rather than falling back to filename heuristics.
    """

    xml_path = Path(xml_path)
    image_dir = Path(image_dir)
    if not xml_path.is_file():
        raise FileNotFoundError(f"training label XML not found: {xml_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"training image directory not found: {image_dir}")

    try:
        # Python's ElementTree on Windows may reject the AIC21 XML declaration
        # with ``ValueError: multi-byte encodings are not supported`` when it
        # parses the file stream directly.  Decode the declared dataset format
        # first, then parse Unicode text.
        raw_xml = xml_path.read_bytes()
        xml_text: str | None = None
        for encoding in ("gb2312", "utf-8-sig", "utf-8"):
            try:
                xml_text = raw_xml.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if xml_text is None:
            raise ValueError("unsupported XML encoding")
        root = ET.fromstring(xml_text)
    except (ET.ParseError, UnicodeError, ValueError) as exc:
        raise ValueError(f"could not parse training label XML: {xml_path}") from exc

    grouped: dict[str, list[TrainingImage]] = defaultdict(list)
    item_count = 0
    for element in root.iter():
        if _local_name(element.tag) != "Item":
            continue
        item_count += 1
        image_name = element.attrib.get("imageName")
        vehicle_id = element.attrib.get("vehicleID")
        if not image_name or not vehicle_id:
            raise ValueError(
                "each XML Item must contain imageName and vehicleID attributes"
            )
        source_path = _safe_source_path(image_dir, image_name)
        camera_id = element.attrib.get("cameraID") or None
        record = TrainingImage(
            image_name=image_name,
            vehicle_id=vehicle_id,
            camera_id=camera_id,
            source_path=source_path,
        )
        grouped[vehicle_id].append(record)

    if item_count == 0:
        raise ValueError(f"no Item records found in training label XML: {xml_path}")
    return dict(grouped)


def default_image_validator(path: Path) -> bool:
    """Return whether a source image exists and has a supported image suffix.

    This intentionally does not decode every image.  The source dataset has
    already been labelled as image files, and avoiding a second full-dataset
    decode keeps preparation fast.  The calibration tool performs its normal
    OpenCV read and reports any unreadable selected image.
    """

    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def valid_records(
    records_by_vehicle: dict[str, list[TrainingImage]],
    validator: ImageValidator = default_image_validator,
) -> dict[str, list[TrainingImage]]:
    """Filter records whose source file is present and image-like."""

    return {
        vehicle_id: [record for record in records if validator(record.source_path)]
        for vehicle_id, records in records_by_vehicle.items()
        if any(validator(record.source_path) for record in records)
    }


def _sample_across_cameras(
    records: Iterable[TrainingImage],
    count: int,
    rng: random.Random,
) -> list[TrainingImage]:
    """Sample records while first spreading selections across cameras."""

    records_by_camera: dict[str, list[TrainingImage]] = defaultdict(list)
    for record in records:
        records_by_camera[record.camera_id or "<unknown-camera>"].append(record)

    camera_ids = sorted(records_by_camera)
    rng.shuffle(camera_ids)
    for camera_id in camera_ids:
        rng.shuffle(records_by_camera[camera_id])

    selected: list[TrainingImage] = []
    while len(selected) < count:
        added_this_round = False
        for camera_id in camera_ids:
            candidates = records_by_camera[camera_id]
            if candidates:
                selected.append(candidates.pop())
                added_this_round = True
                if len(selected) == count:
                    break
        if not added_this_round:
            break
    return selected


def select_records(
    records_by_vehicle: dict[str, list[TrainingImage]],
    vehicles: int = 20,
    images_per_vehicle: int = 8,
    seed: int = 42,
    validator: ImageValidator = default_image_validator,
) -> dict[str, list[TrainingImage]]:
    """Select deterministic vehicle and image subsets.

    Only identities with at least ``images_per_vehicle`` valid images are
    eligible.  Asking for more vehicles than are eligible is an explicit
    error; no image is duplicated to hide a shortage.
    """

    if vehicles < 1:
        raise ValueError("vehicles must be positive")
    if images_per_vehicle < 1:
        raise ValueError("images_per_vehicle must be positive")

    eligible: dict[str, list[TrainingImage]] = {}
    for vehicle_id, records in records_by_vehicle.items():
        valid = [record for record in records if validator(record.source_path)]
        if len(valid) >= images_per_vehicle:
            eligible[vehicle_id] = valid

    if len(eligible) < vehicles:
        counts = ", ".join(
            f"{vehicle_id}={len(records)}"
            for vehicle_id, records in sorted(eligible.items())
        )
        raise ValueError(
            f"only {len(eligible)} vehicles have at least "
            f"{images_per_vehicle} valid images; requested {vehicles}. "
            f"Eligible counts: {counts or 'none'}"
        )

    rng = random.Random(seed)
    vehicle_ids = rng.sample(sorted(eligible), vehicles)
    vehicle_ids.sort()
    return {
        vehicle_id: _sample_across_cameras(
            eligible[vehicle_id], images_per_vehicle, rng
        )
        for vehicle_id in vehicle_ids
    }


def copy_selected_records(
    selected: dict[str, list[TrainingImage]],
    output_dir: Path,
) -> dict[str, dict[str, int]]:
    """Copy selected files into calibration layout and return summary data."""

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; "
            "choose another directory or clear it explicitly"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = {}
    for vehicle_id in sorted(selected):
        records = selected[vehicle_id]
        vehicle_dir = output_dir / f"vehicle_{vehicle_id}"
        vehicle_dir.mkdir(parents=True, exist_ok=False)
        for index, record in enumerate(records, start=1):
            destination = vehicle_dir / f"{index:04d}{record.source_path.suffix.lower()}"
            shutil.copy2(record.source_path, destination)
        camera_count = len({record.camera_id for record in records if record.camera_id})
        summary[vehicle_id] = {
            "available_images": len(records),
            "selected_images": len(records),
            "selected_camera_count": camera_count,
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vehicles", type=int, default=20)
    parser.add_argument("--images-per-vehicle", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    image_dir = dataset_root / "image_train"
    xml_path = dataset_root / "train_label.xml"
    records = parse_training_labels(xml_path, image_dir)
    valid = valid_records(records)
    selected = select_records(
        valid,
        vehicles=args.vehicles,
        images_per_vehicle=args.images_per_vehicle,
        seed=args.seed,
    )
    summary = copy_selected_records(selected, args.output)

    print(f"dataset_root={dataset_root}")
    print(f"output={Path(args.output).resolve()}")
    print(f"selected_vehicle_count={len(selected)}")
    print(f"total_images={sum(len(records) for records in selected.values())}")
    for vehicle_id in sorted(selected):
        item = summary[vehicle_id]
        print(
            f"vehicle_id={vehicle_id} "
            f"available_images={len(valid[vehicle_id])} "
            f"selected_images={item['selected_images']} "
            f"selected_camera_count={item['selected_camera_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
