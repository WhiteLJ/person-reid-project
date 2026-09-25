from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.prepare_vehicle_calibration import (
    copy_selected_records,
    parse_training_labels,
    select_records,
)


_TEST_TEMP_PARENT = Path(__file__).resolve().parents[1] / ".test_tmp"
_TEST_TEMP_PARENT.mkdir(parents=True, exist_ok=True)


class PrepareVehicleCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=_TEST_TEMP_PARENT)
        self.root = Path(self.temp_dir.name)
        self.image_dir = self.root / "image_train"
        self.image_dir.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_xml(self, rows: list[tuple[str, str, str]]) -> Path:
        xml = [
            '<?xml version="1.0" encoding="gb2312" ?>',
            '<TrainingImages Version="1.0">',
        ]
        for image_name, vehicle_id, camera_id in rows:
            xml.append(
                f'<Item imageName="{image_name}" vehicleID="{vehicle_id}" '
                f'cameraID="{camera_id}" />'
            )
        xml.append("</TrainingImages>")
        path = self.root / "train_label.xml"
        path.write_bytes("\n".join(xml).encode("gb2312"))
        return path

    def test_identity_and_camera_are_read_from_xml(self) -> None:
        image = self.image_dir / "renamed.jpg"
        image.write_bytes(b"image")
        labels = self._write_xml([("renamed.jpg", "0269", "c026")])

        grouped = parse_training_labels(labels, self.image_dir)

        self.assertEqual(list(grouped), ["0269"])
        self.assertEqual(grouped["0269"][0].camera_id, "c026")
        self.assertEqual(grouped["0269"][0].image_name, "renamed.jpg")

    def test_insufficient_identity_is_not_selected(self) -> None:
        rows: list[tuple[str, str, str]] = []
        for index in range(3):
            name = f"a{index}.jpg"
            (self.image_dir / name).write_bytes(name.encode())
            rows.append((name, "A", "c1"))
        name = "b0.jpg"
        (self.image_dir / name).write_bytes(b"b")
        rows.append((name, "B", "c2"))
        labels = self._write_xml(rows)
        grouped = parse_training_labels(labels, self.image_dir)

        selected = select_records(grouped, vehicles=1, images_per_vehicle=3, seed=42)
        self.assertEqual(set(selected), {"A"})

    def test_fixed_seed_reproduces_vehicle_and_image_selection(self) -> None:
        rows: list[tuple[str, str, str]] = []
        for vehicle_id in ("A", "B", "C"):
            for index in range(5):
                name = f"{vehicle_id}{index}.jpg"
                (self.image_dir / name).write_bytes(name.encode())
                rows.append((name, vehicle_id, f"c{index + 1}"))
        labels = self._write_xml(rows)
        grouped = parse_training_labels(labels, self.image_dir)

        first = select_records(grouped, vehicles=2, images_per_vehicle=4, seed=42)
        second = select_records(grouped, vehicles=2, images_per_vehicle=4, seed=42)
        self.assertEqual(
            {
                vehicle: [record.image_name for record in records]
                for vehicle, records in first.items()
            },
            {
                vehicle: [record.image_name for record in records]
                for vehicle, records in second.items()
            },
        )

    def test_copy_has_correct_counts_and_does_not_modify_source(self) -> None:
        rows = [(f"a{index}.jpg", "A", "c1") for index in range(2)]
        source_bytes = {}
        for name, _, _ in rows:
            path = self.image_dir / name
            path.write_bytes(name.encode())
            source_bytes[name] = hashlib.sha256(path.read_bytes()).digest()
        labels = self._write_xml(rows)
        grouped = parse_training_labels(labels, self.image_dir)
        selected = select_records(grouped, vehicles=1, images_per_vehicle=2, seed=42)
        output = self.root / "output"

        summary = copy_selected_records(selected, output)

        files = sorted((output / "vehicle_A").iterdir())
        self.assertEqual(len(files), 2)
        self.assertEqual(summary["A"]["selected_images"], 2)
        self.assertEqual(
            {hashlib.sha256(path.read_bytes()).digest() for path in files},
            set(source_bytes.values()),
        )
        self.assertEqual(
            {name: hashlib.sha256((self.image_dir / name).read_bytes()).digest()
             for name in source_bytes},
            source_bytes,
        )

    def test_output_vehicle_directory_is_identity_scoped(self) -> None:
        rows = [(f"a{index}.jpg", "A", "c1") for index in range(2)]
        for name, _, _ in rows:
            (self.image_dir / name).write_bytes(name.encode())
        labels = self._write_xml(rows)
        grouped = parse_training_labels(labels, self.image_dir)
        selected = select_records(grouped, vehicles=1, images_per_vehicle=2, seed=42)
        output = self.root / "output"

        copy_selected_records(selected, output)

        self.assertEqual(
            [path.name for path in output.iterdir() if path.is_dir()],
            ["vehicle_A"],
        )


if __name__ == "__main__":
    unittest.main()
