from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import unittest

import numpy as np

from src.vehicle_database import VehicleGalleryRepository
from src.vehicle_gallery import GalleryVehicle
from tools.vehicle_gallery_admin import main


_TEST_TEMP_PARENT = Path(__file__).resolve().parents[1] / ".test_tmp"
_TEST_TEMP_PARENT.mkdir(parents=True, exist_ok=True)


def _remove_database_files(path: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _vehicle(vehicle_id: int) -> GalleryVehicle:
    reference = np.zeros((2048,), dtype=np.float32)
    reference[0] = 1.0
    return GalleryVehicle(
        vehicle_id=vehicle_id,
        label=f"Target V{vehicle_id:03d}",
        reference_embeddings=[reference.copy()],
        centroid=reference.copy(),
    )


class VehicleGalleryAdminTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = _TEST_TEMP_PARENT / "vehicle_gallery_admin_tests.db"
        _remove_database_files(self.path)
        VehicleGalleryRepository(self.path).save_vehicle(_vehicle(1))

    def tearDown(self) -> None:
        _remove_database_files(self.path)

    def test_list_remove_and_clear(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--db", str(self.path), "list"]), 0)
        self.assertIn("V001\tTarget V001\treferences=1", output.getvalue())

        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main(["--db", str(self.path), "remove", "V001"]),
                0,
            )
        self.assertIn("Removed V001", output.getvalue())
        self.assertEqual(VehicleGalleryRepository(self.path).load_all(), ())

        VehicleGalleryRepository(self.path).save_vehicle(_vehicle(2))
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--db", str(self.path), "clear"]), 0)
        self.assertIn("Vehicle Gallery cleared", output.getvalue())
        self.assertEqual(VehicleGalleryRepository(self.path).load_all(), ())

    def test_missing_remove_is_reported(self) -> None:
        error = StringIO()
        with redirect_stderr(error):
            result = main(["--db", str(self.path), "remove", "V999"])
        self.assertEqual(result, 1)
        self.assertIn("Vehicle not found: V999", error.getvalue())


if __name__ == "__main__":
    unittest.main()
