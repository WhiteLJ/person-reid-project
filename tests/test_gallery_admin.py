from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import unittest

import numpy as np

from src.database import GalleryRepository
from src.gallery import GalleryPerson
from tools.gallery_admin import main


_TEST_TEMP_PARENT = Path(__file__).resolve().parents[1] / ".test_tmp"
_TEST_TEMP_PARENT.mkdir(parents=True, exist_ok=True)


def _remove_database_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)


def _person(person_id: int) -> GalleryPerson:
    embedding = np.zeros((512,), dtype=np.float32)
    embedding[person_id - 1] = 1.0
    return GalleryPerson(
        person_id=person_id,
        label=f"Person {person_id}",
        reference_embeddings=[embedding.copy()],
        centroid=embedding.copy(),
    )


class GalleryAdminTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database_path = _TEST_TEMP_PARENT / "gallery_admin_tests.db"
        _remove_database_files(self.database_path)
        self.repository = GalleryRepository(self.database_path)
        self.repository.save_person(_person(1))

    def tearDown(self) -> None:
        _remove_database_files(self.database_path)

    def test_list_command(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--db", str(self.database_path), "list"])

        self.assertEqual(result, 0)
        self.assertIn("P001", output.getvalue())
        self.assertIn("references=1", output.getvalue())

    def test_remove_command(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--db", str(self.database_path), "remove", "P001"])

        self.assertEqual(result, 0)
        self.assertEqual(self.repository.load_all(), ())
        self.assertIn("Removed P001", output.getvalue())

    def test_clear_command(self) -> None:
        self.repository.save_person(_person(2))
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--db", str(self.database_path), "clear"])

        self.assertEqual(result, 0)
        self.assertEqual(self.repository.load_all(), ())
        self.assertIn("Gallery cleared", output.getvalue())


if __name__ == "__main__":
    unittest.main()
