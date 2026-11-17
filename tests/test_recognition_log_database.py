from pathlib import Path
import unittest

from src.recognition_log_database import RecognitionLogRepository


class RecognitionLogDatabaseTests(unittest.TestCase):
    def test_events_are_independent_append_only_records(self) -> None:
        path = Path(__file__).resolve().parents[1] / ".test_tmp" / "recognition_log.db"
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        repository = RecognitionLogRepository(path)
        repository.record_event(
            event_time="2026-01-01 12:00:00",
            domain="vehicle",
            identity_id=3,
            display_id="V003",
            event_type="recognition",
            similarity=0.91,
            track_id=12,
            frame_index=42,
            source_label="Camera 0",
            plate_number="粤B12345",
        )
        events = repository.load_all()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].display_id, "V003")
        self.assertEqual(events[0].plate_number, "粤B12345")
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
