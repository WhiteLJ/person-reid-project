from pathlib import Path
import unittest

from src.recognition_log_database import RecognitionLogRepository


class RecognitionLogDatabaseTests(unittest.TestCase):
    def test_events_are_independent_append_only_records(self) -> None:
        path = Path(__file__).resolve().parents[1] / ".test_tmp" / "recognition_log.db"
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)

    def test_max_events_is_fifo_and_survives_reopen(self) -> None:
        path = Path(__file__).resolve().parents[1] / ".test_tmp" / "recognition_log_fifo.db"
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        repository = RecognitionLogRepository(path, max_events=3)
        for index in range(5):
            repository.record_event(
                event_time=f"2026-01-01 12:00:0{index}",
                domain="person",
                identity_id=index + 1,
                display_id=f"P{index + 1:03d}",
                event_type="recognition",
            )
        self.assertEqual(
            [event.display_id for event in repository.load_all()],
            ["P003", "P004", "P005"],
        )

        reopened = RecognitionLogRepository(path, max_events=3)
        self.assertEqual(
            [event.display_id for event in reopened.load_all()],
            ["P003", "P004", "P005"],
        )
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)

    def test_initialize_trims_existing_database_to_new_limit(self) -> None:
        path = Path(__file__).resolve().parents[1] / ".test_tmp" / "recognition_log_upgrade.db"
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        old_repository = RecognitionLogRepository(path, max_events=10)
        for index in range(10):
            old_repository.record_event(
                event_time=f"2026-01-01 12:00:{index:02d}",
                domain="vehicle",
                identity_id=index + 1,
                display_id=f"V{index + 1:03d}",
                event_type="recovery",
            )
        upgraded = RecognitionLogRepository(path, max_events=3)
        upgraded.initialize()
        self.assertEqual(
            [event.display_id for event in upgraded.load_all()],
            ["V008", "V009", "V010"],
        )
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
