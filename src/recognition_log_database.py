"""Append-only SQLite log for accepted identity events.

This database is intentionally independent from the Person and Vehicle Gallery
databases.  Deleting a Gallery identity therefore does not erase its history.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterator


class RecognitionLogRepositoryError(RuntimeError):
    """Raised when the optional recognition log cannot be written."""


@dataclass(frozen=True)
class RecognitionLogEvent:
    event_id: int
    event_time: str
    domain: str
    identity_id: int
    display_id: str
    event_type: str
    similarity: float | None = None
    track_id: int | None = None
    frame_index: int | None = None
    source_label: str | None = None
    plate_number: str | None = None


class RecognitionLogRepository:
    """Persist accepted Recognition/Recovery events without runtime coupling."""

    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        try:
            with self._transaction() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS recognition_event (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_time TEXT NOT NULL,
                        domain TEXT NOT NULL,
                        identity_id INTEGER NOT NULL,
                        display_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        similarity REAL NULL,
                        track_id INTEGER NULL,
                        frame_index INTEGER NULL,
                        source_label TEXT NULL,
                        plate_number TEXT NULL
                    )
                    """
                )
        except sqlite3.Error as exc:
            raise RecognitionLogRepositoryError(
                f"failed to initialize recognition log: {self.path}"
            ) from exc

    def record_event(
        self,
        *,
        event_time: str,
        domain: str,
        identity_id: int,
        display_id: str,
        event_type: str,
        similarity: float | None = None,
        track_id: int | None = None,
        frame_index: int | None = None,
        source_label: str | None = None,
        plate_number: str | None = None,
    ) -> RecognitionLogEvent:
        if domain not in {"person", "vehicle"}:
            raise ValueError("recognition log domain must be person or vehicle")
        if event_type not in {"recognition", "recovery"}:
            raise ValueError("recognition log event_type is invalid")
        self.initialize()
        try:
            with self._transaction() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO recognition_event(
                        event_time, domain, identity_id, display_id, event_type,
                        similarity, track_id, frame_index, source_label,
                        plate_number
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_time,
                        domain,
                        int(identity_id),
                        display_id,
                        event_type,
                        None if similarity is None else float(similarity),
                        None if track_id is None else int(track_id),
                        None if frame_index is None else int(frame_index),
                        source_label,
                        plate_number,
                    ),
                )
                event_id = int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise RecognitionLogRepositoryError(
                f"failed to write recognition event to {self.path}"
            ) from exc
        return RecognitionLogEvent(
            event_id=event_id,
            event_time=event_time,
            domain=domain,
            identity_id=int(identity_id),
            display_id=display_id,
            event_type=event_type,
            similarity=similarity,
            track_id=track_id,
            frame_index=frame_index,
            source_label=source_label,
            plate_number=plate_number,
        )

    def load_all(self) -> tuple[RecognitionLogEvent, ...]:
        self.initialize()
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            rows = connection.execute(
                """
                SELECT event_id, event_time, domain, identity_id, display_id,
                       event_type, similarity, track_id, frame_index,
                       source_label, plate_number
                FROM recognition_event ORDER BY event_id
                """
            ).fetchall()
        finally:
            connection.close()
        return tuple(RecognitionLogEvent(*row) for row in rows)
