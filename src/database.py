"""SQLite persistence for GalleryPerson records.

This module is deliberately independent from the in-memory TargetGallery
business rules.  It stores only GalleryPerson data and a monotonic allocator
for person IDs; runtime SessionTarget and Track state is never persisted.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Sequence
from pathlib import Path
from typing import Iterator
import sqlite3

import numpy as np

from .gallery import GalleryPerson
EMBEDDING_DIMENSION = 512
EMBEDDING_DTYPE = "float32"
_EMBEDDING_BYTES = EMBEDDING_DIMENSION * np.dtype("<f4").itemsize
_NEXT_PERSON_ID_KEY = "next_person_id"


class RepositoryError(RuntimeError):
    """Raised when Gallery persistence or validation fails."""


class GalleryRepository:
    """Persist GalleryPerson records in a small, self-contained SQLite DB."""

    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)

    def initialize(self) -> None:
        """Create the database directory, schema, and ID allocator."""

        with self._transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS gallery_person (
                    person_id INTEGER PRIMARY KEY,
                    label TEXT NOT NULL,
                    centroid BLOB NOT NULL,
                    centroid_dim INTEGER NOT NULL,
                    centroid_dtype TEXT NOT NULL,
                    CHECK (centroid_dim = 512),
                    CHECK (centroid_dtype = 'float32')
                );

                CREATE TABLE IF NOT EXISTS gallery_embedding (
                    person_id INTEGER NOT NULL,
                    embedding_index INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding_dtype TEXT NOT NULL,
                    PRIMARY KEY (person_id, embedding_index),
                    FOREIGN KEY (person_id)
                        REFERENCES gallery_person(person_id)
                        ON DELETE CASCADE,
                    CHECK (embedding_index >= 0),
                    CHECK (embedding_dim = 512),
                    CHECK (embedding_dtype = 'float32')
                );

                CREATE TABLE IF NOT EXISTS gallery_meta (
                    key TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                );
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO gallery_meta(key, value)
                VALUES (?, 1)
                """,
                (_NEXT_PERSON_ID_KEY,),
            )
            self._ensure_next_person_id(connection)

    def load_all(self) -> tuple[GalleryPerson, ...]:
        """Load all persisted people, validating every stored embedding."""

        self.initialize()
        connection = self._connect()
        try:
            person_rows = connection.execute(
                """
                SELECT person_id, label, centroid, centroid_dim, centroid_dtype
                FROM gallery_person
                ORDER BY person_id
                """
            ).fetchall()
            embedding_rows = connection.execute(
                """
                SELECT person_id, embedding_index, embedding,
                       embedding_dim, embedding_dtype
                FROM gallery_embedding
                ORDER BY person_id, embedding_index
                """
            ).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError(
                f"failed to load Gallery from SQLite database: {self.path}"
            ) from exc
        finally:
            connection.close()

        references_by_person: dict[int, list[np.ndarray]] = {
            int(row[0]): [] for row in person_rows
        }
        for row in embedding_rows:
            person_id = int(row[0])
            if person_id not in references_by_person:
                raise RepositoryError(
                    f"database contains an embedding for unknown person_id={person_id}"
                )
            embedding_index = int(row[1])
            if embedding_index != len(references_by_person[person_id]):
                raise RepositoryError(
                    f"person_id={person_id} has non-contiguous embedding indexes"
                )
            references_by_person[person_id].append(
                self._decode_embedding(
                    row[2],
                    row[3],
                    row[4],
                    context=(
                        f"person_id={person_id} embedding_index={embedding_index}"
                    ),
                )
            )

        people: list[GalleryPerson] = []
        for row in person_rows:
            person_id = int(row[0])
            label = row[1]
            if not isinstance(label, str) or not label.strip():
                raise RepositoryError(f"person_id={person_id} has an invalid label")
            references = references_by_person[person_id]
            if not references:
                raise RepositoryError(
                    f"person_id={person_id} has no reference embeddings"
                )
            centroid = self._decode_embedding(
                row[2],
                row[3],
                row[4],
                context=f"person_id={person_id} centroid",
            )
            people.append(
                GalleryPerson(
                    person_id=person_id,
                    label=label,
                    reference_embeddings=[reference.copy() for reference in references],
                    centroid=centroid.copy(),
                )
            )
        return tuple(people)

    def load_next_person_id(self) -> int:
        """Return a validated, monotonic next ID and repair stale low metadata."""

        self.initialize()
        with self._transaction() as connection:
            return self._ensure_next_person_id(connection)

    def save_person(self, person: GalleryPerson) -> None:
        """Atomically insert one person, all references, and allocator metadata."""

        person_id = self._validate_person_id(person.person_id)
        label = self._validate_label(person.label, person_id)
        centroid_blob = self._encode_embedding(
            person.centroid,
            context=f"person_id={person_id} centroid",
        )
        reference_blobs = [
            self._encode_embedding(
                reference,
                context=f"person_id={person_id} embedding_index={index}",
            )
            for index, reference in enumerate(person.reference_embeddings)
        ]
        if not reference_blobs:
            raise RepositoryError(
                f"person_id={person_id} must contain at least one embedding"
            )

        self.initialize()
        try:
            with self._transaction() as connection:
                next_person_id = self._ensure_next_person_id(connection)
                if person_id < next_person_id:
                    raise RepositoryError(
                        f"person_id={person_id} would reuse an allocated historical ID"
                    )
                connection.execute(
                    """
                    INSERT INTO gallery_person(
                        person_id, label, centroid, centroid_dim, centroid_dtype
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        person_id,
                        label,
                        centroid_blob,
                        EMBEDDING_DIMENSION,
                        EMBEDDING_DTYPE,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO gallery_embedding(
                        person_id, embedding_index, embedding,
                        embedding_dim, embedding_dtype
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            person_id,
                            index,
                            blob,
                            EMBEDDING_DIMENSION,
                            EMBEDDING_DTYPE,
                        )
                        for index, blob in enumerate(reference_blobs)
                    ],
                )
                connection.execute(
                    """
                    UPDATE gallery_meta
                    SET value = MAX(value, ?)
                    WHERE key = ?
                    """,
                    (person_id + 1, _NEXT_PERSON_ID_KEY),
                )
        except sqlite3.IntegrityError as exc:
            raise RepositoryError(
                f"failed to save person_id={person_id}: database constraint error"
            ) from exc
        except sqlite3.Error as exc:
            raise RepositoryError(
                f"failed to save person_id={person_id} in {self.path}"
            ) from exc

    def update_person_features(
        self,
        person_id: int,
        reference_embeddings: Sequence[np.ndarray],
        centroid: np.ndarray,
    ) -> None:
        """Atomically replace one person's centroid and reference embeddings.

        The person row and its child rows are updated in one transaction.  The
        stable person ID, label, and monotonic ID allocator are intentionally
        untouched.
        """

        person_id = self._validate_person_id(person_id)
        centroid_blob = self._encode_embedding(
            centroid,
            context=f"person_id={person_id} centroid",
        )
        reference_blobs = [
            self._encode_embedding(
                reference,
                context=f"person_id={person_id} embedding_index={index}",
            )
            for index, reference in enumerate(reference_embeddings)
        ]
        if not reference_blobs:
            raise RepositoryError(
                f"person_id={person_id} must contain at least one embedding"
            )

        self.initialize()
        try:
            with self._transaction() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM gallery_person WHERE person_id = ?",
                    (person_id,),
                ).fetchone()
                if exists is None:
                    raise RepositoryError(
                        f"cannot update unknown person_id={person_id}"
                    )

                connection.execute(
                    """
                    UPDATE gallery_person
                    SET centroid = ?, centroid_dim = ?, centroid_dtype = ?
                    WHERE person_id = ?
                    """,
                    (
                        centroid_blob,
                        EMBEDDING_DIMENSION,
                        EMBEDDING_DTYPE,
                        person_id,
                    ),
                )
                connection.execute(
                    "DELETE FROM gallery_embedding WHERE person_id = ?",
                    (person_id,),
                )
                connection.executemany(
                    """
                    INSERT INTO gallery_embedding(
                        person_id, embedding_index, embedding,
                        embedding_dim, embedding_dtype
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            person_id,
                            index,
                            blob,
                            EMBEDDING_DIMENSION,
                            EMBEDDING_DTYPE,
                        )
                        for index, blob in enumerate(reference_blobs)
                    ],
                )
        except RepositoryError:
            raise
        except sqlite3.IntegrityError as exc:
            raise RepositoryError(
                f"failed to update person_id={person_id}: database constraint error"
            ) from exc
        except sqlite3.Error as exc:
            raise RepositoryError(
                f"failed to update person_id={person_id} in {self.path}"
            ) from exc

    def load_person(self, person_id: int) -> GalleryPerson | None:
        """Load one persisted person for consistency recovery after an apply error."""

        person_id = self._validate_person_id(person_id)
        return next(
            (person for person in self.load_all() if person.person_id == person_id),
            None,
        )

    def delete_person(self, person_id: int) -> bool:
        """Delete one person; SQLite cascades to all of its embeddings."""

        person_id = self._validate_person_id(person_id)
        self.initialize()
        try:
            with self._transaction() as connection:
                cursor = connection.execute(
                    "DELETE FROM gallery_person WHERE person_id = ?",
                    (person_id,),
                )
                return cursor.rowcount == 1
        except sqlite3.Error as exc:
            raise RepositoryError(
                f"failed to delete person_id={person_id} from {self.path}"
            ) from exc

    def clear(self) -> None:
        """Delete all people and embeddings without resetting the ID sequence."""

        self.initialize()
        try:
            with self._transaction() as connection:
                connection.execute("DELETE FROM gallery_person")
        except sqlite3.Error as exc:
            raise RepositoryError(
                f"failed to clear Gallery database: {self.path}"
            ) from exc

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path))
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _ensure_next_person_id(self, connection: sqlite3.Connection) -> int:
        max_person_id = connection.execute(
            "SELECT COALESCE(MAX(person_id), 0) FROM gallery_person"
        ).fetchone()[0]
        row = connection.execute(
            "SELECT value FROM gallery_meta WHERE key = ?",
            (_NEXT_PERSON_ID_KEY,),
        ).fetchone()
        if row is None:
            next_person_id = int(max_person_id) + 1
            connection.execute(
                "INSERT INTO gallery_meta(key, value) VALUES (?, ?)",
                (_NEXT_PERSON_ID_KEY, next_person_id),
            )
            return next_person_id

        value = row[0]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RepositoryError(
                "gallery_meta.next_person_id is invalid; refusing to allocate IDs"
            )
        if value <= int(max_person_id):
            value = int(max_person_id) + 1
            connection.execute(
                "UPDATE gallery_meta SET value = ? WHERE key = ?",
                (value, _NEXT_PERSON_ID_KEY),
            )
        return value

    @staticmethod
    def _validate_person_id(person_id: int) -> int:
        if isinstance(person_id, bool) or not isinstance(person_id, int) or person_id < 1:
            raise RepositoryError("person_id must be a positive integer")
        return person_id

    @staticmethod
    def _validate_label(label: str, person_id: int) -> str:
        if not isinstance(label, str) or not label.strip():
            raise RepositoryError(f"person_id={person_id} has an invalid label")
        return label

    @classmethod
    def _encode_embedding(cls, embedding: np.ndarray, *, context: str) -> bytes:
        array = np.asarray(embedding)
        if array.dtype != np.float32:
            raise RepositoryError(f"{context} must use dtype float32")
        if array.shape != (EMBEDDING_DIMENSION,):
            raise RepositoryError(
                f"{context} must have shape ({EMBEDDING_DIMENSION},), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise RepositoryError(f"{context} contains non-finite values")
        norm = float(np.linalg.norm(array))
        if not np.isclose(norm, 1.0, atol=1e-3):
            raise RepositoryError(f"{context} must be L2 normalized")
        return np.asarray(array, dtype="<f4", order="C").tobytes()

    @classmethod
    def _decode_embedding(
        cls,
        blob: bytes,
        dimension: int,
        dtype: str,
        *,
        context: str,
    ) -> np.ndarray:
        if dimension != EMBEDDING_DIMENSION or dtype != EMBEDDING_DTYPE:
            raise RepositoryError(
                f"{context} metadata must be dimension=512 and dtype=float32"
            )
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise RepositoryError(f"{context} BLOB has an invalid type")
        raw = bytes(blob)
        if len(raw) != _EMBEDDING_BYTES:
            raise RepositoryError(
                f"{context} BLOB has invalid length {len(raw)}; expected {_EMBEDDING_BYTES}"
            )
        array = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
        if array.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(array).all():
            raise RepositoryError(f"{context} contains invalid float32 data")
        norm = float(np.linalg.norm(array))
        if not np.isclose(norm, 1.0, atol=1e-3):
            raise RepositoryError(f"{context} is not L2 normalized")
        return array
