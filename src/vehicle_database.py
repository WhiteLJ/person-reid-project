"""SQLite persistence isolated from the Vehicle Gallery domain model."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Sequence
from pathlib import Path
import sqlite3
from typing import Iterator

import numpy as np

from .vehicle_gallery import (
    GalleryVehicle,
    VEHICLE_EMBEDDING_DIMENSION,
    VEHICLE_EMBEDDING_DTYPE,
    _validate_vehicle_id,
)


_EMBEDDING_BYTES = VEHICLE_EMBEDDING_DIMENSION * np.dtype("<f4").itemsize
_NEXT_VEHICLE_ID_KEY = "next_vehicle_id"


class VehicleRepositoryError(RuntimeError):
    """Raised when Vehicle Gallery persistence or validation fails."""


class VehicleGalleryRepository:
    """Persist only Vehicle Gallery identities and their embeddings."""

    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)

    def initialize(self) -> None:
        try:
            with self._transaction() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS gallery_vehicle (
                        vehicle_id INTEGER PRIMARY KEY,
                        label TEXT NOT NULL,
                        centroid BLOB NOT NULL,
                        centroid_dim INTEGER NOT NULL,
                        centroid_dtype TEXT NOT NULL,
                        CHECK (centroid_dim = 2048),
                        CHECK (centroid_dtype = 'float32')
                    );

                    CREATE TABLE IF NOT EXISTS gallery_vehicle_embedding (
                        vehicle_id INTEGER NOT NULL,
                        embedding_index INTEGER NOT NULL,
                        embedding BLOB NOT NULL,
                        embedding_dim INTEGER NOT NULL,
                        embedding_dtype TEXT NOT NULL,
                        PRIMARY KEY (vehicle_id, embedding_index),
                        FOREIGN KEY (vehicle_id)
                            REFERENCES gallery_vehicle(vehicle_id)
                            ON DELETE CASCADE,
                        CHECK (embedding_index >= 0),
                        CHECK (embedding_dim = 2048),
                        CHECK (embedding_dtype = 'float32')
                    );

                    CREATE TABLE IF NOT EXISTS gallery_vehicle_meta (
                        key TEXT PRIMARY KEY,
                        value INTEGER NOT NULL
                    );
                    """
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO gallery_vehicle_meta(key, value)
                    VALUES (?, 1)
                    """,
                    (_NEXT_VEHICLE_ID_KEY,),
                )
                self._ensure_next_vehicle_id(connection)
        except VehicleRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise VehicleRepositoryError(
                f"failed to initialize Vehicle Gallery database: {self.path}"
            ) from exc

    def load_all(self) -> tuple[GalleryVehicle, ...]:
        self.initialize()
        connection = self._connect()
        try:
            vehicle_rows = connection.execute(
                """
                SELECT vehicle_id, label, centroid, centroid_dim, centroid_dtype
                FROM gallery_vehicle
                ORDER BY vehicle_id
                """
            ).fetchall()
            embedding_rows = connection.execute(
                """
                SELECT vehicle_id, embedding_index, embedding,
                       embedding_dim, embedding_dtype
                FROM gallery_vehicle_embedding
                ORDER BY vehicle_id, embedding_index
                """
            ).fetchall()
        except sqlite3.Error as exc:
            raise VehicleRepositoryError(
                f"failed to load Vehicle Gallery database: {self.path}"
            ) from exc
        finally:
            connection.close()

        references_by_vehicle: dict[int, list[np.ndarray]] = {
            int(row[0]): [] for row in vehicle_rows
        }
        for row in embedding_rows:
            vehicle_id = int(row[0])
            if vehicle_id not in references_by_vehicle:
                raise VehicleRepositoryError(
                    f"database contains embedding for unknown vehicle_id={vehicle_id}"
                )
            embedding_index = int(row[1])
            if embedding_index != len(references_by_vehicle[vehicle_id]):
                raise VehicleRepositoryError(
                    f"vehicle_id={vehicle_id} has non-contiguous embedding indexes"
                )
            references_by_vehicle[vehicle_id].append(
                self._decode_embedding(
                    row[2],
                    row[3],
                    row[4],
                    context=(
                        f"vehicle_id={vehicle_id} "
                        f"embedding_index={embedding_index}"
                    ),
                )
            )

        vehicles: list[GalleryVehicle] = []
        for row in vehicle_rows:
            vehicle_id = _validate_vehicle_id(int(row[0]))
            label = row[1]
            if not isinstance(label, str) or not label.strip():
                raise VehicleRepositoryError(
                    f"vehicle_id={vehicle_id} has an invalid label"
                )
            references = references_by_vehicle[vehicle_id]
            if not references:
                raise VehicleRepositoryError(
                    f"vehicle_id={vehicle_id} has no reference embeddings"
                )
            centroid = self._decode_embedding(
                row[2],
                row[3],
                row[4],
                context=f"vehicle_id={vehicle_id} centroid",
            )
            vehicles.append(
                GalleryVehicle(
                    vehicle_id=vehicle_id,
                    label=label,
                    reference_embeddings=[reference.copy() for reference in references],
                    centroid=centroid.copy(),
                )
            )
        return tuple(vehicles)

    def load_vehicle(self, vehicle_id: int) -> GalleryVehicle | None:
        vehicle_id = _validate_vehicle_id(vehicle_id)
        return next(
            (vehicle for vehicle in self.load_all() if vehicle.vehicle_id == vehicle_id),
            None,
        )

    def load_next_vehicle_id(self) -> int:
        self.initialize()
        with self._transaction() as connection:
            return self._ensure_next_vehicle_id(connection)

    def save_vehicle(self, vehicle: GalleryVehicle) -> None:
        vehicle_id = _validate_vehicle_id(vehicle.vehicle_id)
        label = self._validate_label(vehicle.label, vehicle_id)
        centroid_blob = self._encode_embedding(
            vehicle.centroid,
            context=f"vehicle_id={vehicle_id} centroid",
        )
        reference_blobs = [
            self._encode_embedding(
                reference,
                context=f"vehicle_id={vehicle_id} embedding_index={index}",
            )
            for index, reference in enumerate(vehicle.reference_embeddings)
        ]
        if not reference_blobs:
            raise VehicleRepositoryError(
                f"vehicle_id={vehicle_id} must contain at least one embedding"
            )

        self.initialize()
        try:
            with self._transaction() as connection:
                next_vehicle_id = self._ensure_next_vehicle_id(connection)
                if vehicle_id < next_vehicle_id:
                    raise VehicleRepositoryError(
                        f"vehicle_id={vehicle_id} would reuse an allocated historical ID"
                    )
                connection.execute(
                    """
                    INSERT INTO gallery_vehicle(
                        vehicle_id, label, centroid, centroid_dim, centroid_dtype
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        vehicle_id,
                        label,
                        centroid_blob,
                        VEHICLE_EMBEDDING_DIMENSION,
                        VEHICLE_EMBEDDING_DTYPE,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO gallery_vehicle_embedding(
                        vehicle_id, embedding_index, embedding,
                        embedding_dim, embedding_dtype
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            vehicle_id,
                            index,
                            blob,
                            VEHICLE_EMBEDDING_DIMENSION,
                            VEHICLE_EMBEDDING_DTYPE,
                        )
                        for index, blob in enumerate(reference_blobs)
                    ],
                )
                connection.execute(
                    """
                    UPDATE gallery_vehicle_meta
                    SET value = MAX(value, ?)
                    WHERE key = ?
                    """,
                    (vehicle_id + 1, _NEXT_VEHICLE_ID_KEY),
                )
        except VehicleRepositoryError:
            raise
        except sqlite3.IntegrityError as exc:
            raise VehicleRepositoryError(
                f"failed to save vehicle_id={vehicle_id}: database constraint error"
            ) from exc
        except sqlite3.Error as exc:
            raise VehicleRepositoryError(
                f"failed to save vehicle_id={vehicle_id} in {self.path}"
            ) from exc

    def update_vehicle_features(
        self,
        vehicle_id: int,
        reference_embeddings: Sequence[np.ndarray],
        centroid: np.ndarray,
    ) -> None:
        vehicle_id = _validate_vehicle_id(vehicle_id)
        centroid_blob = self._encode_embedding(
            centroid,
            context=f"vehicle_id={vehicle_id} centroid",
        )
        reference_blobs = [
            self._encode_embedding(
                reference,
                context=f"vehicle_id={vehicle_id} embedding_index={index}",
            )
            for index, reference in enumerate(reference_embeddings)
        ]
        if not reference_blobs:
            raise VehicleRepositoryError(
                f"vehicle_id={vehicle_id} must contain at least one embedding"
            )

        self.initialize()
        try:
            with self._transaction() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM gallery_vehicle WHERE vehicle_id = ?",
                    (vehicle_id,),
                ).fetchone()
                if exists is None:
                    raise VehicleRepositoryError(
                        f"cannot update unknown vehicle_id={vehicle_id}"
                    )
                connection.execute(
                    """
                    UPDATE gallery_vehicle
                    SET centroid = ?, centroid_dim = ?, centroid_dtype = ?
                    WHERE vehicle_id = ?
                    """,
                    (
                        centroid_blob,
                        VEHICLE_EMBEDDING_DIMENSION,
                        VEHICLE_EMBEDDING_DTYPE,
                        vehicle_id,
                    ),
                )
                connection.execute(
                    "DELETE FROM gallery_vehicle_embedding WHERE vehicle_id = ?",
                    (vehicle_id,),
                )
                connection.executemany(
                    """
                    INSERT INTO gallery_vehicle_embedding(
                        vehicle_id, embedding_index, embedding,
                        embedding_dim, embedding_dtype
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            vehicle_id,
                            index,
                            blob,
                            VEHICLE_EMBEDDING_DIMENSION,
                            VEHICLE_EMBEDDING_DTYPE,
                        )
                        for index, blob in enumerate(reference_blobs)
                    ],
                )
        except VehicleRepositoryError:
            raise
        except sqlite3.IntegrityError as exc:
            raise VehicleRepositoryError(
                f"failed to update vehicle_id={vehicle_id}: database constraint error"
            ) from exc
        except sqlite3.Error as exc:
            raise VehicleRepositoryError(
                f"failed to update vehicle_id={vehicle_id} in {self.path}"
            ) from exc

    def delete_vehicle(self, vehicle_id: int) -> bool:
        vehicle_id = _validate_vehicle_id(vehicle_id)
        self.initialize()
        try:
            with self._transaction() as connection:
                cursor = connection.execute(
                    "DELETE FROM gallery_vehicle WHERE vehicle_id = ?",
                    (vehicle_id,),
                )
                return cursor.rowcount == 1
        except sqlite3.Error as exc:
            raise VehicleRepositoryError(
                f"failed to delete vehicle_id={vehicle_id} from {self.path}"
            ) from exc

    def clear(self) -> None:
        self.initialize()
        try:
            with self._transaction() as connection:
                connection.execute("DELETE FROM gallery_vehicle")
        except sqlite3.Error as exc:
            raise VehicleRepositoryError(
                f"failed to clear Vehicle Gallery database: {self.path}"
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

    def _ensure_next_vehicle_id(self, connection: sqlite3.Connection) -> int:
        max_vehicle_id = connection.execute(
            "SELECT COALESCE(MAX(vehicle_id), 0) FROM gallery_vehicle"
        ).fetchone()[0]
        row = connection.execute(
            "SELECT value FROM gallery_vehicle_meta WHERE key = ?",
            (_NEXT_VEHICLE_ID_KEY,),
        ).fetchone()
        if row is None:
            next_vehicle_id = int(max_vehicle_id) + 1
            connection.execute(
                "INSERT INTO gallery_vehicle_meta(key, value) VALUES (?, ?)",
                (_NEXT_VEHICLE_ID_KEY, next_vehicle_id),
            )
            return next_vehicle_id

        value = row[0]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise VehicleRepositoryError(
                "gallery_vehicle_meta.next_vehicle_id is invalid; "
                "refusing to allocate IDs"
            )
        if value <= int(max_vehicle_id):
            value = int(max_vehicle_id) + 1
            connection.execute(
                "UPDATE gallery_vehicle_meta SET value = ? WHERE key = ?",
                (value, _NEXT_VEHICLE_ID_KEY),
            )
        return value

    @staticmethod
    def _validate_label(label: str, vehicle_id: int) -> str:
        if not isinstance(label, str) or not label.strip():
            raise VehicleRepositoryError(
                f"vehicle_id={vehicle_id} has an invalid label"
            )
        return label

    @classmethod
    def _encode_embedding(cls, embedding: np.ndarray, *, context: str) -> bytes:
        array = np.asarray(embedding)
        if array.dtype != np.dtype(np.float32):
            raise VehicleRepositoryError(f"{context} must use dtype float32")
        if array.shape != (VEHICLE_EMBEDDING_DIMENSION,):
            raise VehicleRepositoryError(
                f"{context} must have shape ({VEHICLE_EMBEDDING_DIMENSION},), "
                f"got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise VehicleRepositoryError(f"{context} contains non-finite values")
        if not np.isclose(float(np.linalg.norm(array)), 1.0, atol=1e-3):
            raise VehicleRepositoryError(f"{context} must be L2 normalized")
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
        if dimension != VEHICLE_EMBEDDING_DIMENSION or dtype != VEHICLE_EMBEDDING_DTYPE:
            raise VehicleRepositoryError(
                f"{context} metadata must be dimension=2048 and dtype=float32"
            )
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise VehicleRepositoryError(f"{context} BLOB has an invalid type")
        raw = bytes(blob)
        if len(raw) != _EMBEDDING_BYTES:
            raise VehicleRepositoryError(
                f"{context} BLOB has invalid length {len(raw)}; "
                f"expected {_EMBEDDING_BYTES}"
            )
        array = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
        if array.shape != (VEHICLE_EMBEDDING_DIMENSION,) or not np.isfinite(array).all():
            raise VehicleRepositoryError(f"{context} contains invalid float32 data")
        if not np.isclose(float(np.linalg.norm(array)), 1.0, atol=1e-3):
            raise VehicleRepositoryError(f"{context} is not L2 normalized")
        return array
