from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from src.replay_recorder import RollingReplayRecorder


class _Clock:
    def __init__(self) -> None:
        self.monotonic_value = 0.0
        self.wall_value = datetime(2026, 11, 17, 18, 0, 0)

    def monotonic(self) -> float:
        return self.monotonic_value

    def wall(self) -> datetime:
        return self.wall_value


class _FakeWriter:
    def __init__(self, path: Path, store: list["_FakeWriter"], opened=True) -> None:
        self.path = path
        self.store = store
        self.opened = opened
        self.frames: list[np.ndarray] = []
        self.released = False
        store.append(self)

    def isOpened(self) -> bool:
        return self.opened

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


def _wait_until(predicate, timeout=2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    if not predicate():
        raise AssertionError("timed out waiting for replay worker")


class ReplayRecorderTests(unittest.TestCase):
    def test_first_segment_uses_start_time_and_raw_frame(self) -> None:
        clock = _Clock()
        writers: list[_FakeWriter] = []
        recorder = RollingReplayRecorder(
            Path("replay-test"),
            segment_minutes=10,
            writer_factory=lambda path, _fourcc, _fps, _size: _FakeWriter(
                path, writers
            ),
            monotonic_clock=clock.monotonic,
            wall_clock=clock.wall,
        )
        with patch.object(Path, "mkdir"):
            self.assertTrue(recorder.start(nominal_fps=30.0))
            frame = np.zeros((4, 5, 3), dtype=np.uint8)
            frame[0, 0] = (1, 2, 3)
            self.assertTrue(recorder.submit(frame))
            frame[0, 0] = (9, 9, 9)
            _wait_until(lambda: recorder.written_frames == 1)
            recorder.close()

        self.assertEqual(len(writers), 1)
        self.assertEqual(writers[0].path.name, "2026-11-17_18-00-00.mp4")
        np.testing.assert_array_equal(writers[0].frames[0][0, 0], (1, 2, 3))
        self.assertTrue(writers[0].released)

    def test_rotation_retention_and_unmanaged_files(self) -> None:
        clock = _Clock()
        writers: list[_FakeWriter] = []
        managed: list[tuple[datetime, int, Path]] = []
        deleted: list[Path] = []
        recorder = RollingReplayRecorder(
            Path("replay-test"),
            segment_minutes=1 / 60,
            retention_segments=4,
            writer_factory=lambda path, _fourcc, _fps, _size: (
                managed.append((clock.wall(), 0, path))
                or _FakeWriter(path, writers)
            ),
            monotonic_clock=clock.monotonic,
            wall_clock=clock.wall,
        )
        recorder._managed_files = lambda: list(managed)
        recorder._delete_managed_file = lambda path: (
            deleted.append(path),
            managed.__setitem__(
                slice(None), [item for item in managed if item[2] != path]
            ),
        )
        with patch.object(Path, "mkdir"):
            recorder.start()
            for index in range(5):
                clock.monotonic_value = index * 61.0
                clock.wall_value = datetime(2026, 11, 17, 18, 0, 0) + timedelta(
                    seconds=index * 61
                )
                recorder.submit(np.full((4, 5, 3), index, dtype=np.uint8))
                _wait_until(lambda: recorder.written_frames >= index + 1)
            recorder.close()

        self.assertEqual(len(managed), 4)
        self.assertEqual(
            [path.name for path in deleted],
            ["2026-11-17_18-00-00.mp4"],
        )

    def test_same_second_does_not_overwrite(self) -> None:
        clock = _Clock()
        writers: list[_FakeWriter] = []
        managed: list[tuple[datetime, int, Path]] = []
        recorder = RollingReplayRecorder(
            Path("replay-test"),
            segment_minutes=1 / 60,
            writer_factory=lambda path, _fourcc, _fps, _size: (
                managed.append((clock.wall(), 0, path))
                or _FakeWriter(path, writers)
            ),
            monotonic_clock=clock.monotonic,
            wall_clock=clock.wall,
        )
        recorder._managed_files = lambda: list(managed)
        recorder._delete_managed_file = lambda path: managed.__setitem__(
            slice(None), [item for item in managed if item[2] != path]
        )
        with patch.object(Path, "mkdir"):
            recorder.start()
            recorder.submit(np.zeros((4, 5, 3), dtype=np.uint8))
            _wait_until(lambda: recorder.written_frames == 1)
            clock.monotonic_value = 61.0
            recorder.submit(np.ones((4, 5, 3), dtype=np.uint8))
            _wait_until(lambda: recorder.written_frames == 2)
            recorder.close()
        self.assertEqual(
            {writer.path.name for writer in writers},
            {"2026-11-17_18-00-00.mp4", "2026-11-17_18-00-00_1.mp4"},
        )

    def test_resolution_change_starts_a_new_segment(self) -> None:
        clock = _Clock()
        writers: list[_FakeWriter] = []
        recorder = RollingReplayRecorder(
            Path("replay-test"),
            writer_factory=lambda path, _fourcc, _fps, _size: _FakeWriter(
                path, writers
            ),
            monotonic_clock=clock.monotonic,
            wall_clock=clock.wall,
        )
        with patch.object(Path, "mkdir"):
            recorder.start()
            recorder.submit(np.zeros((4, 5, 3), dtype=np.uint8))
            _wait_until(lambda: recorder.written_frames == 1)
            recorder.submit(np.zeros((6, 7, 3), dtype=np.uint8))
            _wait_until(lambda: recorder.written_frames == 2)
            recorder.close()

        self.assertEqual(len(writers), 2)
        self.assertEqual([writer.path.name for writer in writers], [
            "2026-11-17_18-00-00.mp4",
            "2026-11-17_18-00-00_1.mp4",
        ])
        self.assertTrue(all(writer.released for writer in writers))

    def test_full_queue_drops_without_blocking(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        recorder = RollingReplayRecorder(Path("replay-test"), queue_size=1)

        def blocked_write(_frame):
            entered.set()
            release.wait(2.0)

        recorder._write_frame = blocked_write
        with patch.object(Path, "mkdir"):
            recorder.start()
            frame = np.zeros((4, 5, 3), dtype=np.uint8)
            recorder.submit(frame)
            self.assertTrue(entered.wait(1.0))
            recorder.submit(frame)
            started = time.monotonic()
            recorder.submit(frame)
            self.assertLess(time.monotonic() - started, 0.2)
            self.assertGreaterEqual(recorder.dropped_frames, 1)
            release.set()
            recorder.close()

    def test_writer_failure_is_non_fatal(self) -> None:
        recorder = RollingReplayRecorder(
            Path("replay-test"),
            writer_factory=lambda path, _fourcc, _fps, _size: _FakeWriter(
                path, [], opened=False
            ),
        )
        with patch.object(Path, "mkdir"):
            self.assertTrue(recorder.start())
            recorder.submit(np.zeros((4, 5, 3), dtype=np.uint8))
            _wait_until(lambda: recorder.disabled)
            recorder.close()
            self.assertTrue(recorder.disabled)


if __name__ == "__main__":
    unittest.main()
