from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication
    from src.config import load_config
    from ui.qt.runtime_worker import RuntimeWorker

    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


class _FakeCameraSource:
    source_label = "Camera 0"
    source_kind = "camera"
    is_camera = True
    read_status = "frame"
    consecutive_read_failures = 0

    def __init__(self, frames):
        self.frames = list(frames)
        self.read_count = 0
        self.reconnect_count = 0

    def read(self):
        self.read_count += 1
        if self.frames:
            return self.frames.pop(0)
        self.read_status = "temporary_failure"
        self.consecutive_read_failures += 1
        return None

    def reconnect(self):
        self.reconnect_count += 1
        return False


@unittest.skipUnless(QT_AVAILABLE, "PyQt6 is an optional UI dependency")
class RuntimeWorkerCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.config = load_config("config/config.yaml")

    @staticmethod
    def _worker_with_stubs(source) -> RuntimeWorker:
        worker = RuntimeWorker(RuntimeWorkerCameraTests.config)
        worker._source = source
        worker._tracking_pipeline = Mock()
        worker._tracking_pipeline.process.return_value = SimpleNamespace(
            person_tracks=[], vehicle_tracks=[]
        )
        worker._person_guard = Mock()
        worker._person_guard.process_frame.return_value = SimpleNamespace(
            blocked_reference_update_target_ids=()
        )
        worker._person_recovery = Mock()
        worker._person_recovery.last_recovered_track_ids = set()
        worker._vehicle_recovery = Mock()
        worker._vehicle_recovery.last_recovered_track_ids = set()
        worker._person_manager = SimpleNamespace(targets={})
        worker._vehicle_manager = SimpleNamespace(targets={})
        worker._person_service = Mock()
        worker._vehicle_service = Mock()
        worker._person_recognition = Mock()
        worker._vehicle_recognition = Mock()
        worker._vehicle_recognition.process_frame.return_value = []
        worker._render = lambda frame, **kwargs: frame
        worker._runtime_frame = lambda frame: object()
        return worker

    def test_editing_keeps_camera_pipeline_running_but_does_not_emit_live_frame(self) -> None:
        source = _FakeCameraSource([np.zeros((8, 10, 3), dtype=np.uint8)])
        worker = self._worker_with_stubs(source)
        worker._editing = True
        frames = []
        worker.frame_ready.connect(frames.append)

        worker.process_next_frame()

        self.assertEqual(source.read_count, 1)
        worker._tracking_pipeline.process.assert_called_once()
        self.assertEqual(frames, [])

    def test_camera_read_failure_does_not_shutdown_worker(self) -> None:
        source = _FakeCameraSource([])
        worker = self._worker_with_stubs(source)
        shutdown = []
        worker.shutdown_complete.connect(lambda: shutdown.append(True))

        worker.process_next_frame()

        self.assertEqual(shutdown, [])
        self.assertEqual(source.reconnect_count, 1)


if __name__ == "__main__":
    unittest.main()
