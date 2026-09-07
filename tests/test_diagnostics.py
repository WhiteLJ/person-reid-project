from __future__ import annotations

import unittest

from src.diagnostics import RuntimeDiagnostics
from src.models import Track


class DiagnosticsTests(unittest.TestCase):
    def test_track_lifecycle_is_reported_without_calling_fragmentation(self) -> None:
        diagnostics = RuntimeDiagnostics(enabled=True, log_interval_frames=10)
        diagnostics.observe_tracks([Track(3, (0, 0, 10, 20), 0.9, 0)], 0)
        diagnostics.record_frame(0.1, 0)
        diagnostics.observe_tracks([Track(8, (0, 0, 10, 20), 0.9, 0)], 1)
        diagnostics.record_frame(0.1, 1)

        self.assertEqual(diagnostics.processed_frames, 2)
        self.assertEqual(diagnostics.track_created_count, 2)
        self.assertEqual(diagnostics.track_ended_count, 1)
        self.assertEqual(diagnostics.unique_track_ids, {3, 8})
        self.assertIn("average_fps=10.00", diagnostics.summary())

    def test_disabled_diagnostics_do_not_collect(self) -> None:
        diagnostics = RuntimeDiagnostics(enabled=False)
        diagnostics.observe_tracks([Track(3, (0, 0, 10, 20), 0.9, 0)], 0)
        diagnostics.record_frame(0.1, 0)
        self.assertEqual(diagnostics.processed_frames, 0)
        self.assertEqual(diagnostics.unique_track_ids, set())


if __name__ == "__main__":
    unittest.main()
