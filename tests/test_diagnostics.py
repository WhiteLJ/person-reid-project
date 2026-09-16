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

    def test_recovery_timing_and_workload_are_reported_separately(self) -> None:
        diagnostics = RuntimeDiagnostics(enabled=True, log_interval_frames=100)
        diagnostics.record_frame(
            0.050,
            0,
            tracking_seconds=0.010,
            recovery_seconds=0.020,
            gallery_seconds=0.005,
            render_seconds=0.003,
            ui_seconds=0.002,
            recovery_due=True,
            recovery_candidate_count=12,
            recovery_quality_valid_count=8,
            recovery_reid_batch_count=2,
            recovery_reid_seconds=0.015,
        )
        diagnostics.record_frame(
            0.010,
            1,
            tracking_seconds=0.006,
            recovery_seconds=0.001,
            gallery_seconds=0.002,
        )

        summary = diagnostics.summary()
        self.assertIn("frame_total_ms=30.00", summary)
        self.assertIn("tracking_ms=8.00", summary)
        self.assertIn("recovery_ms=10.50", summary)
        self.assertIn("normal_frame_ms=10.00", summary)
        self.assertIn("recovery_frame_ms=50.00", summary)
        self.assertIn("recovery_frame_max_ms=50.00", summary)
        self.assertIn("recovery_candidate_count=12", summary)
        self.assertIn("recovery_quality_valid_count=8", summary)
        self.assertIn("recovery_reid_batch_count=2", summary)
        self.assertIn("recovery_reid_ms=15.00", summary)


if __name__ == "__main__":
    unittest.main()
