from __future__ import annotations

import unittest

from src.models import Track
from src.target_manager import TargetManager


class TargetManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track_a = Track(7, (0, 0, 20, 40), 0.9, 0)
        self.track_b = Track(8, (30, 0, 50, 40), 0.9, 0)
        self.track_c = Track(9, (60, 0, 80, 40), 0.9, 0)

    def test_selects_multiple_tracks(self) -> None:
        manager = TargetManager()

        manager.select(self.track_a)
        manager.select(self.track_b)

        self.assertEqual(manager.selected_track_ids, {7, 8})
        self.assertEqual(manager.selected_tracks([self.track_a, self.track_b]), [self.track_a, self.track_b])

    def test_selecting_same_track_is_idempotent(self) -> None:
        manager = TargetManager()

        manager.select(self.track_a)
        manager.select(self.track_a)

        self.assertEqual(manager.selected_track_ids, {7})

    def test_deselecting_one_track_preserves_other_targets(self) -> None:
        manager = TargetManager()
        manager.select(self.track_a)
        manager.select(self.track_b)

        manager.deselect(self.track_a)

        self.assertEqual(manager.selected_track_ids, {8})

    def test_deselecting_non_selected_track_changes_nothing(self) -> None:
        manager = TargetManager()
        manager.select(self.track_a)

        manager.deselect(self.track_c)

        self.assertEqual(manager.selected_track_ids, {7})

    def test_clear_removes_all_targets(self) -> None:
        manager = TargetManager()
        manager.select(self.track_a)
        manager.select(self.track_b)

        manager.clear()

        self.assertEqual(manager.selected_track_ids, set())

    def test_missing_selected_track_is_not_rebound(self) -> None:
        manager = TargetManager()
        manager.select(self.track_a)

        visible_tracks = manager.selected_tracks([self.track_b])

        self.assertEqual(visible_tracks, [])
        self.assertEqual(manager.selected_track_ids, {7})


if __name__ == "__main__":
    unittest.main()
