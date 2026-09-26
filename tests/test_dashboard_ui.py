from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.config import UIConfig
from ui.dashboard_models import DashboardState
from ui.dashboard_ui import DashboardButton, DashboardUI, button_at
from ui.roi_editor import EditMode, UIAction


def _ui_config() -> UIConfig:
    return UIConfig(
        window_name="dashboard-test",
        wait_key_ms=1,
        show_class_name=True,
        show_confidence=True,
        show_unselected_tracks=False,
        max_display_width=1280,
        sidebar_width=320,
        dashboard_enabled=True,
    )


class DashboardButtonTests(unittest.TestCase):
    def test_all_enabled_button_centers_return_their_actions(self) -> None:
        ui = DashboardUI(_ui_config())
        buttons = ui.build_buttons(DashboardState(), 1280, 720)

        for button in buttons[:-1]:
            x1, y1, x2, y2 = button.rect
            self.assertEqual(
                button_at(buttons, (x1 + x2) // 2, (y1 + y2) // 2),
                button.action,
            )
        reserved = buttons[-1]
        x1, y1, x2, y2 = reserved.rect
        self.assertEqual(
            button_at(buttons, (x1 + x2) // 2, (y1 + y2) // 2),
            UIAction.NONE,
        )

    def test_button_hit_test_and_disabled_reserved_button(self) -> None:
        buttons = (
            DashboardButton("Select", UIAction.SELECT_TARGET, (100, 10, 180, 40)),
            DashboardButton("Reserved", UIAction.NONE, (100, 50, 180, 80), False),
        )
        self.assertEqual(button_at(buttons, 120, 20), UIAction.SELECT_TARGET)
        self.assertEqual(button_at(buttons, 120, 60), UIAction.NONE)
        self.assertEqual(button_at(buttons, 90, 20), UIAction.NONE)


class DashboardLayoutTests(unittest.TestCase):
    def test_dashboard_keeps_video_left_and_adds_sidebar(self) -> None:
        ui = DashboardUI(_ui_config())
        source = np.zeros((1080, 1920, 3), dtype=np.uint8)
        source[:, :10] = (1, 2, 3)

        dashboard = ui.compose(source, DashboardState(source_label="test.mp4"))

        self.assertEqual(dashboard.shape[:2], (720, 1600))
        self.assertTrue(np.array_equal(dashboard[0, 0], (1, 2, 3)))
        self.assertTrue(np.array_equal(dashboard[0, 1280], (32, 36, 42)))
        self.assertEqual(len(ui._buttons), 7)

    @patch("ui.dashboard_ui.cv2.setMouseCallback")
    @patch("ui.dashboard_ui.cv2.imshow")
    @patch("ui.dashboard_ui.cv2.waitKey", return_value=13)
    @patch("ui.dashboard_ui.cv2.namedWindow")
    def test_edit_session_restores_dashboard_mouse_callback(
        self,
        named_window,
        wait_key,
        imshow,
        set_mouse_callback,
    ) -> None:
        del named_window, wait_key, imshow
        ui = DashboardUI(_ui_config())
        frame = np.zeros((80, 120, 3), dtype=np.uint8)

        result = ui.run_edit_session(
            frame=frame,
            tracks=(),
            mode=EditMode.ADD_TARGETS,
            on_roi=lambda roi, tracks, mode: None,
            render_frame=lambda frozen, tracks: frozen.copy(),
        )

        self.assertEqual(result, UIAction.NONE)
        self.assertGreaterEqual(set_mouse_callback.call_count, 3)
        self.assertIsNotNone(set_mouse_callback.call_args_list[-1].args[1])


if __name__ == "__main__":
    unittest.main()
