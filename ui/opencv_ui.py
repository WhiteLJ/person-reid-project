"""Backward-compatible formal OpenCV UI façade backed by the Dashboard."""

from __future__ import annotations

import cv2  # Compatibility export for existing UI tests/tools.

from ui.dashboard_ui import DashboardUI, key_to_action
from ui.roi_editor import EditMode, UIAction


class OpenCVUI(DashboardUI):
    """Keep the existing app/tools import while using one Dashboard UI."""


__all__ = ["DashboardUI", "EditMode", "OpenCVUI", "UIAction", "key_to_action"]
