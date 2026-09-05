"""Minimal OpenCV display and keyboard handling for MVP-3."""

from __future__ import annotations

from enum import Enum, auto

import cv2
import numpy as np

from src.config import UIConfig


class UIAction(Enum):
    NONE = auto()
    QUIT = auto()
    SELECT_TARGET = auto()
    REMOVE_TARGET = auto()
    CLEAR_TARGETS = auto()


def key_to_action(key: int) -> UIAction:
    """Map one OpenCV keyboard value to a UI action."""

    normalized_key = int(key) & 0xFF
    if normalized_key in (ord("q"), ord("Q")):
        return UIAction.QUIT
    if normalized_key in (ord("s"), ord("S")):
        return UIAction.SELECT_TARGET
    if normalized_key in (ord("r"), ord("R")):
        return UIAction.REMOVE_TARGET
    if normalized_key in (ord("c"), ord("C")):
        return UIAction.CLEAR_TARGETS
    return UIAction.NONE


class OpenCVUI:
    """Display frames and handle MVP-3 keyboard/ROI interactions."""

    def __init__(self, config: UIConfig) -> None:
        self.config = config
        self.roi_window_name = f"{config.window_name} - Select Target"

    def show(self, frame: np.ndarray) -> UIAction:
        cv2.imshow(self.config.window_name, frame)
        key = cv2.waitKey(self.config.wait_key_ms) & 0xFF
        return key_to_action(key)

    def select_roi(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Block on an independent OpenCV ROI window until selection is confirmed."""

        roi = cv2.selectROI(
            self.roi_window_name,
            frame,
            showCrosshair=True,
            fromCenter=False,
        )
        if roi is None:
            return None

        x, y, width, height = (int(value) for value in roi)
        if width <= 0 or height <= 0:
            return None
        return x, y, width, height

    @staticmethod
    def close() -> None:
        cv2.destroyAllWindows()
