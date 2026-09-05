"""Minimal OpenCV display and keyboard handling for MVP-1."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import UIConfig


class OpenCVUI:
    """Display annotated frames and report whether the user pressed Q."""

    def __init__(self, config: UIConfig) -> None:
        self.config = config

    def show(self, frame: np.ndarray) -> bool:
        cv2.imshow(self.config.window_name, frame)
        key = cv2.waitKey(self.config.wait_key_ms) & 0xFF
        return key in (ord("q"), ord("Q"))

    @staticmethod
    def close() -> None:
        cv2.destroyAllWindows()
