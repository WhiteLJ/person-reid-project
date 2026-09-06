"""OpenCV display, actions, and ROI editing entry points."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np

from src.config import UIConfig
from src.models import Track
from ui.roi_editor import EditMode, ROIEditSession, UIAction


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
    if normalized_key in (ord("g"), ord("G")):
        return UIAction.ENROLL_GALLERY
    return UIAction.NONE


class OpenCVUI:
    """Display frames and run the blocking ROI edit session."""

    def __init__(self, config: UIConfig) -> None:
        self.config = config

    def show(self, frame: np.ndarray) -> UIAction:
        cv2.imshow(self.config.window_name, frame)
        key = cv2.waitKey(self.config.wait_key_ms) & 0xFF
        return key_to_action(key)

    def run_edit_session(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        mode: EditMode,
        on_roi: Callable[
            [tuple[int, int, int, int], tuple[Track, ...], EditMode], None
        ],
        render_frame: Callable[[np.ndarray, tuple[Track, ...]], np.ndarray],
    ) -> UIAction:
        """Run a frozen-frame mouse session without reading or processing frames."""

        session = ROIEditSession(
            window_name=self.config.window_name,
            frame=frame,
            tracks=tracks,
            mode=mode,
            wait_key_ms=self.config.wait_key_ms,
            on_roi=on_roi,
            render_frame=render_frame,
        )
        return session.run()

    @staticmethod
    def close() -> None:
        cv2.destroyAllWindows()
