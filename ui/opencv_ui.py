"""OpenCV display, actions, and ROI editing entry points."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np

from src.config import UIConfig
from src.display_transform import DisplayTransform
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
        self._window_created = False

    def _ensure_window(self) -> None:
        if self._window_created:
            return
        try:
            cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
        except cv2.error:
            # Keep the display API usable in headless/unit-test OpenCV builds;
            # imshow will still provide the same behavior where a GUI exists.
            pass
        self._window_created = True

    def show(self, frame: np.ndarray) -> UIAction:
        self._ensure_window()
        transform = DisplayTransform.from_frame(frame, self.config.max_display_width)
        cv2.imshow(self.config.window_name, transform.source_to_display(frame))
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

        self._ensure_window()
        transform = DisplayTransform.from_frame(frame, self.config.max_display_width)
        session = ROIEditSession(
            window_name=self.config.window_name,
            frame=frame,
            tracks=tracks,
            mode=mode,
            wait_key_ms=self.config.wait_key_ms,
            on_roi=on_roi,
            render_frame=render_frame,
            display_transform=transform,
        )
        return session.run()

    @staticmethod
    def close() -> None:
        cv2.destroyAllWindows()
