"""Mouse-driven paused ROI editing sessions for MVP-3.1."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum, auto
from logging import getLogger

import cv2
import numpy as np

from src.models import Track


LOGGER = getLogger(__name__)
MIN_ROI_SIZE = 2


class UIAction(Enum):
    NONE = auto()
    QUIT = auto()
    SELECT_TARGET = auto()
    REMOVE_TARGET = auto()
    CLEAR_TARGETS = auto()


class EditMode(Enum):
    LIVE = auto()
    ADD_TARGETS = auto()
    REMOVE_TARGETS = auto()


def normalize_roi_xyxy(
    start: Sequence[float],
    end: Sequence[float],
    frame_shape: Sequence[int],
    min_size: int = MIN_ROI_SIZE,
) -> tuple[int, int, int, int] | None:
    """Normalize, clip, and validate a mouse drag as an ``xyxy`` ROI."""

    if len(start) != 2 or len(end) != 2:
        raise ValueError("ROI drag points must contain exactly two coordinates")
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    if min_size < 1:
        raise ValueError("min_size must be positive")

    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("frame dimensions must be positive")

    raw_x1, raw_x2 = sorted((int(round(float(start[0]))), int(round(float(end[0])))))
    raw_y1, raw_y2 = sorted((int(round(float(start[1]))), int(round(float(end[1])))))
    x1 = max(0, min(width, raw_x1))
    y1 = max(0, min(height, raw_y1))
    x2 = max(0, min(width, raw_x2))
    y2 = max(0, min(height, raw_y2))

    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return x1, y1, x2, y2


def roi_xyxy_to_xywh(roi: Sequence[int]) -> tuple[int, int, int, int]:
    """Convert a valid normalized ``xyxy`` ROI to OpenCV ``xywh`` format."""

    if len(roi) != 4:
        raise ValueError("ROI must contain exactly four values")
    x1, y1, x2, y2 = (int(value) for value in roi)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI must have positive area")
    return x1, y1, x2 - x1, y2 - y1


class ROIEditSession:
    """Run one blocking multi-ROI edit session on the existing main window."""

    def __init__(
        self,
        window_name: str,
        frame: np.ndarray,
        tracks: Sequence[Track],
        mode: EditMode,
        wait_key_ms: int,
        on_roi: Callable[[tuple[int, int, int, int], tuple[Track, ...], EditMode], None],
        render_frame: Callable[[np.ndarray, tuple[Track, ...]], np.ndarray],
        min_roi_size: int = MIN_ROI_SIZE,
    ) -> None:
        self.window_name = window_name
        self.frozen_frame = frame.copy()
        self.frozen_tracks = tuple(tracks)
        self.mode = mode
        self.wait_key_ms = max(1, int(wait_key_ms))
        self.on_roi = on_roi
        self.render_frame = render_frame
        self.min_roi_size = min_roi_size

        self._dragging = False
        self._drag_start: tuple[int, int] | None = None
        self._drag_current: tuple[int, int] | None = None
        self._preview_roi: tuple[int, int, int, int] | None = None

    def run(self) -> UIAction:
        """Run until Enter/Space, Q, or an exception, always clearing the callback."""

        callback_registered = False
        try:
            cv2.setMouseCallback(self.window_name, self._on_mouse)
            callback_registered = True
            self._redraw()

            while True:
                key = cv2.waitKey(self.wait_key_ms) & 0xFF
                if key in (13, 32):  # Enter / Space
                    LOGGER.info("ROI_EDIT_FINISHED mode=%s", self.mode.name)
                    return UIAction.NONE
                if key == 27:  # Esc: cancel only the unfinished drag
                    self._cancel_drag()
                    self._redraw()
                    LOGGER.info("ROI_DRAG_CANCELLED mode=%s", self.mode.name)
                    continue
                if key in (ord("q"), ord("Q")):
                    LOGGER.info("ROI_EDIT_QUIT mode=%s", self.mode.name)
                    return UIAction.QUIT
        finally:
            try:
                if callback_registered:
                    self._clear_mouse_callback()
            finally:
                self._reset_mouse_state()

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        point = (int(x), int(y))
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._drag_start = point
            self._drag_current = point
            self._preview_roi = None
            self._redraw()
            return

        if event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self._drag_current = point
            self._preview_roi = self._normalized_drag()
            self._redraw()
            return

        if event == cv2.EVENT_LBUTTONUP and self._dragging:
            self._drag_current = point
            normalized_roi = self._normalized_drag()
            self._dragging = False
            self._drag_start = None
            self._drag_current = None
            self._preview_roi = None
            if normalized_roi is None:
                LOGGER.info(
                    "ROI_IGNORED mode=%s reason=zero_or_too_small_or_outside",
                    self.mode.name,
                )
            else:
                self.on_roi(
                    roi_xyxy_to_xywh(normalized_roi),
                    self.frozen_tracks,
                    self.mode,
                )
            self._redraw()

    def _normalized_drag(self) -> tuple[int, int, int, int] | None:
        if self._drag_start is None or self._drag_current is None:
            return None
        return normalize_roi_xyxy(
            self._drag_start,
            self._drag_current,
            self.frozen_frame.shape,
            min_size=self.min_roi_size,
        )

    def _redraw(self) -> None:
        annotated = self.render_frame(self.frozen_frame, self.frozen_tracks)
        if self._preview_roi is not None:
            x1, y1, x2, y2 = self._preview_roi
            cv2.rectangle(
                annotated,
                (x1, y1),
                (max(x1, x2 - 1), max(y1, y2 - 1)),
                (255, 255, 0),
                2,
            )
        cv2.imshow(self.window_name, annotated)

    def _cancel_drag(self) -> None:
        self._dragging = False
        self._drag_start = None
        self._drag_current = None
        self._preview_roi = None

    def _reset_mouse_state(self) -> None:
        self._cancel_drag()

    def _clear_mouse_callback(self) -> None:
        try:
            cv2.setMouseCallback(self.window_name, None)
        except (cv2.error, TypeError):
            # Some OpenCV builds require a callable; a stateless no-op is the
            # compatibility fallback and retains no edit-session state.
            cv2.setMouseCallback(self.window_name, self._noop_mouse)

    @staticmethod
    def _noop_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del event, x, y, flags, param
