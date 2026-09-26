"""Single-window OpenCV Dashboard and button/event handling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from logging import getLogger

import cv2
import numpy as np

from src.config import UIConfig
from src.display_transform import DisplayTransform
from src.models import Track
from ui.dashboard_models import DashboardState
from ui.roi_editor import EditMode, ROIEditSession, UIAction


LOGGER = getLogger(__name__)


def key_to_action(key: int) -> UIAction:
    """Map one OpenCV keyboard value to a UI action."""

    normalized_key = int(key) & 0xFF
    if normalized_key in (ord("q"), ord("Q")):
        return UIAction.QUIT
    if normalized_key in (ord("p"), ord("P")):
        return UIAction.PAUSE_TOGGLE
    if normalized_key in (ord("s"), ord("S")):
        return UIAction.SELECT_TARGET
    if normalized_key in (ord("r"), ord("R")):
        return UIAction.REMOVE_TARGET
    if normalized_key in (ord("c"), ord("C")):
        return UIAction.CLEAR_TARGETS
    if normalized_key in (ord("g"), ord("G")):
        return UIAction.ENROLL_GALLERY
    return UIAction.NONE


@dataclass(frozen=True)
class DashboardButton:
    """Pure button geometry and action mapping for headless testing."""

    label: str
    action: UIAction
    rect: tuple[int, int, int, int]
    enabled: bool = True

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= int(x) < x2 and y1 <= int(y) < y2


def button_at(
    buttons: Sequence[DashboardButton],
    x: int,
    y: int,
) -> UIAction:
    """Return the enabled action at a dashboard coordinate, or ``NONE``."""

    for button in buttons:
        if button.enabled and button.contains(x, y):
            return button.action
    return UIAction.NONE


class DashboardUI:
    """Render video plus sidebar and expose actions without business logic."""

    _PANEL_COLOR = (32, 36, 42)
    _PANEL_MUTED = (155, 165, 175)
    _PANEL_TEXT = (235, 240, 245)
    _ACCENT = (80, 190, 255)
    _BUTTON_COLOR = (58, 72, 86)
    _BUTTON_DISABLED = (55, 55, 55)

    def __init__(self, config: UIConfig) -> None:
        self.config = config
        self._window_created = False
        self._pending_action = UIAction.NONE
        self._buttons: tuple[DashboardButton, ...] = ()
        self._panel_left = 0

    def _ensure_window(self) -> None:
        if self._window_created:
            return
        try:
            cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
        except cv2.error:
            # Keep headless/unit-test OpenCV builds usable.
            pass
        self._window_created = True

    def build_buttons(
        self,
        state: DashboardState,
        video_width: int,
        video_height: int,
    ) -> tuple[DashboardButton, ...]:
        del video_height
        panel_left = int(video_width)
        x1 = panel_left + 20
        x2 = panel_left + self.config.sidebar_width - 20
        y = 335
        height = 32
        gap = 9
        button_specs = (
            ("Select Target", UIAction.SELECT_TARGET, True),
            ("Remove Target", UIAction.REMOVE_TARGET, True),
            ("Enroll Gallery", UIAction.ENROLL_GALLERY, True),
            ("Clear Session", UIAction.CLEAR_TARGETS, True),
            ("Resume" if state.paused else "Pause", UIAction.PAUSE_TOGGLE, True),
            ("Quit", UIAction.QUIT, True),
            ("Camera / RTSP  Reserved", UIAction.NONE, False),
        )
        buttons: list[DashboardButton] = []
        for label, action, enabled in button_specs:
            buttons.append(
                DashboardButton(label, action, (x1, y, x2, y + height), enabled)
            )
            y += height + gap
        return tuple(buttons)

    def compose(self, frame: np.ndarray, state: DashboardState) -> np.ndarray:
        """Compose a display-only dashboard copy from a source-sized frame."""

        transform = DisplayTransform.from_frame(
            frame,
            self.config.max_display_width,
        )
        video = transform.source_to_display(frame)
        if not self.config.dashboard_enabled:
            self._buttons = ()
            self._panel_left = video.shape[1]
            return video

        display_height, display_width = video.shape[:2]
        sidebar_width = self.config.sidebar_width
        dashboard = np.zeros(
            (display_height, display_width + sidebar_width, 3),
            dtype=video.dtype,
        )
        dashboard[:, :display_width] = video
        panel = dashboard[:, display_width:]
        panel[:] = self._PANEL_COLOR
        self._panel_left = display_width
        self._buttons = self.build_buttons(state, display_width, display_height)
        self._draw_sidebar(panel, state)
        return dashboard

    def _draw_sidebar(self, panel: np.ndarray, state: DashboardState) -> None:
        self._text(panel, "PERSON + VEHICLE", 18, 30, self._ACCENT, 0.65, 2)
        self._text(panel, "TRACKING SYSTEM", 18, 55, self._ACCENT, 0.55, 1)
        self._text(panel, "SYSTEM", 18, 88, self._PANEL_MUTED, 0.48, 1)
        self._text(panel, f"Backend   {state.backend.upper()}", 18, 110)
        self._text(panel, f"Source    {state.source_label}", 18, 132)
        fps_text = "PAUSED" if state.paused else f"{state.fps:.1f}"
        self._text(panel, f"FPS       {fps_text}", 18, 154)
        self._text(panel, f"Frame     {max(0, state.frame_index)}", 18, 176)
        self._text(panel, f"Status    {state.mode}", 18, 198)

        self._text(panel, "TARGETS", 18, 226, self._PANEL_MUTED, 0.48, 1)
        self._text(panel, f"Person    A {state.person_active_count}  L {state.person_lost_count}", 18, 248)
        self._text(
            panel,
            f"Vehicle   A {state.vehicle_active_count}  L {state.vehicle_lost_count}",
            18,
            270,
        )
        self._text(panel, f"P Gallery {state.person_gallery_count}", 18, 292)
        self._text(panel, f"V Gallery {state.vehicle_gallery_count}", 18, 314)

        for button in self._buttons:
            x1, y1, x2, y2 = button.rect
            bx1 = x1 - self._panel_left
            bx2 = x2 - self._panel_left
            color = self._BUTTON_COLOR if button.enabled else self._BUTTON_DISABLED
            cv2.rectangle(panel, (bx1, y1), (bx2 - 1, y2 - 1), color, -1)
            text_color = self._PANEL_TEXT if button.enabled else self._PANEL_MUTED
            self._text(panel, button.label, bx1 + 9, y1 + 21, text_color, 0.43, 1)

        self._text(panel, "SHORTCUTS", 18, 635, self._PANEL_MUTED, 0.48, 1)
        self._text(panel, "S Select   R Remove   G Enroll", 18, 657, self._PANEL_TEXT, 0.39, 1)
        self._text(panel, "C Clear    P Pause    Q Quit", 18, 677, self._PANEL_TEXT, 0.39, 1)
        if state.status_message:
            self._text(panel, state.status_message[:34], 18, 705, self._ACCENT, 0.42, 1)

    @staticmethod
    def _text(
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int] = (235, 240, 245),
        scale: float = 0.45,
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            image,
            text,
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def show(
        self,
        frame: np.ndarray,
        state: DashboardState | None = None,
    ) -> UIAction:
        """Display one dashboard frame and return one pending UI action."""

        self._ensure_window()
        state = state or DashboardState()
        self._pending_action = UIAction.NONE
        display = self.compose(frame, state)
        self._register_dashboard_callback()
        cv2.imshow(self.config.window_name, display)
        key = cv2.waitKey(self.config.wait_key_ms) & 0xFF
        if self._pending_action is not UIAction.NONE:
            return self._pending_action
        action = key_to_action(key)
        if action is UIAction.NONE and self._window_is_closed():
            return UIAction.QUIT
        return action

    def _register_dashboard_callback(self) -> None:
        try:
            cv2.setMouseCallback(self.config.window_name, self._on_dashboard_mouse)
        except cv2.error:
            # Some headless/test builds do not expose a real HighGUI window.
            pass

    def _on_dashboard_mouse(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: object,
    ) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONUP:
            self._pending_action = button_at(self._buttons, x, y)

    def _window_is_closed(self) -> bool:
        try:
            return cv2.getWindowProperty(
                self.config.window_name,
                cv2.WND_PROP_VISIBLE,
            ) < 1
        except (cv2.error, AttributeError):
            return False

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
        """Run the existing frozen multi-ROI editor and restore Dashboard input."""

        self._ensure_window()
        transform = DisplayTransform.from_frame(frame, self.config.max_display_width)

        def render_with_prompt(
            source_frame: np.ndarray,
            frozen_tracks: tuple[Track, ...],
        ) -> np.ndarray:
            rendered = render_frame(source_frame, frozen_tracks)
            prompt = {
                EditMode.ADD_TARGETS: "SELECT TARGET | Drag ROI | Enter finish | Esc cancel",
                EditMode.REMOVE_TARGETS: "REMOVE TARGET | Drag ROI | Enter finish | Esc cancel",
                EditMode.ENROLL_GALLERY: "ENROLL GALLERY | Drag ROI | Enter finish | Esc cancel",
            }.get(mode, "EDIT | Drag ROI | Enter finish | Esc cancel")
            cv2.putText(
                rendered,
                prompt,
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            return rendered

        session = ROIEditSession(
            window_name=self.config.window_name,
            frame=frame,
            tracks=tracks,
            mode=mode,
            wait_key_ms=self.config.wait_key_ms,
            on_roi=on_roi,
            render_frame=render_with_prompt,
            display_transform=transform,
        )
        try:
            return session.run()
        finally:
            # ROIEditSession intentionally clears its callback; Dashboard owns
            # the live-mode callback and must restore it after every edit.
            self._register_dashboard_callback()

    def close(self) -> None:
        cv2.destroyAllWindows()
