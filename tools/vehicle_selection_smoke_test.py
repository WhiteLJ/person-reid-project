"""PC3 Vehicle manual selection smoke test."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.config import load_config, parse_source
from src.models import Track
from src.pc_multiclass_tracking import MultiClassTrackingPipeline
from src.target_manager import TargetManager
from src.vehicle_reid import VehicleReIDExtractor
from src.vehicle_selection import VehicleSelectionController
from src.video_source import VideoSource
from tools.multiclass_tracking_smoke_test import (
    DEFAULT_MAX_DISPLAY_WIDTH,
    _draw_track,
    _fit_display_width,
)


LOGGER = logging.getLogger(__name__)
WINDOW_NAME = "MVP-8.3-PC3 Vehicle Selection"


def _display_roi_to_source(
    roi: Sequence[float],
    source_shape: Sequence[int],
    display_shape: Sequence[int],
) -> tuple[int, int, int, int] | None:
    """Map a ROI drawn on the scaled display back to the original frame."""

    if len(roi) != 4 or len(source_shape) < 2 or len(display_shape) < 2:
        raise ValueError("invalid ROI or frame shape")
    x, y, width, height = (float(value) for value in roi)
    if width <= 0 or height <= 0:
        return None
    source_height, source_width = int(source_shape[0]), int(source_shape[1])
    display_height, display_width = int(display_shape[0]), int(display_shape[1])
    if min(source_height, source_width, display_height, display_width) <= 0:
        raise ValueError("frame dimensions must be positive")

    scale_x = source_width / float(display_width)
    scale_y = source_height / float(display_height)
    x1 = max(0, min(source_width, int(round(x * scale_x))))
    y1 = max(0, min(source_height, int(round(y * scale_y))))
    x2 = max(0, min(source_width, int(round((x + width) * scale_x))))
    y2 = max(0, min(source_height, int(round((y + height) * scale_y))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _draw_vehicle_track(
    frame: np.ndarray,
    track: Track,
    target_manager: TargetManager,
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (
        max(0, min(width - 1, int(round(value))))
        for value in track.bbox
    )
    if x2 <= x1 or y2 <= y1:
        return

    target = target_manager.target_for_track(track.track_id)
    selected = target is not None
    color = (0, 0, 255) if selected else (0, 165, 255)
    thickness = 4 if selected else 2
    label = (
        f"VT-{target.target_id} / V-T{track.track_id} {track.confidence:.2f}"
        if target is not None
        else f"Vehicle V-T{track.track_id} {track.confidence:.2f}"
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _render_frame(
    frame: np.ndarray,
    person_tracks: Sequence[Track],
    vehicle_tracks: Sequence[Track],
    target_manager: TargetManager,
) -> np.ndarray:
    annotated = frame.copy()
    for track in person_tracks:
        _draw_track(
            annotated,
            track,
            prefix="Person P-T",
            color=(0, 200, 0),
        )
    for track in vehicle_tracks:
        _draw_vehicle_track(annotated, track, target_manager)
    return annotated


def _select_vehicle_roi(
    frame: np.ndarray,
    person_tracks: Sequence[Track],
    vehicle_tracks: Sequence[Track],
    target_manager: TargetManager,
    controller: VehicleSelectionController,
    frame_index: int,
    *,
    remove: bool,
    wait_key_ms: int,
) -> None:
    """Pause on the current frame, select one ROI, then resume playback."""

    frozen_frame = frame.copy()
    frozen_person_tracks = tuple(person_tracks)
    frozen_vehicle_tracks = tuple(vehicle_tracks)
    selectable_tracks = (
        target_manager.selected_tracks(list(frozen_vehicle_tracks))
        if remove
        else list(frozen_vehicle_tracks)
    )
    display_source = _render_frame(
        frozen_frame,
        frozen_person_tracks,
        frozen_vehicle_tracks,
        target_manager,
    )
    display = _fit_display_width(display_source, DEFAULT_MAX_DISPLAY_WIDTH)
    cv2.imshow(WINDOW_NAME, display)
    roi_on_display = cv2.selectROI(
        WINDOW_NAME,
        display,
        fromCenter=False,
        showCrosshair=True,
    )
    roi_on_source = _display_roi_to_source(
        roi_on_display,
        frozen_frame.shape,
        display.shape,
    )
    if roi_on_source is not None:
        if remove:
            controller.remove_from_roi(selectable_tracks, roi_on_source)
        else:
            controller.select_from_roi(
                frozen_frame,
                frozen_vehicle_tracks,
                roi_on_source,
                frame_index,
            )
    cv2.imshow(
        WINDOW_NAME,
        _fit_display_width(
            _render_frame(
                frozen_frame,
                frozen_person_tracks,
                frozen_vehicle_tracks,
                target_manager,
            ),
            DEFAULT_MAX_DISPLAY_WIDTH,
        ),
    )
    cv2.waitKey(max(1, int(wait_key_ms)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PC3 manual Vehicle SessionTarget selection smoke test"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--source",
        default=None,
        help="camera index or video path; defaults to config video.source",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    if config.inference.backend != "torch":
        raise ValueError("PC3 is PC-only and requires inference.backend=torch")
    if not config.vehicle_reid.enabled:
        raise ValueError("vehicle_reid.enabled must be true for PC3")

    source = config.video.source if args.source is None else parse_source(args.source)
    tracking_pipeline = MultiClassTrackingPipeline(config)
    vehicle_target_manager = TargetManager()
    vehicle_reid = VehicleReIDExtractor(config.vehicle_reid)
    selection_controller = VehicleSelectionController(
        vehicle_target_manager,
        vehicle_reid,
        min_iou=config.selection.min_iou,
    )
    frame_count = 0
    started = time.perf_counter()

    LOGGER.info("PC3_START source=%s vehicle_reid=%s", source, config.vehicle_reid.model_name)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        with VideoSource(source) as video:
            while True:
                frame = video.read()
                if frame is None:
                    break
                output = tracking_pipeline.process(frame)
                display = _fit_display_width(
                    _render_frame(
                        frame,
                        output.person_tracks,
                        output.vehicle_tracks,
                        vehicle_target_manager,
                    ),
                    DEFAULT_MAX_DISPLAY_WIDTH,
                )
                cv2.imshow(WINDOW_NAME, display)
                frame_count += 1
                key = cv2.waitKey(config.ui.wait_key_ms) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")):
                    _select_vehicle_roi(
                        frame,
                        output.person_tracks,
                        output.vehicle_tracks,
                        vehicle_target_manager,
                        selection_controller,
                        frame_count - 1,
                        remove=False,
                        wait_key_ms=config.ui.wait_key_ms,
                    )
                elif key in (ord("r"), ord("R")):
                    _select_vehicle_roi(
                        frame,
                        output.person_tracks,
                        output.vehicle_tracks,
                        vehicle_target_manager,
                        selection_controller,
                        frame_count - 1,
                        remove=True,
                        wait_key_ms=config.ui.wait_key_ms,
                    )
                elif key in (ord("c"), ord("C")):
                    count = len(vehicle_target_manager.targets)
                    vehicle_target_manager.clear()
                    LOGGER.info("VEHICLE_TARGETS_CLEARED count=%d", count)
    finally:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    stats = tracking_pipeline.stats()
    average_fps = frame_count / elapsed if elapsed > 0 else 0.0
    LOGGER.info(
        "PC3_STATS frames=%d average_fps=%.2f unique_person_tracks=%d "
        "unique_vehicle_tracks=%d yolo_inference_count=%d vehicle_targets=%d",
        frame_count,
        average_fps,
        stats.unique_person_tracks,
        stats.unique_vehicle_tracks,
        stats.yolo_inference_count,
        len(vehicle_target_manager.targets),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main())

