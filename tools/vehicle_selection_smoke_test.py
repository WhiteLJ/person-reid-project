"""PC3 Vehicle manual selection smoke test."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from src.config import load_config, parse_source
from src.models import Track
from src.pc_multiclass_tracking import MultiClassTrackingPipeline
from src.target_manager import TargetManager
from src.vehicle_reid import VehicleReIDExtractor
from src.vehicle_selection import VehicleSelectionController
from src.video_source import VideoSource
from src.display_transform import DisplayTransform
from src.visualization import VEHICLE_COLOR, VEHICLE_SELECTED_COLOR
from tools.multiclass_tracking_smoke_test import _draw_track
from ui.roi_editor import EditMode, ROIEditSession, UIAction


LOGGER = logging.getLogger(__name__)
WINDOW_NAME = "MVP-8.3-PC3 Vehicle Selection"


def _draw_vehicle_track(
    frame: np.ndarray,
    track: Track,
    target_manager: TargetManager,
    *,
    show_unselected_tracks: bool,
    gallery_labels_by_target: Mapping[int, str] | None = None,
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
    if not selected and not show_unselected_tracks:
        return
    color = VEHICLE_SELECTED_COLOR if selected else VEHICLE_COLOR
    thickness = 4 if selected else 2
    if target is not None:
        gallery_label = (
            gallery_labels_by_target or {}
        ).get(target.target_id)
        prefix = f"{gallery_label} / " if gallery_label else ""
        label = (
            f"{prefix}VT-{target.target_id} / V-T{track.track_id} "
            f"{track.confidence:.2f}"
        )
    else:
        label = f"Vehicle V-T{track.track_id} {track.confidence:.2f}"
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
    *,
    show_unselected_tracks: bool,
    gallery_labels_by_target: Mapping[int, str] | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    if show_unselected_tracks:
        for track in person_tracks:
            _draw_track(
                annotated,
                track,
                prefix="Person P-T",
                color=(0, 200, 0),
            )
    for track in vehicle_tracks:
        _draw_vehicle_track(
            annotated,
            track,
            target_manager,
            show_unselected_tracks=show_unselected_tracks,
            gallery_labels_by_target=gallery_labels_by_target,
        )
    return annotated


def _run_vehicle_edit_session(
    frame: np.ndarray,
    person_tracks: Sequence[Track],
    vehicle_tracks: Sequence[Track],
    target_manager: TargetManager,
    controller: VehicleSelectionController,
    frame_index: int,
    *,
    remove: bool,
    wait_key_ms: int,
    max_display_width: int | None,
    show_unselected_tracks: bool,
    select_handler: Callable[..., object] | None = None,
    gallery_labels_by_target: Mapping[int, str] | None = None,
    window_name: str = WINDOW_NAME,
) -> UIAction:
    """Run a frozen multi-ROI Vehicle add/remove edit session."""

    frozen_frame = frame.copy()
    frozen_person_tracks = tuple(person_tracks)
    frozen_vehicle_tracks = tuple(vehicle_tracks)
    selectable_tracks = (
        target_manager.selected_tracks(list(frozen_vehicle_tracks))
        if remove
        else list(frozen_vehicle_tracks)
    )
    display_transform = DisplayTransform.from_frame(
        frozen_frame,
        max_display_width,
    )

    def render_frame(
        source_frame: np.ndarray,
        ignored_tracks: tuple[Track, ...],
    ) -> np.ndarray:
        del ignored_tracks
        return _render_frame(
            source_frame,
            frozen_person_tracks,
            frozen_vehicle_tracks,
            target_manager,
            show_unselected_tracks=show_unselected_tracks,
            gallery_labels_by_target=gallery_labels_by_target,
        )

    def on_roi(
        roi: tuple[int, int, int, int],
        ignored_tracks: tuple[Track, ...],
        mode: EditMode,
    ) -> None:
        del ignored_tracks
        if mode is EditMode.REMOVE_TARGETS:
            controller.remove_from_roi(selectable_tracks, roi)
        elif select_handler is not None:
            select_handler(
                frozen_frame,
                frozen_vehicle_tracks,
                roi,
                frame_index,
            )
        else:
            controller.select_from_roi(
                frozen_frame,
                frozen_vehicle_tracks,
                roi,
                frame_index,
            )

    session = ROIEditSession(
        window_name=window_name,
        frame=frozen_frame,
        tracks=tuple(selectable_tracks),
        mode=EditMode.REMOVE_TARGETS if remove else EditMode.ADD_TARGETS,
        wait_key_ms=wait_key_ms,
        on_roi=on_roi,
        render_frame=render_frame,
        display_transform=display_transform,
    )
    return session.run()


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
                display_transform = DisplayTransform.from_frame(
                    frame,
                    config.ui.max_display_width,
                )
                display = display_transform.source_to_display(
                    _render_frame(
                        frame,
                        output.person_tracks,
                        output.vehicle_tracks,
                        vehicle_target_manager,
                        show_unselected_tracks=config.ui.show_unselected_tracks,
                    )
                )
                cv2.imshow(WINDOW_NAME, display)
                frame_count += 1
                key = cv2.waitKey(config.ui.wait_key_ms) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")):
                    edit_action = _run_vehicle_edit_session(
                        frame,
                        output.person_tracks,
                        output.vehicle_tracks,
                        vehicle_target_manager,
                        selection_controller,
                        frame_count - 1,
                        remove=False,
                        wait_key_ms=config.ui.wait_key_ms,
                        max_display_width=config.ui.max_display_width,
                        show_unselected_tracks=config.ui.show_unselected_tracks,
                    )
                    if edit_action is UIAction.QUIT:
                        break
                elif key in (ord("r"), ord("R")):
                    edit_action = _run_vehicle_edit_session(
                        frame,
                        output.person_tracks,
                        output.vehicle_tracks,
                        vehicle_target_manager,
                        selection_controller,
                        frame_count - 1,
                        remove=True,
                        wait_key_ms=config.ui.wait_key_ms,
                        max_display_width=config.ui.max_display_width,
                        show_unselected_tracks=config.ui.show_unselected_tracks,
                    )
                    if edit_action is UIAction.QUIT:
                        break
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
