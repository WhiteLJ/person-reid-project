"""MVP-5 entry point: BoT-SORT tracks with in-session ReID recovery."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from src.config import AppConfig, load_config, parse_source
from src.logging_utils import configure_logging
from src.reid import ReIDExtractor
from src.roi_selector import find_track_by_roi
from src.target_manager import TargetManager
from src.target_recovery import TargetRecoveryCoordinator
from src.tracking_pipeline import TrackingPipeline
from src.video_source import VideoSource
from src.visualization import draw_tracks
from ui.opencv_ui import EditMode, OpenCVUI, UIAction


LOGGER = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MVP-5 YOLOv8n/BoT-SORT tracking with session-target ReID recovery"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="path to the YAML configuration file",
    )
    parser.add_argument(
        "--source",
        help="optional camera index or local video path overriding config",
    )
    return parser


def _override_source(config: AppConfig, source: str | None) -> AppConfig:
    if source is None:
        return config
    return replace(config, video=replace(config.video, source=parse_source(source)))


def run(config: AppConfig) -> int:
    LOGGER.info(
        "APP_START device=%s workers=%d",
        config.model.device,
        config.runtime.num_workers,
    )
    tracking_pipeline = TrackingPipeline(
        model_config=config.model,
        runtime_config=config.runtime,
        tracking_config=config.tracking,
    )
    LOGGER.info(
        "MODEL_LOADED path=%s tracker=%s persist=%s",
        config.model.yolo_weight,
        config.tracking.tracker,
        config.tracking.persist,
    )
    reid_extractor = ReIDExtractor(config.reid, config.model.device)
    LOGGER.info(
        "REID_MODEL_LOADED name=%s checkpoint=%s device=%s",
        config.reid.model_name,
        config.reid.weight,
        reid_extractor.device,
    )

    source = VideoSource(config.video.source)
    ui = OpenCVUI(config.ui)
    target_manager = TargetManager()
    target_recovery = TargetRecoveryCoordinator(
        target_manager=target_manager,
        reid_extractor=reid_extractor,
        reid_config=config.reid,
        recovery_config=config.reid_recovery,
    )
    frame_index = 0
    current_frame_index = -1

    def render_tracks(render_frame, render_tracks):
        return draw_tracks(
            render_frame,
            render_tracks,
            class_name=tracking_pipeline.class_name,
            show_class_name=config.ui.show_class_name,
            show_track_id=config.tracking.show_track_id,
            show_confidence=config.ui.show_confidence,
            selected_track_ids=target_manager.selected_track_ids,
            show_unselected_tracks=config.ui.show_unselected_tracks,
        )

    def handle_roi(roi, frozen_tracks, mode):
        track = find_track_by_roi(
            roi,
            frozen_tracks,
            min_iou=config.selection.min_iou,
        )
        if track is None:
            LOGGER.info(
                "TARGET_ROI_FAILED mode=%s roi=%s min_iou=%.3f",
                mode.name,
                roi,
                config.selection.min_iou,
            )
            return

        if mode == EditMode.ADD_TARGETS:
            target_recovery.select_from_track(
                frame,
                track,
                current_frame_index,
            )
            return

        if mode == EditMode.REMOVE_TARGETS:
            target = target_manager.target_for_track(track.track_id)
            if target is None:
                LOGGER.info(
                    "TARGET_REMOVAL_IGNORED track=%d reason=not_selected",
                    track.track_id,
                )
            else:
                target_manager.deselect(track)
                LOGGER.info(
                    "TARGET_REMOVED target=%d track=%d",
                    target.target_id,
                    track.track_id,
                )

    try:
        source.open()
        LOGGER.info("SOURCE_OPENED source=%s", config.video.source)
        while True:
            frame = source.read()
            if frame is None:
                LOGGER.info("SOURCE_END source=%s", config.video.source)
                break

            current_frame_index = frame_index
            tracks = tracking_pipeline.process(frame)
            target_recovery.process_frame(frame, tracks, current_frame_index)
            frame_index += 1
            annotated = render_tracks(frame, tracks)
            action = ui.show(annotated)
            if action == UIAction.QUIT:
                LOGGER.info("USER_QUIT key=q")
                break
            if action == UIAction.CLEAR_TARGETS:
                selected_count = len(target_manager.targets)
                target_manager.clear()
                LOGGER.info("TARGETS_CLEARED count=%d", selected_count)
                continue

            if action in (UIAction.SELECT_TARGET, UIAction.REMOVE_TARGET):
                edit_mode = (
                    EditMode.ADD_TARGETS
                    if action == UIAction.SELECT_TARGET
                    else EditMode.REMOVE_TARGETS
                )
                edit_action = ui.run_edit_session(
                    frame=frame,
                    tracks=tracks,
                    mode=edit_mode,
                    on_roi=handle_roi,
                    render_frame=render_tracks,
                )
                if edit_action == UIAction.QUIT:
                    LOGGER.info("USER_QUIT key=q edit_mode=%s", edit_mode.name)
                    break
    finally:
        source.release()
        ui.close()
        LOGGER.info("APP_STOP")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    configure_logging()
    try:
        config = load_config(Path(args.config))
        config = _override_source(config, args.source)
        configure_logging(config.runtime.log_level)
        return run(config)
    except KeyboardInterrupt:
        LOGGER.info("APP_STOP interrupted")
        return 0
    except Exception:
        LOGGER.exception("APPLICATION_ERROR")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
