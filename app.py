"""MVP-3 entry point: multi-target ROI selection over BoT-SORT Tracks."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from src.config import AppConfig, load_config, parse_source
from src.logging_utils import configure_logging
from src.roi_selector import find_track_by_roi
from src.target_manager import TargetManager
from src.tracking_pipeline import TrackingPipeline
from src.video_source import VideoSource
from src.visualization import draw_tracks
from ui.opencv_ui import OpenCVUI, UIAction


LOGGER = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MVP-3 YOLOv8n person tracking with multi-target ROI selection"
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

    source = VideoSource(config.video.source)
    ui = OpenCVUI(config.ui)
    target_manager = TargetManager()
    try:
        source.open()
        LOGGER.info("SOURCE_OPENED source=%s", config.video.source)
        while True:
            frame = source.read()
            if frame is None:
                LOGGER.info("SOURCE_END source=%s", config.video.source)
                break

            tracks = tracking_pipeline.process(frame)
            annotated = draw_tracks(
                frame,
                tracks,
                class_name=tracking_pipeline.class_name,
                show_class_name=config.ui.show_class_name,
                show_track_id=config.tracking.show_track_id,
                show_confidence=config.ui.show_confidence,
                selected_track_ids=target_manager.selected_track_ids,
            )
            action = ui.show(annotated)
            if action == UIAction.QUIT:
                LOGGER.info("USER_QUIT key=q")
                break
            if action == UIAction.SELECT_TARGET:
                roi = ui.select_roi(frame)
                if roi is None:
                    LOGGER.info("TARGET_SELECTION_CANCELLED")
                    continue
                track = find_track_by_roi(
                    roi,
                    tracks,
                    min_iou=config.selection.min_iou,
                )
                if track is None:
                    LOGGER.info(
                        "TARGET_SELECTION_FAILED roi=%s min_iou=%.3f",
                        roi,
                        config.selection.min_iou,
                    )
                else:
                    already_selected = target_manager.is_selected(track.track_id)
                    target_manager.select(track)
                    LOGGER.info(
                        "TARGET_SELECTED track=%d already_selected=%s",
                        track.track_id,
                        already_selected,
                    )
            elif action == UIAction.REMOVE_TARGET:
                roi = ui.select_roi(frame)
                if roi is None:
                    LOGGER.info("TARGET_REMOVAL_CANCELLED")
                    continue
                track = find_track_by_roi(
                    roi,
                    tracks,
                    min_iou=config.selection.min_iou,
                )
                if track is None:
                    LOGGER.info(
                        "TARGET_REMOVAL_FAILED roi=%s min_iou=%.3f",
                        roi,
                        config.selection.min_iou,
                    )
                elif not target_manager.is_selected(track.track_id):
                    LOGGER.info(
                        "TARGET_REMOVAL_IGNORED track=%d reason=not_selected",
                        track.track_id,
                    )
                else:
                    target_manager.deselect(track)
                    LOGGER.info("TARGET_REMOVED track=%d", track.track_id)
            elif action == UIAction.CLEAR_TARGETS:
                selected_count = len(target_manager.selected_track_ids)
                target_manager.clear()
                LOGGER.info("TARGETS_CLEARED count=%d", selected_count)
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
