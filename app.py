"""MVP-1 entry point: YOLOv8n person detection with OpenCV display."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from src.config import AppConfig, load_config, parse_source
from src.detector import PersonDetector
from src.logging_utils import configure_logging
from src.video_source import VideoSource
from src.visualization import draw_detections
from ui.opencv_ui import OpenCVUI


LOGGER = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MVP-1 YOLOv8n person detection for camera or local video"
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
    LOGGER.info("APP_START device=%s workers=%d", config.model.device, config.runtime.num_workers)
    detector = PersonDetector(config.model, config.runtime)
    LOGGER.info("MODEL_LOADED path=%s", config.model.yolo_weight)

    source = VideoSource(config.video.source)
    ui = OpenCVUI(config.ui)
    try:
        source.open()
        LOGGER.info("SOURCE_OPENED source=%s", config.video.source)
        while True:
            frame = source.read()
            if frame is None:
                LOGGER.info("SOURCE_END source=%s", config.video.source)
                break

            detections = detector.predict(frame)
            annotated = draw_detections(
                frame,
                detections,
                detector.class_name,
                show_class_name=config.ui.show_class_name,
                show_confidence=config.ui.show_confidence,
            )
            if ui.show(annotated):
                LOGGER.info("USER_QUIT key=q")
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
