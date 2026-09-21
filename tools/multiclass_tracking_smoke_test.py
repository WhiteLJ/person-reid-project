"""Run the PC2 Person + Vehicle detection/tracking smoke test."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config, parse_source
from src.models import Track
from src.pc_multiclass_tracking import MultiClassTrackingPipeline
from src.video_source import VideoSource


LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_DISPLAY_WIDTH = 960


def _fit_display_width(
    frame: np.ndarray,
    max_width: int = DEFAULT_MAX_DISPLAY_WIDTH,
) -> np.ndarray:
    """Resize only the display copy while preserving the frame aspect ratio."""

    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    display_size = (max_width, max(1, int(round(height * scale))))
    return cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)


def _draw_track(
    frame: np.ndarray,
    track: Track,
    *,
    prefix: str,
    color: tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (
        max(0, min(width - 1, int(round(value))))
        for value in track.bbox
    )
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{prefix}{track.track_id} {track.confidence:.2f}"
    text_y = max(20, y1 - 8)
    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PC2 one-pass Person + Vehicle BoT-SORT smoke test"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="project configuration path",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="camera index or video path; defaults to config video.source",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    source = config.video.source if args.source is None else parse_source(args.source)
    pipeline = MultiClassTrackingPipeline(config)
    start = time.perf_counter()
    frames = 0

    LOGGER.info(
        "PC2_START source=%s person_class=%s vehicle_classes=%s tracker=%s",
        source,
        config.model.person_class_id,
        config.multiclass_tracking.vehicle_class_ids,
        config.tracking.tracker,
    )
    cv2.namedWindow("MVP-8.3-PC2", cv2.WINDOW_NORMAL)
    try:
        with VideoSource(source) as video:
            while True:
                frame = video.read()
                if frame is None:
                    break
                output = pipeline.process(frame)
                display = frame.copy()
                for track in output.person_tracks:
                    _draw_track(
                        display,
                        track,
                        prefix="Person P-T",
                        color=(0, 200, 0),
                    )
                for track in output.vehicle_tracks:
                    _draw_track(
                        display,
                        track,
                        prefix="Vehicle V-T",
                        color=(0, 165, 255),
                    )
                display = _fit_display_width(display)
                cv2.imshow("MVP-8.3-PC2", display)
                frames += 1
                if cv2.waitKey(config.ui.wait_key_ms) & 0xFF in (ord("q"), ord("Q")):
                    break
    finally:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    stats = pipeline.stats()
    average_fps = frames / elapsed if elapsed > 0 else 0.0
    LOGGER.info(
        "PC2_STATS frames=%s average_fps=%.2f unique_person_tracks=%s "
        "unique_vehicle_tracks=%s yolo_inference_count=%s",
        stats.frames,
        average_fps,
        stats.unique_person_tracks,
        stats.unique_vehicle_tracks,
        stats.yolo_inference_count,
    )
    if stats.yolo_inference_count != stats.frames:
        LOGGER.error(
            "PC2_INFERENCE_COUNT_MISMATCH frames=%s inferences=%s",
            stats.frames,
            stats.yolo_inference_count,
        )
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main())
