"""Board-side smoke test for Atlas Person + Vehicle detection and tracking."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from src.ascend_multiclass_tracking import AscendMultiClassTrackingPipeline
from src.ascend_runtime import AscendRuntime
from src.config import load_config, parse_source
from src.logging_utils import configure_logging
from src.target_manager import TargetManager
from src.video_source import VideoSource
from src.visualization import draw_multiclass_tracks
from ui.opencv_ui import OpenCVUI, UIAction


LOGGER = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atlas one-YOLO Person + Vehicle tracking smoke test"
    )
    parser.add_argument("--config", default="config/config_atlas.yaml")
    parser.add_argument("--source", help="optional camera index or local video path")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    configure_logging("INFO")
    config = load_config(Path(args.config))
    if config.inference.backend != "ascend":
        raise ValueError("Atlas multiclass smoke test requires inference.backend=ascend")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if args.source is not None:
        config = replace(config, video=replace(config.video, source=parse_source(args.source)))

    runtime = AscendRuntime(config.ascend.device_id)
    source = VideoSource(config.video.source)
    ui = OpenCVUI(config.ui)
    person_targets = TargetManager()
    vehicle_targets = TargetManager()
    pipeline = None
    frames = 0
    started = perf_counter()
    person_detection_count = 0
    vehicle_detection_count = 0
    try:
        pipeline = AscendMultiClassTrackingPipeline(config, runtime)
        source.open()
        while True:
            frame = source.read()
            if frame is None:
                break
            result = pipeline.process(frame)
            frames += 1
            person_detection_count += len(result.person_tracks)
            vehicle_detection_count += len(result.vehicle_tracks)
            annotated = draw_multiclass_tracks(
                frame,
                result.person_tracks,
                result.vehicle_tracks,
                person_targets,
                vehicle_targets,
                class_name=pipeline.class_name,
                show_class_name=config.ui.show_class_name,
                show_track_id=config.tracking.show_track_id,
                show_confidence=config.ui.show_confidence,
                show_unselected_tracks=config.ui.show_unselected_tracks,
            )
            if ui.show(annotated) is UIAction.QUIT:
                break
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        source.release()
        ui.close()
        runtime.close()

    if pipeline is None:
        raise RuntimeError("Atlas tracking pipeline was not initialized")
    stats = pipeline.stats()
    elapsed = perf_counter() - started
    if stats.yolo_inference_count != frames:
        raise RuntimeError(
            "one-YOLO invariant failed: "
            f"frames={frames} yolo_inference_count={stats.yolo_inference_count}"
        )
    LOGGER.info(
        "ATLAS_MULTICLASS_TRACKING_STATS frames=%d average_fps=%.2f "
        "yolo_inference_count=%d unique_person_tracks=%d "
        "unique_vehicle_tracks=%d person_tracks_seen=%d vehicle_tracks_seen=%d "
        "yolo_ms=%.2f person_tracker_ms=%.2f vehicle_tracker_ms=%.2f",
        stats.frames,
        frames / elapsed if elapsed > 0 else 0.0,
        stats.yolo_inference_count,
        stats.unique_person_tracks,
        stats.unique_vehicle_tracks,
        person_detection_count,
        vehicle_detection_count,
        stats.yolo_ms_total / max(1, stats.frames),
        stats.person_tracker_ms_total / max(1, stats.frames),
        stats.vehicle_tracker_ms_total / max(1, stats.frames),
    )
    LOGGER.info("ATLAS_MULTICLASS_TRACKING_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
