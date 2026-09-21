"""PC4A Vehicle ACTIVE/LOST/incremental-ReID recovery smoke test."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from src.config import load_config, parse_source
from src.display_transform import DisplayTransform
from src.pc_multiclass_tracking import MultiClassTrackingPipeline
from src.reid_frame_cache import ReIDFrameCache
from src.roi_selector import find_track_by_roi
from src.target_manager import TargetManager
from src.vehicle_recovery import VehicleRecoveryCoordinator
from src.vehicle_reid import VehicleReIDExtractor
from src.vehicle_selection import VehicleSelectionController
from src.video_source import VideoSource
from tools.vehicle_selection_smoke_test import (
    _render_frame,
    _run_vehicle_edit_session,
)
from ui.roi_editor import UIAction


LOGGER = logging.getLogger(__name__)
WINDOW_NAME = "MVP-8.3-PC4A Vehicle Recovery"


def _draw_lost_status(frame, target_manager: TargetManager) -> None:
    y = 28
    for target in sorted(target_manager.lost_targets(), key=lambda item: item.target_id):
        cv2.putText(
            frame,
            f"VT-{target.target_id} LOST",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (139, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 26


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PC4A Vehicle LOST and incremental ReID recovery smoke test"
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
        raise ValueError("PC4A is PC-only and requires inference.backend=torch")
    if not config.vehicle_reid.enabled:
        raise ValueError("vehicle_reid.enabled must be true for PC4A")

    source = config.video.source if args.source is None else parse_source(args.source)
    tracking_pipeline = MultiClassTrackingPipeline(config)
    vehicle_target_manager = TargetManager()
    vehicle_reid = VehicleReIDExtractor(config.vehicle_reid)
    vehicle_cache = ReIDFrameCache()
    vehicle_recovery = VehicleRecoveryCoordinator(
        target_manager=vehicle_target_manager,
        reid_extractor=vehicle_reid,
        reid_config=config.vehicle_reid,
        recovery_config=config.vehicle_recovery,
        quality_config=config.vehicle_reid_quality,
        vehicle_class_ids=config.multiclass_tracking.vehicle_class_ids,
        embedding_cache=vehicle_cache,
    )
    selection_controller = VehicleSelectionController(
        vehicle_target_manager,
        vehicle_reid,
        min_iou=config.selection.min_iou,
    )

    frame_count = 0
    started = time.perf_counter()
    recovery_candidate_total = 0
    recovery_quality_valid_total = 0
    recovery_reid_batch_total = 0
    recovery_reid_ms_total = 0.0
    sweep_count = 0

    def select_vehicle_from_roi(frame, vehicle_tracks, roi, frame_index):
        track = find_track_by_roi(
            roi,
            vehicle_tracks,
            min_iou=config.selection.min_iou,
        )
        if track is None:
            LOGGER.info("VEHICLE_ROI_FAILED roi=%s", roi)
            return None
        return vehicle_recovery.select_from_track(
            frame,
            track,
            frame_index,
            tracks=vehicle_tracks,
        )

    LOGGER.info(
        "PC4A_START source=%s budget=%d tracker=%s",
        source,
        config.vehicle_recovery.recovery_candidates_per_frame,
        config.tracking.tracker,
    )
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        with VideoSource(source) as video:
            while True:
                frame = video.read()
                if frame is None:
                    break
                output = tracking_pipeline.process(frame)
                vehicle_recovery.process_frame(
                    frame,
                    output.vehicle_tracks,
                    frame_count,
                )
                recovery_stats = vehicle_recovery.last_frame_recovery_stats
                recovery_candidate_total += recovery_stats.candidate_count
                recovery_quality_valid_total += recovery_stats.quality_valid_count
                recovery_reid_batch_total += recovery_stats.reid_batch_count
                recovery_reid_ms_total += recovery_stats.reid_ms
                sweep_count += int(recovery_stats.sweep_completed)

                annotated = _render_frame(
                    frame,
                    output.person_tracks,
                    output.vehicle_tracks,
                    vehicle_target_manager,
                    show_unselected_tracks=config.ui.show_unselected_tracks,
                )
                _draw_lost_status(annotated, vehicle_target_manager)
                transform = DisplayTransform.from_frame(
                    frame,
                    config.ui.max_display_width,
                )
                cv2.imshow(WINDOW_NAME, transform.source_to_display(annotated))
                frame_count += 1
                key = cv2.waitKey(config.ui.wait_key_ms) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")):
                    action = _run_vehicle_edit_session(
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
                        select_handler=select_vehicle_from_roi,
                        window_name=WINDOW_NAME,
                    )
                    if action is UIAction.QUIT:
                        break
                elif key in (ord("r"), ord("R")):
                    action = _run_vehicle_edit_session(
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
                        window_name=WINDOW_NAME,
                    )
                    if action is UIAction.QUIT:
                        break
                elif key in (ord("c"), ord("C")):
                    vehicle_target_manager.clear()
                    LOGGER.info("VEHICLE_TARGETS_CLEARED")
    finally:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    stats = tracking_pipeline.stats()
    LOGGER.info(
        "PC4A_STATS frames=%d average_fps=%.2f unique_person_tracks=%d "
        "unique_vehicle_tracks=%d yolo_inference_count=%d vehicle_targets=%d",
        frame_count,
        frame_count / elapsed if elapsed > 0 else 0.0,
        stats.unique_person_tracks,
        stats.unique_vehicle_tracks,
        stats.yolo_inference_count,
        len(vehicle_target_manager.targets),
    )
    LOGGER.info(
        "VEHICLE_RECOVERY_STATS attempted=%d candidate_count=%d "
        "quality_valid_count=%d reid_batch_count=%d reid_ms=%.2f sweeps=%d",
        vehicle_recovery.recovery_attempted_count,
        recovery_candidate_total,
        recovery_quality_valid_total,
        recovery_reid_batch_total,
        recovery_reid_ms_total,
        sweep_count,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main())
