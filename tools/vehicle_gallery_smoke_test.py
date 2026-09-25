"""PC5 Vehicle SessionTarget + persistent Vehicle Gallery smoke test.

This is intentionally a standalone PC tool.  It does not add Vehicle Gallery
recognition to ``app.py``; automatic recognition remains a later PC6 feature.
"""

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
from src.vehicle_database import VehicleGalleryRepository
from src.vehicle_gallery import VehicleTargetGallery, format_vehicle_id
from src.vehicle_gallery_service import VehicleGalleryPersistenceService
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
WINDOW_NAME = "MVP-8.3-PC5 Vehicle Gallery"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PC5 Vehicle Gallery enrollment and persistence smoke test"
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
        raise ValueError("PC5 is PC-only and requires inference.backend=torch")
    if not config.vehicle_reid.enabled:
        raise ValueError("vehicle_reid.enabled must be true for PC5")

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
    vehicle_gallery = VehicleTargetGallery()
    vehicle_repository = VehicleGalleryRepository(config.vehicle_database.path)
    gallery_service = VehicleGalleryPersistenceService(
        vehicle_gallery,
        vehicle_repository,
    )
    gallery_service.load()
    LOGGER.info(
        "VEHICLE_GALLERY_LOADED path=%s vehicles=%d",
        config.vehicle_database.path,
        len(vehicle_gallery.all_vehicles()),
    )

    gallery_labels_by_target: dict[int, str] = {}

    def refresh_gallery_labels() -> None:
        gallery_labels_by_target.clear()
        for target in vehicle_target_manager.targets.values():
            vehicle = vehicle_gallery.vehicle_for_session_target(target.target_id)
            if vehicle is not None:
                gallery_labels_by_target[target.target_id] = format_vehicle_id(
                    vehicle.vehicle_id
                )

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

    def enroll_vehicle_from_roi(frame, vehicle_tracks, roi, frame_index):
        del frame, frame_index
        track = find_track_by_roi(
            roi,
            vehicle_tracks,
            min_iou=config.selection.min_iou,
        )
        if track is None:
            LOGGER.info("VEHICLE_GALLERY_ROI_FAILED roi=%s", roi)
            return None
        target = vehicle_target_manager.target_for_track(track.track_id)
        if target is None:
            LOGGER.info(
                "VEHICLE_GALLERY_ENROLL_REJECTED track=%d reason=not_session_target",
                track.track_id,
            )
            return None
        vehicle = gallery_service.enroll(target)
        refresh_gallery_labels()
        LOGGER.info(
            "VEHICLE_GALLERY_ENROLLED vehicle=%s target_id=%d track_id=%d",
            format_vehicle_id(vehicle.vehicle_id),
            target.target_id,
            track.track_id,
        )
        return vehicle

    frame_count = 0
    started = time.perf_counter()
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
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
                annotated = _render_frame(
                    frame,
                    output.person_tracks,
                    output.vehicle_tracks,
                    vehicle_target_manager,
                    show_unselected_tracks=config.ui.show_unselected_tracks,
                    gallery_labels_by_target=gallery_labels_by_target,
                )
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
                        gallery_labels_by_target=gallery_labels_by_target,
                        window_name=WINDOW_NAME,
                    )
                    if action is UIAction.QUIT:
                        break
                elif key in (ord("g"), ord("G")):
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
                        select_handler=enroll_vehicle_from_roi,
                        gallery_labels_by_target=gallery_labels_by_target,
                        window_name=WINDOW_NAME,
                    )
                    if action is UIAction.QUIT:
                        break
                elif key in (ord("r"), ord("R")):
                    target_ids_before = set(vehicle_target_manager.targets)
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
                        gallery_labels_by_target=gallery_labels_by_target,
                        window_name=WINDOW_NAME,
                    )
                    removed_target_ids = target_ids_before.difference(
                        vehicle_target_manager.targets
                    )
                    gallery_service.detach_all_session_targets(removed_target_ids)
                    refresh_gallery_labels()
                    if action is UIAction.QUIT:
                        break
                elif key in (ord("c"), ord("C")):
                    target_ids = tuple(vehicle_target_manager.targets)
                    vehicle_target_manager.clear()
                    gallery_service.detach_all_session_targets(target_ids)
                    refresh_gallery_labels()
                    LOGGER.info("VEHICLE_TARGETS_CLEARED")
    finally:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    stats = tracking_pipeline.stats()
    LOGGER.info(
        "PC5_STATS frames=%d average_fps=%.2f unique_person_tracks=%d "
        "unique_vehicle_tracks=%d yolo_inference_count=%d vehicle_targets=%d "
        "gallery_vehicles=%d",
        frame_count,
        frame_count / elapsed if elapsed > 0 else 0.0,
        stats.unique_person_tracks,
        stats.unique_vehicle_tracks,
        stats.yolo_inference_count,
        len(vehicle_target_manager.targets),
        len(vehicle_gallery.all_vehicles()),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main())
