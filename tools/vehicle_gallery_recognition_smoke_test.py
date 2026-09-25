"""PC6 Vehicle Gallery recognition, recovery, and enrichment smoke test."""

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
from src.vehicle_gallery_recognition import VehicleGalleryRecognitionCoordinator
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
WINDOW_NAME = "MVP-8.3-PC6 Vehicle Gallery Recognition"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PC6 Vehicle persistent Gallery recognition smoke test"
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
        raise ValueError("PC6 is PC-only and requires inference.backend=torch")
    if not config.vehicle_reid.enabled:
        raise ValueError("vehicle_reid.enabled must be true for PC6")

    source = config.video.source if args.source is None else parse_source(args.source)
    tracking_pipeline = MultiClassTrackingPipeline(config)
    target_manager = TargetManager()
    vehicle_reid = VehicleReIDExtractor(config.vehicle_reid)
    vehicle_cache = ReIDFrameCache()
    recovery = VehicleRecoveryCoordinator(
        target_manager=target_manager,
        reid_extractor=vehicle_reid,
        reid_config=config.vehicle_reid,
        recovery_config=config.vehicle_recovery,
        quality_config=config.vehicle_reid_quality,
        vehicle_class_ids=config.multiclass_tracking.vehicle_class_ids,
        embedding_cache=vehicle_cache,
    )
    gallery = VehicleTargetGallery()
    repository = VehicleGalleryRepository(config.vehicle_database.path)
    persistence = VehicleGalleryPersistenceService(
        gallery,
        repository,
        enrichment_config=config.vehicle_gallery_enrichment,
    )
    loaded = persistence.load()
    LOGGER.info(
        "VEHICLE_GALLERY_LOADED path=%s vehicles=%d",
        repository.path,
        len(loaded),
    )
    recognition = VehicleGalleryRecognitionCoordinator(
        target_manager=target_manager,
        gallery=gallery,
        reid_extractor=vehicle_reid,
        reid_config=config.vehicle_reid,
        recognition_config=config.vehicle_gallery_recognition,
        recovery_config=config.vehicle_recovery,
        quality_config=config.vehicle_reid_quality,
        vehicle_class_ids=config.multiclass_tracking.vehicle_class_ids,
        embedding_cache=vehicle_cache,
    )
    selection = VehicleSelectionController(
        target_manager,
        vehicle_reid,
        min_iou=config.selection.min_iou,
    )
    gallery_labels_by_target: dict[int, str] = {}

    def refresh_gallery_labels() -> None:
        gallery_labels_by_target.clear()
        for target in target_manager.targets.values():
            vehicle = gallery.vehicle_for_session_target(target.target_id)
            if vehicle is not None:
                gallery_labels_by_target[target.target_id] = format_vehicle_id(
                    vehicle.vehicle_id
                )

    def select_vehicle_from_roi(frame, tracks, roi, frame_index):
        track = find_track_by_roi(roi, tracks, min_iou=config.selection.min_iou)
        if track is None:
            LOGGER.info("VEHICLE_ROI_FAILED roi=%s", roi)
            return None
        return recovery.select_from_track(frame, track, frame_index, tracks=tracks)

    def enroll_vehicle_from_roi(frame, tracks, roi, frame_index):
        del frame, frame_index
        track = find_track_by_roi(roi, tracks, min_iou=config.selection.min_iou)
        if track is None:
            LOGGER.info("VEHICLE_GALLERY_ROI_FAILED roi=%s", roi)
            return None
        target = target_manager.target_for_track(track.track_id)
        if target is None:
            LOGGER.info(
                "VEHICLE_GALLERY_ENROLL_REJECTED track=%d reason=not_session_target",
                track.track_id,
            )
            return None
        vehicle = persistence.enroll(target)
        recognition.notify_gallery_changed()
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
    recognition_matches = 0
    enrichment_updates = 0
    recovery_reid_ms_total = 0.0
    recognition_reid_ms_total = 0.0
    recognition_reid_batches = 0
    recognition_retry_skipped = 0
    recovery_sweep_started = 0
    recovery_sweep_completed = 0
    recovery_sweep_candidate_total = 0
    recovery_sweep_processed = 0
    recovery_sweep_frames = 0
    recovery_sweep_reid_ms = 0.0
    recognition_sweep_started = 0
    recognition_sweep_completed = 0
    recognition_sweep_candidate_total = 0
    recognition_sweep_processed = 0
    recognition_sweep_frames = 0
    recognition_sweep_reid_ms = 0.0
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        with VideoSource(source) as video:
            while True:
                frame = video.read()
                if frame is None:
                    break

                output = tracking_pipeline.process(frame)
                recovery.process_frame(frame, output.vehicle_tracks, frame_count)
                recovery_stats = recovery.last_frame_recovery_stats
                recovery_reid_ms_total += recovery_stats.reid_ms
                recovery_sweep_started += int(recovery_stats.sweep_started)
                recovery_sweep_completed += int(recovery_stats.sweep_completed)
                if recovery_stats.sweep_started:
                    recovery_sweep_candidate_total += (
                        recovery_stats.sweep_candidate_total
                    )
                recovery_sweep_processed += recovery_stats.sweep_processed_this_frame
                if recovery_stats.sweep_completed:
                    recovery_sweep_frames += recovery_stats.sweep_frames
                    recovery_sweep_reid_ms += recovery_stats.sweep_reid_ms
                persistence.update_runtime_state(
                    target_manager.targets.values(),
                    frame_count,
                )
                enrichment_updates += persistence.enrich_reference_updates(
                    recovery.drain_reference_updates()
                )
                matches = recognition.process_frame(
                    frame,
                    output.vehicle_tracks,
                    frame_count,
                    protected_track_ids=recovery.last_recovered_track_ids,
                )
                recognition_stats = recognition.last_frame_recognition_stats
                recognition_reid_ms_total += recognition_stats.reid_ms
                recognition_reid_batches += recognition_stats.reid_batch_count
                recognition_retry_skipped += recognition_stats.skipped_retry_cooldown
                recognition_sweep_started += int(recognition_stats.sweep_started)
                recognition_sweep_completed += int(recognition_stats.sweep_completed)
                if recognition_stats.sweep_started:
                    recognition_sweep_candidate_total += (
                        recognition_stats.sweep_candidate_total
                    )
                recognition_sweep_processed += (
                    recognition_stats.sweep_processed_this_frame
                )
                if recognition_stats.sweep_completed:
                    recognition_sweep_frames += recognition_stats.sweep_frames
                    recognition_sweep_reid_ms += recognition_stats.sweep_reid_ms
                for match in matches:
                    target = target_manager.target_for_track(
                        match.candidate.track.track_id
                    )
                    if target is None:
                        continue
                    persistence.mark_auto_recognized(target.target_id)
                    LOGGER.info(
                        "VEHICLE_GALLERY_RECOGNIZED vehicle=%s target_id=%d "
                        "track_id=%d similarity=%.4f",
                        format_vehicle_id(match.person_id),
                        target.target_id,
                        match.candidate.track.track_id,
                        match.similarity,
                    )
                    recognition_matches += 1
                refresh_gallery_labels()

                annotated = _render_frame(
                    frame,
                    output.person_tracks,
                    output.vehicle_tracks,
                    target_manager,
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
                        target_manager,
                        selection,
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
                        target_manager,
                        selection,
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
                    before = set(target_manager.targets)
                    action = _run_vehicle_edit_session(
                        frame,
                        output.person_tracks,
                        output.vehicle_tracks,
                        target_manager,
                        selection,
                        frame_count - 1,
                        remove=True,
                        wait_key_ms=config.ui.wait_key_ms,
                        max_display_width=config.ui.max_display_width,
                        show_unselected_tracks=config.ui.show_unselected_tracks,
                        gallery_labels_by_target=gallery_labels_by_target,
                        window_name=WINDOW_NAME,
                    )
                    persistence.detach_all_session_targets(
                        before.difference(target_manager.targets)
                    )
                    recognition.notify_gallery_changed()
                    refresh_gallery_labels()
                    if action is UIAction.QUIT:
                        break
                elif key in (ord("c"), ord("C")):
                    target_ids = tuple(target_manager.targets)
                    target_manager.clear()
                    persistence.detach_all_session_targets(target_ids)
                    recognition.notify_gallery_changed()
                    refresh_gallery_labels()
                    LOGGER.info("VEHICLE_TARGETS_CLEARED")
    finally:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    stats = tracking_pipeline.stats()
    LOGGER.info(
        "PC6_STATS frames=%d average_fps=%.2f unique_person_tracks=%d "
        "unique_vehicle_tracks=%d yolo_inference_count=%d vehicle_targets=%d "
        "gallery_vehicles=%d recognized=%d enrichment_updates=%d "
        "recognition_reid_ms=%.2f recognition_reid_batches=%d "
        "recognition_retry_skipped=%d recognition_quality_rejected=%d "
        "recovery_reid_ms=%.2f recovery_sweep_started=%d "
        "recovery_sweep_completed=%d recovery_sweep_candidate_total=%d "
        "recovery_sweep_processed=%d recovery_sweep_frames=%d "
        "recovery_sweep_reid_ms=%.2f recognition_sweep_started=%d "
        "recognition_sweep_completed=%d recognition_sweep_candidate_total=%d "
        "recognition_sweep_processed=%d recognition_sweep_frames=%d "
        "recognition_sweep_reid_ms=%.2f",
        frame_count,
        frame_count / elapsed if elapsed > 0 else 0.0,
        stats.unique_person_tracks,
        stats.unique_vehicle_tracks,
        stats.yolo_inference_count,
        len(target_manager.targets),
        len(gallery.all_vehicles()),
        recognition_matches,
        enrichment_updates,
        recognition_reid_ms_total,
        recognition_reid_batches,
        recognition_retry_skipped,
        recognition.quality_rejected_count,
        recovery_reid_ms_total,
        recovery_sweep_started,
        recovery_sweep_completed,
        recovery_sweep_candidate_total,
        recovery_sweep_processed,
        recovery_sweep_frames,
        recovery_sweep_reid_ms,
        recognition_sweep_started,
        recognition_sweep_completed,
        recognition_sweep_candidate_total,
        recognition_sweep_processed,
        recognition_sweep_frames,
        recognition_sweep_reid_ms,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main())
