"""MVP-8.2 entry point: conservative tracking, recognition, and enrichment."""

from __future__ import annotations

import argparse
import logging
from time import perf_counter
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from src.config import AppConfig, load_config, parse_source
from src.ascend_runtime import AscendRuntime
from src.database import GalleryRepository
from src.diagnostics import RuntimeDiagnostics
from src.gallery import TargetGallery, format_person_id
from src.gallery_service import GalleryPersistenceService
from src.gallery_recognition import GalleryRecognitionCoordinator
from src.logging_utils import configure_logging
from src.reid import ReIDExtractor
from src.reid_frame_cache import ReIDFrameCache
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
        description=(
            "MVP-8.2 tracking, session-target recovery, Gallery recognition, "
            "and safe persistent feature enrichment"
        )
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
    repository = GalleryRepository(config.database.path)
    repository.initialize()
    gallery = TargetGallery()
    gallery_service = GalleryPersistenceService(
        gallery,
        repository,
        enrichment_config=config.gallery_enrichment,
    )
    loaded_people = gallery_service.load()
    LOGGER.info(
        "GALLERY_LOADED path=%s people=%d",
        repository.path,
        len(loaded_people),
    )

    LOGGER.info(
        "APP_START backend=%s device=%s workers=%d",
        config.inference.backend,
        config.model.device,
        config.runtime.num_workers,
    )
    ascend_runtime = None
    if config.inference.backend == "ascend":
        ascend_runtime = AscendRuntime(config.ascend.device_id)
    try:
        tracking_pipeline = TrackingPipeline(
            model_config=config.model,
            runtime_config=config.runtime,
            tracking_config=config.tracking,
            inference_config=config.inference,
            ascend_config=config.ascend,
            ascend_runtime=ascend_runtime,
        )
        reid_extractor = ReIDExtractor(
            config.reid,
            config.model.device,
            backend=config.inference.backend,
            ascend_config=config.ascend,
            ascend_runtime=ascend_runtime,
        )
    except Exception:
        if ascend_runtime is not None:
            ascend_runtime.close()
        raise
    detector_model_path = (
        config.model.yolo_weight
        if config.inference.backend == "torch"
        else config.ascend.yolo_model
    )
    LOGGER.info(
        "MODEL_LOADED backend=%s path=%s tracker=%s persist=%s",
        config.inference.backend,
        detector_model_path,
        config.tracking.tracker,
        config.tracking.persist,
    )
    LOGGER.info(
        "REID_MODEL_LOADED backend=%s name=%s checkpoint=%s device=%s",
        config.inference.backend,
        config.reid.model_name,
        config.reid.weight if config.inference.backend == "torch" else config.ascend.reid_model,
        reid_extractor.device,
    )

    source = VideoSource(config.video.source)
    ui = OpenCVUI(config.ui)
    target_manager = TargetManager()
    embedding_cache = ReIDFrameCache()
    target_recovery = TargetRecoveryCoordinator(
        target_manager=target_manager,
        reid_extractor=reid_extractor,
        reid_config=config.reid,
        recovery_config=config.reid_recovery,
        embedding_cache=embedding_cache,
        quality_config=config.reid_quality,
        person_class_id=config.model.person_class_id,
    )
    gallery_recognition = GalleryRecognitionCoordinator(
        target_manager=target_manager,
        gallery=gallery,
        reid_extractor=reid_extractor,
        reid_config=config.reid,
        recognition_config=config.gallery_recognition,
        recovery_config=config.reid_recovery,
        person_class_id=config.model.person_class_id,
        embedding_cache=embedding_cache,
        quality_config=config.reid_quality,
    )
    diagnostics = RuntimeDiagnostics(
        enabled=config.diagnostics.enabled,
        log_interval_frames=config.diagnostics.log_interval_frames,
    )
    frame_index = 0
    current_frame_index = -1

    def render_tracks(render_frame, render_tracks):
        gallery_labels_by_track = {}
        for target in target_manager.active_targets():
            if target.current_track_id is None:
                continue
            person = gallery.person_for_session_target(target.target_id)
            if person is not None:
                gallery_labels_by_track[target.current_track_id] = format_person_id(
                    person.person_id
                )
        return draw_tracks(
            render_frame,
            render_tracks,
            class_name=tracking_pipeline.class_name,
            show_class_name=config.ui.show_class_name,
            show_track_id=config.tracking.show_track_id,
            show_confidence=config.ui.show_confidence,
            selected_track_ids=target_manager.selected_track_ids,
            show_unselected_tracks=config.ui.show_unselected_tracks,
            gallery_labels_by_track=gallery_labels_by_track,
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
                gallery.detach_session_target(target.target_id)
                target_manager.deselect(track)
                LOGGER.info(
                    "TARGET_REMOVED target=%d track=%d",
                    target.target_id,
                    track.track_id,
                )
            return

        if mode == EditMode.ENROLL_GALLERY:
            target = target_manager.target_for_track(track.track_id)
            if target is None:
                LOGGER.info(
                    "GALLERY_ENROLL_REJECTED track=%d reason=not_session_target",
                    track.track_id,
                )
                return
            already_enrolled = gallery.person_for_session_target(target.target_id)
            try:
                person = gallery_service.enroll(target)
            except Exception:
                LOGGER.exception(
                    "GALLERY_ENROLL_FAILED target=%d track=%d",
                    target.target_id,
                    track.track_id,
                )
                return
            LOGGER.info(
                "GALLERY_ENROLLED person=%s target=%d track=%d already_enrolled=%s",
                format_person_id(person.person_id),
                target.target_id,
                track.track_id,
                already_enrolled is not None,
            )

    try:
        source.open()
        LOGGER.info("SOURCE_OPENED source=%s", config.video.source)
        while True:
            frame_started = perf_counter()
            frame = source.read()
            if frame is None:
                LOGGER.info("SOURCE_END source=%s", config.video.source)
                break

            current_frame_index = frame_index
            tracking_started = perf_counter()
            tracks = tracking_pipeline.process(frame)
            tracking_seconds = perf_counter() - tracking_started
            diagnostics.observe_tracks(tracks, current_frame_index)
            recovery_started = perf_counter()
            target_recovery.process_frame(frame, tracks, current_frame_index)
            recovery_seconds = perf_counter() - recovery_started
            gallery_started = perf_counter()
            gallery_service.update_runtime_state(
                target_manager.targets.values(),
                current_frame_index,
            )
            gallery_service.enrich_reference_updates(
                target_recovery.drain_reference_updates()
            )
            gallery_recognition.process_frame(
                frame,
                tracks,
                current_frame_index,
                protected_track_ids=target_recovery.last_recovered_track_ids,
            )
            gallery_seconds = perf_counter() - gallery_started
            frame_index += 1
            render_started = perf_counter()
            annotated = render_tracks(frame, tracks)
            render_seconds = perf_counter() - render_started
            ui_started = perf_counter()
            action = ui.show(annotated)
            ui_seconds = perf_counter() - ui_started
            recovery_stats = target_recovery.last_frame_recovery_stats
            diagnostics.record_frame(
                perf_counter() - frame_started,
                current_frame_index,
                tracking_seconds=tracking_seconds,
                recovery_seconds=recovery_seconds,
                gallery_seconds=gallery_seconds,
                render_seconds=render_seconds,
                ui_seconds=ui_seconds,
                recovery_due=recovery_stats.recovery_due,
                recovery_candidate_count=recovery_stats.candidate_count,
                recovery_quality_valid_count=recovery_stats.quality_valid_count,
                recovery_reid_batch_count=recovery_stats.reid_batch_count,
                recovery_reid_seconds=recovery_stats.reid_ms / 1000.0,
                recovery_sweep_started=recovery_stats.sweep_started,
                recovery_sweep_completed=recovery_stats.sweep_completed,
                recovery_sweep_candidate_total=recovery_stats.sweep_candidate_total,
                recovery_sweep_processed_this_frame=(
                    recovery_stats.sweep_processed_this_frame
                ),
                recovery_sweep_frames=recovery_stats.sweep_frames,
                recovery_sweep_reid_ms=recovery_stats.sweep_reid_ms,
            )
            if action == UIAction.QUIT:
                LOGGER.info("USER_QUIT key=q")
                break
            if action == UIAction.CLEAR_TARGETS:
                selected_count = len(target_manager.targets)
                gallery.detach_all_session_targets(tuple(target_manager.targets))
                target_manager.clear()
                LOGGER.info("TARGETS_CLEARED count=%d", selected_count)
                continue

            if action in (
                UIAction.SELECT_TARGET,
                UIAction.REMOVE_TARGET,
                UIAction.ENROLL_GALLERY,
            ):
                edit_modes = {
                    UIAction.SELECT_TARGET: EditMode.ADD_TARGETS,
                    UIAction.REMOVE_TARGET: EditMode.REMOVE_TARGETS,
                    UIAction.ENROLL_GALLERY: EditMode.ENROLL_GALLERY,
                }
                edit_mode = edit_modes[action]
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
        diagnostics.log_summary(
            target_lost=target_manager.target_lost_count,
            target_recovered=target_manager.target_recovered_count,
            recovery_attempted=target_recovery.recovery_attempted_count,
            recovery_pending=target_recovery.recovery_pending_count,
            recovery_accepted=target_recovery.recovery_accepted_count,
            quality_rejected=(
                target_recovery.quality_rejected_count
                + gallery_recognition.quality_rejected_count
            ),
            gallery_recognized=gallery_recognition.recognized_count,
        )
        LOGGER.info("APP_STOP")
        if ascend_runtime is not None:
            ascend_runtime.close()
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
