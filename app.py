"""MVP-8 entry point: tracking, ReID recovery, and Gallery recognition."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from src.config import AppConfig, load_config, parse_source
from src.database import GalleryRepository
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
            "MVP-8 tracking, session-target ReID recovery, and automatic "
            "persistent Gallery recognition"
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
    gallery_service = GalleryPersistenceService(gallery, repository)
    loaded_people = gallery_service.load()
    LOGGER.info(
        "GALLERY_LOADED path=%s people=%d",
        repository.path,
        len(loaded_people),
    )

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
    embedding_cache = ReIDFrameCache()
    target_recovery = TargetRecoveryCoordinator(
        target_manager=target_manager,
        reid_extractor=reid_extractor,
        reid_config=config.reid,
        recovery_config=config.reid_recovery,
        embedding_cache=embedding_cache,
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
            frame = source.read()
            if frame is None:
                LOGGER.info("SOURCE_END source=%s", config.video.source)
                break

            current_frame_index = frame_index
            tracks = tracking_pipeline.process(frame)
            target_recovery.process_frame(frame, tracks, current_frame_index)
            gallery_recognition.process_frame(
                frame,
                tracks,
                current_frame_index,
                protected_track_ids=target_recovery.last_recovered_track_ids,
            )
            frame_index += 1
            annotated = render_tracks(frame, tracks)
            action = ui.show(annotated)
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
