"""Formal Person + Vehicle entry point for PC Torch and Atlas 310B."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from src.ascend_runtime import AscendRuntime
from src.config import AppConfig, load_config, parse_source
from src.database import GalleryRepository
from src.diagnostics import RuntimeDiagnostics
from src.gallery import TargetGallery, format_person_id
from src.gallery_recognition import GalleryRecognitionCoordinator
from src.gallery_service import GalleryPersistenceService
from src.logging_utils import configure_logging
from src.reid import ReIDExtractor
from src.reid_frame_cache import ReIDFrameCache
from src.reid_frame_budget import ReIDFrameBudget
from src.roi_selector import find_track_by_roi
from src.target_manager import TargetManager
from src.target_recovery import TargetRecoveryCoordinator
from src.video_source import VideoSource
from src.visualization import draw_multiclass_tracks
from ui.opencv_ui import EditMode, OpenCVUI, UIAction


LOGGER = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Person + Vehicle PC tracking, recovery, recognition, and Gallery"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--source", help="optional camera index or local video path")
    return parser


def _override_source(config: AppConfig, source: str | None) -> AppConfig:
    if source is None:
        return config
    return replace(config, video=replace(config.video, source=parse_source(source)))


def _person_gallery_labels(manager: TargetManager, gallery: TargetGallery) -> dict[int, str]:
    labels: dict[int, str] = {}
    for target in manager.active_targets():
        if target.current_track_id is None:
            continue
        person = gallery.person_for_session_target(target.target_id)
        if person is not None:
            labels[target.current_track_id] = format_person_id(person.person_id)
    return labels


def _vehicle_gallery_labels(manager: TargetManager, gallery: Any, format_id: Any) -> dict[int, str]:
    labels: dict[int, str] = {}
    if manager is None or gallery is None:
        return labels
    for target in manager.active_targets():
        if target.current_track_id is None:
            continue
        vehicle = gallery.vehicle_for_session_target(target.target_id)
        if vehicle is not None:
            labels[target.target_id] = format_id(vehicle.vehicle_id)
    return labels


def run(config: AppConfig) -> int:
    """Run the formal Person + Vehicle integration on the selected backend."""

    is_torch_pc = config.inference.backend == "torch"
    ascend_runtime = None

    person_repository = GalleryRepository(config.database.path)
    person_repository.initialize()
    person_gallery = TargetGallery()
    person_gallery_service = GalleryPersistenceService(
        person_gallery,
        person_repository,
        enrichment_config=config.gallery_enrichment,
    )
    loaded_people = person_gallery_service.load()
    LOGGER.info("GALLERY_LOADED path=%s people=%d", person_repository.path, len(loaded_people))
    LOGGER.info(
        "APP_START backend=%s device=%s workers=%d",
        config.inference.backend,
        config.model.device,
        config.runtime.num_workers,
    )

    person_target_manager = TargetManager()
    person_cache = ReIDFrameCache()
    person_reid_budget = ReIDFrameBudget(
        config.reid_scheduling.person_max_new_embeddings_per_frame
    )
    vehicle_target_manager = None
    vehicle_cache = None
    vehicle_gallery = None
    vehicle_gallery_service = None
    vehicle_recovery = None
    vehicle_gallery_recognition = None
    vehicle_reid_extractor = None
    vehicle_class_ids: tuple[int, ...] = ()
    vehicle_format_id = None
    vehicle_reid_budget = ReIDFrameBudget(
        config.reid_scheduling.vehicle_max_new_embeddings_per_frame
    )

    try:
        if is_torch_pc:
            from src.pc_multiclass_tracking import MultiClassTrackingPipeline
            from src.vehicle_reid import VehicleReIDExtractor

            tracking_pipeline = MultiClassTrackingPipeline(config)
            reid_extractor = ReIDExtractor(config.reid, config.model.device, backend="torch")
            vehicle_reid_extractor = VehicleReIDExtractor(config.vehicle_reid)
            detector_model_path = config.model.yolo_weight
        else:
            from src.ascend_multiclass_tracking import AscendMultiClassTrackingPipeline

            ascend_runtime = AscendRuntime(config.ascend.device_id)
            tracking_pipeline = AscendMultiClassTrackingPipeline(config, ascend_runtime)
            reid_extractor = ReIDExtractor(
                config.reid,
                config.model.device,
                backend=config.inference.backend,
                ascend_config=config.ascend,
                ascend_runtime=ascend_runtime,
            )
            from src.ascend_vehicle_reid import AscendVehicleReIDExtractor

            vehicle_reid_extractor = AscendVehicleReIDExtractor(
                config.vehicle_reid,
                config.ascend.vehicle_reid_model,
                config.ascend.vehicle_reid_dynamic_batches,
                ascend_runtime,
            )
            detector_model_path = config.ascend.yolo_model

        # Person and Vehicle state is always separate.  The only difference
        # between Torch and Atlas here is which extractor is injected.
        from src.vehicle_database import VehicleGalleryRepository
        from src.vehicle_gallery import VehicleTargetGallery, format_vehicle_id
        from src.vehicle_gallery_recognition import VehicleGalleryRecognitionCoordinator
        from src.vehicle_gallery_service import VehicleGalleryPersistenceService
        from src.vehicle_recovery import VehicleRecoveryCoordinator

        vehicle_class_ids = tuple(config.multiclass_tracking.vehicle_class_ids)
        vehicle_target_manager = TargetManager()
        vehicle_cache = ReIDFrameCache()
        vehicle_repository = VehicleGalleryRepository(config.vehicle_database.path)
        vehicle_repository.initialize()
        vehicle_gallery = VehicleTargetGallery()
        vehicle_gallery_service = VehicleGalleryPersistenceService(
            vehicle_gallery,
            vehicle_repository,
            enrichment_config=config.vehicle_gallery_enrichment,
        )
        loaded_vehicles = vehicle_gallery_service.load()
        vehicle_format_id = format_vehicle_id
        LOGGER.info(
            "VEHICLE_GALLERY_LOADED path=%s vehicles=%d",
            vehicle_repository.path,
            len(loaded_vehicles),
        )
        vehicle_recovery = VehicleRecoveryCoordinator(
            target_manager=vehicle_target_manager,
            reid_extractor=vehicle_reid_extractor,
            reid_config=config.vehicle_reid,
            recovery_config=config.vehicle_recovery,
            quality_config=config.vehicle_reid_quality,
            vehicle_class_ids=vehicle_class_ids,
            embedding_cache=vehicle_cache,
            reid_budget=vehicle_reid_budget,
        )
        vehicle_gallery_recognition = VehicleGalleryRecognitionCoordinator(
            target_manager=vehicle_target_manager,
            gallery=vehicle_gallery,
            reid_extractor=vehicle_reid_extractor,
            reid_config=config.vehicle_reid,
            recognition_config=config.vehicle_gallery_recognition,
            recovery_config=config.vehicle_recovery,
            quality_config=config.vehicle_reid_quality,
            vehicle_class_ids=vehicle_class_ids,
            embedding_cache=vehicle_cache,
            reid_budget=vehicle_reid_budget,
        )
    except Exception:
        if ascend_runtime is not None:
            ascend_runtime.close()
        raise

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
        config.reid.weight if is_torch_pc else config.ascend.reid_model,
        reid_extractor.device,
    )
    LOGGER.info(
        "VEHICLE_REID_MODEL_LOADED backend=%s name=%s model=%s device=%s classes=%s",
        config.inference.backend,
        config.vehicle_reid.model_name,
        config.vehicle_reid.weight if is_torch_pc else config.ascend.vehicle_reid_model,
        getattr(
            vehicle_reid_extractor,
            "device",
            f"ascend:{config.ascend.device_id}" if not is_torch_pc else "unknown",
        ),
        vehicle_class_ids,
    )

    person_recovery = TargetRecoveryCoordinator(
        target_manager=person_target_manager,
        reid_extractor=reid_extractor,
        reid_config=config.reid,
        recovery_config=config.reid_recovery,
        embedding_cache=person_cache,
        quality_config=config.reid_quality,
        person_class_id=config.model.person_class_id,
        reid_budget=person_reid_budget,
    )
    person_gallery_recognition = GalleryRecognitionCoordinator(
        target_manager=person_target_manager,
        gallery=person_gallery,
        reid_extractor=reid_extractor,
        reid_config=config.reid,
        recognition_config=config.gallery_recognition,
        recovery_config=config.reid_recovery,
        person_class_id=config.model.person_class_id,
        embedding_cache=person_cache,
        quality_config=config.reid_quality,
        reid_budget=person_reid_budget,
    )
    diagnostics = RuntimeDiagnostics(
        enabled=config.diagnostics.enabled,
        log_interval_frames=config.diagnostics.log_interval_frames,
    )
    source = VideoSource(config.video.source)
    ui = OpenCVUI(config.ui)
    frame_index = 0
    current_frame_index = -1
    tracking_result = None
    tracks: list[Any] = []
    person_tracks: list[Any] = []
    vehicle_tracks: list[Any] = []

    person_recovery_reid_ms = 0.0
    person_recognition_reid_ms = 0.0
    person_recovery_batches = person_recognition_batches = 0
    person_recovery_processed = person_recognition_processed = 0
    person_retry_skipped = 0
    person_new_embeddings_total = 0
    person_new_embeddings_max = 0
    person_budget_deferred = 0
    vehicle_recovery_reid_ms = 0.0
    vehicle_recognition_reid_ms = 0.0
    vehicle_recovery_batches = vehicle_recognition_batches = 0
    vehicle_recovery_processed = vehicle_recognition_processed = 0
    vehicle_retry_skipped = 0
    vehicle_new_embeddings_total = 0
    vehicle_new_embeddings_max = 0
    vehicle_budget_deferred = 0
    last_vehicle_recovered_track_ids: frozenset[int] = frozenset()
    pc7_elapsed_seconds = 0.0

    def render_frames(render_frame: Any, _frozen_tracks: tuple[Any, ...] | None = None) -> Any:
        person_labels = _person_gallery_labels(person_target_manager, person_gallery)
        return draw_multiclass_tracks(
            render_frame,
            person_tracks,
            vehicle_tracks,
            person_target_manager,
            vehicle_target_manager,
            class_name=tracking_pipeline.class_name,
            show_class_name=config.ui.show_class_name,
            show_track_id=config.tracking.show_track_id,
            show_confidence=config.ui.show_confidence,
            show_unselected_tracks=config.ui.show_unselected_tracks,
            person_gallery_labels_by_track=person_labels,
            vehicle_gallery_labels_by_target=_vehicle_gallery_labels(
                vehicle_target_manager, vehicle_gallery, vehicle_format_id
            ),
        )

    def handle_roi(roi, frozen_tracks, mode):
        track = find_track_by_roi(roi, frozen_tracks, min_iou=config.selection.min_iou)
        if track is None:
            LOGGER.info("TARGET_ROI_FAILED mode=%s roi=%s", mode.name, roi)
            return

        is_person = track.class_id == config.model.person_class_id
        is_vehicle = track.class_id in vehicle_class_ids
        if not is_person and not is_vehicle:
            return
        manager = person_target_manager if is_person else vehicle_target_manager
        gallery = person_gallery if is_person else vehicle_gallery
        recovery = person_recovery if is_person else vehicle_recovery
        recognition = (
            person_gallery_recognition if is_person else vehicle_gallery_recognition
        )
        service = person_gallery_service if is_person else vehicle_gallery_service
        if manager is None or gallery is None or recovery is None or service is None:
            return
        domain_tracks = person_tracks if is_person else vehicle_tracks

        if mode == EditMode.ADD_TARGETS:
            recovery.select_from_track(frame, track, current_frame_index, tracks=domain_tracks)
            return

        target = manager.target_for_track(track.track_id)
        if target is None:
            LOGGER.info(
                "TARGET_OPERATION_IGNORED mode=%s track=%d reason=not_session_target",
                mode.name,
                track.track_id,
            )
            return
        if mode == EditMode.REMOVE_TARGETS:
            if is_person:
                person_gallery.detach_session_target(target.target_id)
            else:
                vehicle_gallery_service.detach_session_target(target.target_id)
            manager.deselect(track)
            recognition.notify_gallery_changed()
            LOGGER.info(
                "TARGET_REMOVED domain=%s target=%d track=%d",
                "person" if is_person else "vehicle",
                target.target_id,
                track.track_id,
            )
            return
        if mode == EditMode.ENROLL_GALLERY:
            existing = (
                gallery.person_for_session_target(target.target_id)
                if is_person
                else gallery.vehicle_for_session_target(target.target_id)
            )
            try:
                identity = service.enroll(target)
            except Exception:
                LOGGER.exception(
                    "GALLERY_ENROLL_FAILED domain=%s target=%d track=%d",
                    "person" if is_person else "vehicle",
                    target.target_id,
                    track.track_id,
                )
                return
            label = (
                format_person_id(identity.person_id)
                if is_person
                else vehicle_format_id(identity.vehicle_id)
            )
            LOGGER.info(
                "GALLERY_ENROLLED domain=%s identity=%s target=%d track=%d "
                "already_enrolled=%s",
                "person" if is_person else "vehicle",
                label,
                target.target_id,
                track.track_id,
                existing is not None,
            )
            recognition.notify_gallery_changed()

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
            tracking_stats_before = tracking_pipeline.stats()
            tracking_result = tracking_pipeline.process(frame)
            tracking_stats_after = tracking_pipeline.stats()
            person_tracks = tracking_result.person_tracks
            vehicle_tracks = tracking_result.vehicle_tracks
            tracks = person_tracks
            tracking_seconds = perf_counter() - tracking_started
            diagnostics.observe_tracks(person_tracks, current_frame_index)

            person_recovery_started = perf_counter()
            person_recovery.process_frame(frame, person_tracks, current_frame_index)
            person_recovery_seconds = perf_counter() - person_recovery_started
            person_recovery_stats = person_recovery.last_frame_recovery_stats
            person_recovery_reid_ms += person_recovery_stats.reid_ms
            person_recovery_batches += person_recovery_stats.reid_batch_count
            person_recovery_processed += person_recovery_stats.sweep_processed_this_frame

            vehicle_recovery_started = perf_counter()
            vehicle_recovery.process_frame(frame, vehicle_tracks, current_frame_index)
            vehicle_recovery_seconds = perf_counter() - vehicle_recovery_started
            vehicle_recovery_stats = vehicle_recovery.last_frame_recovery_stats
            vehicle_recovery_reid_ms += vehicle_recovery_stats.reid_ms
            vehicle_recovery_batches += vehicle_recovery_stats.reid_batch_count
            vehicle_recovery_processed += vehicle_recovery_stats.sweep_processed_this_frame
            last_vehicle_recovered_track_ids = vehicle_recovery.last_recovered_track_ids

            gallery_started = perf_counter()
            person_gallery_service.update_runtime_state(
                person_target_manager.targets.values(), current_frame_index
            )
            person_gallery_service.enrich_reference_updates(
                person_recovery.drain_reference_updates()
            )
            vehicle_gallery_service.update_runtime_state(
                vehicle_target_manager.targets.values(), current_frame_index
            )
            vehicle_gallery_service.enrich_reference_updates(
                vehicle_recovery.drain_reference_updates()
            )

            person_gallery_recognition.process_frame(
                frame,
                person_tracks,
                current_frame_index,
                protected_track_ids=person_recovery.last_recovered_track_ids,
            )
            person_recognition_stats = person_gallery_recognition.last_frame_recognition_stats
            person_recognition_reid_ms += person_recognition_stats.reid_ms
            person_recognition_batches += person_recognition_stats.reid_batch_count
            person_recognition_processed += person_recognition_stats.sweep_processed_this_frame
            person_retry_skipped += person_recognition_stats.skipped_retry_cooldown
            person_new_embeddings_total += (
                person_recovery_stats.total_new_embeddings
                + person_recognition_stats.total_new_embeddings
            )
            person_new_embeddings_max = max(
                person_new_embeddings_max,
                person_recovery_stats.total_new_embeddings
                + person_recognition_stats.total_new_embeddings,
            )
            person_budget_deferred += (
                person_recovery_stats.deferred_by_budget
                + person_recognition_stats.deferred_by_budget
            )

            recognized_vehicles = vehicle_gallery_recognition.process_frame(
                frame,
                vehicle_tracks,
                current_frame_index,
                protected_track_ids=last_vehicle_recovered_track_ids,
            )
            for match in recognized_vehicles:
                target = vehicle_target_manager.target_for_track(
                    match.candidate.track.track_id
                )
                if target is not None:
                    vehicle_gallery_service.mark_auto_recognized(target.target_id)
            vehicle_recognition_stats = vehicle_gallery_recognition.last_frame_recognition_stats
            vehicle_recognition_reid_ms += vehicle_recognition_stats.reid_ms
            vehicle_recognition_batches += vehicle_recognition_stats.reid_batch_count
            vehicle_recognition_processed += vehicle_recognition_stats.sweep_processed_this_frame
            vehicle_retry_skipped += vehicle_recognition_stats.skipped_retry_cooldown
            vehicle_new_embeddings_total += (
                vehicle_recovery_stats.total_new_embeddings
                + vehicle_recognition_stats.total_new_embeddings
            )
            vehicle_new_embeddings_max = max(
                vehicle_new_embeddings_max,
                vehicle_recovery_stats.total_new_embeddings
                + vehicle_recognition_stats.total_new_embeddings,
            )
            vehicle_budget_deferred += (
                vehicle_recovery_stats.deferred_by_budget
                + vehicle_recognition_stats.deferred_by_budget
            )
            gallery_seconds = perf_counter() - gallery_started

            frame_index += 1
            render_started = perf_counter()
            annotated = render_frames(frame)
            render_seconds = perf_counter() - render_started
            ui_started = perf_counter()
            action = ui.show(annotated)
            ui_seconds = perf_counter() - ui_started

            frame_seconds = perf_counter() - frame_started
            pc7_elapsed_seconds += frame_seconds
            diagnostics.record_frame(
                frame_seconds,
                current_frame_index,
                tracking_seconds=tracking_seconds,
                recovery_seconds=person_recovery_seconds + vehicle_recovery_seconds,
                gallery_seconds=gallery_seconds,
                render_seconds=render_seconds,
                ui_seconds=ui_seconds,
                recovery_due=person_recovery_stats.recovery_due,
                recovery_candidate_count=person_recovery_stats.candidate_count,
                recovery_quality_valid_count=person_recovery_stats.quality_valid_count,
                recovery_reid_batch_count=person_recovery_stats.reid_batch_count,
                recovery_reid_seconds=person_recovery_stats.reid_ms / 1000.0,
                recovery_sweep_started=person_recovery_stats.sweep_started,
                recovery_sweep_completed=person_recovery_stats.sweep_completed,
                recovery_sweep_candidate_total=person_recovery_stats.sweep_candidate_total,
                recovery_sweep_processed_this_frame=person_recovery_stats.sweep_processed_this_frame,
                recovery_sweep_frames=person_recovery_stats.sweep_frames,
                recovery_sweep_reid_ms=person_recovery_stats.sweep_reid_ms,
                recognition_reid_batch_count=person_recognition_stats.reid_batch_count,
                recognition_reid_seconds=person_recognition_stats.reid_ms / 1000.0,
                recognition_sweep_started=person_recognition_stats.sweep_started,
                recognition_sweep_completed=person_recognition_stats.sweep_completed,
                recognition_sweep_candidate_total=person_recognition_stats.sweep_candidate_total,
                recognition_sweep_processed_this_frame=person_recognition_stats.sweep_processed_this_frame,
                recognition_sweep_frames=person_recognition_stats.sweep_frames,
                recognition_sweep_reid_ms=person_recognition_stats.sweep_reid_ms,
                recognition_retry_skipped=person_recognition_stats.skipped_retry_cooldown,
                reid_breakdown={
                    "yolo_ms": tracking_stats_after.yolo_ms_total
                    - tracking_stats_before.yolo_ms_total,
                    "person_tracker_ms": tracking_stats_after.person_tracker_ms_total
                    - tracking_stats_before.person_tracker_ms_total,
                    "vehicle_tracker_ms": tracking_stats_after.vehicle_tracker_ms_total
                    - tracking_stats_before.vehicle_tracker_ms_total,
                    "person_recovery_candidate_ms": person_recovery_stats.candidate_reid_ms,
                    "person_recovery_revalidation_ms": person_recovery_stats.revalidation_reid_ms,
                    "person_reference_update_ms": person_recovery_stats.reference_update_reid_ms,
                    "person_recognition_candidate_ms": person_recognition_stats.candidate_reid_ms,
                    "person_recognition_revalidation_ms": person_recognition_stats.revalidation_reid_ms,
                    "vehicle_recovery_candidate_ms": vehicle_recovery_stats.candidate_reid_ms,
                    "vehicle_recovery_revalidation_ms": vehicle_recovery_stats.revalidation_reid_ms,
                    "vehicle_reference_update_ms": vehicle_recovery_stats.reference_update_reid_ms,
                    "vehicle_recognition_candidate_ms": vehicle_recognition_stats.candidate_reid_ms,
                    "vehicle_recognition_revalidation_ms": vehicle_recognition_stats.revalidation_reid_ms,
                    "person_recovery_candidate_new_embeddings": person_recovery_stats.candidate_new_embeddings,
                    "person_recovery_revalidation_new_embeddings": person_recovery_stats.revalidation_new_embeddings,
                    "person_reference_update_new_embeddings": person_recovery_stats.reference_update_new_embeddings,
                    "person_recognition_candidate_new_embeddings": person_recognition_stats.candidate_new_embeddings,
                    "person_recognition_revalidation_new_embeddings": person_recognition_stats.revalidation_new_embeddings,
                    "vehicle_recovery_candidate_new_embeddings": vehicle_recovery_stats.candidate_new_embeddings,
                    "vehicle_recovery_revalidation_new_embeddings": vehicle_recovery_stats.revalidation_new_embeddings,
                    "vehicle_reference_update_new_embeddings": vehicle_recovery_stats.reference_update_new_embeddings,
                    "vehicle_recognition_candidate_new_embeddings": vehicle_recognition_stats.candidate_new_embeddings,
                    "vehicle_recognition_revalidation_new_embeddings": vehicle_recognition_stats.revalidation_new_embeddings,
                    "person_reid_cache_hits": person_recovery_stats.cache_hits
                    + person_recognition_stats.cache_hits,
                    "vehicle_reid_cache_hits": vehicle_recovery_stats.cache_hits
                    + vehicle_recognition_stats.cache_hits,
                    "person_recovery_deferred_by_budget": person_recovery_stats.deferred_by_budget,
                    "person_recognition_deferred_by_budget": person_recognition_stats.deferred_by_budget,
                    "vehicle_recovery_deferred_by_budget": vehicle_recovery_stats.deferred_by_budget,
                    "vehicle_recognition_deferred_by_budget": vehicle_recognition_stats.deferred_by_budget,
                    "render_ms": render_seconds * 1000.0,
                    "ui_ms": ui_seconds * 1000.0,
                    "person_new_embeddings": person_recovery_stats.total_new_embeddings
                    + person_recognition_stats.total_new_embeddings,
                    "vehicle_new_embeddings": vehicle_recovery_stats.total_new_embeddings
                    + vehicle_recognition_stats.total_new_embeddings,
                    "person_budget_deferred": person_recovery_stats.deferred_by_budget
                    + person_recognition_stats.deferred_by_budget,
                    "vehicle_budget_deferred": vehicle_recovery_stats.deferred_by_budget
                    + vehicle_recognition_stats.deferred_by_budget,
                },
            )

            if action == UIAction.QUIT:
                LOGGER.info("USER_QUIT key=q")
                break
            if action == UIAction.CLEAR_TARGETS:
                person_ids = tuple(person_target_manager.targets)
                person_gallery.detach_all_session_targets(person_ids)
                person_target_manager.clear()
                person_gallery_recognition.notify_gallery_changed()
                vehicle_ids = tuple(vehicle_target_manager.targets)
                vehicle_gallery_service.detach_all_session_targets(vehicle_ids)
                vehicle_target_manager.clear()
                vehicle_gallery_recognition.notify_gallery_changed()
                vehicle_count = len(vehicle_ids)
                LOGGER.info("TARGETS_CLEARED persons=%d vehicles=%d", len(person_ids), vehicle_count)
                continue

            if action in (UIAction.SELECT_TARGET, UIAction.REMOVE_TARGET, UIAction.ENROLL_GALLERY):
                if action == UIAction.REMOVE_TARGET:
                    edit_tracks = tuple(
                        track
                        for track in (*person_tracks, *vehicle_tracks)
                        if (
                            person_target_manager.target_for_track(track.track_id) is not None
                            if track.class_id == config.model.person_class_id
                            else vehicle_target_manager.target_for_track(track.track_id) is not None
                        )
                    )
                else:
                    edit_tracks = tuple((*person_tracks, *vehicle_tracks))
                edit_mode = {
                    UIAction.SELECT_TARGET: EditMode.ADD_TARGETS,
                    UIAction.REMOVE_TARGET: EditMode.REMOVE_TARGETS,
                    UIAction.ENROLL_GALLERY: EditMode.ENROLL_GALLERY,
                }[action]
                edit_action = ui.run_edit_session(
                    frame=frame,
                    tracks=edit_tracks,
                    mode=edit_mode,
                    on_roi=handle_roi,
                    render_frame=lambda frozen_frame, frozen: render_frames(frozen_frame, frozen),
                )
                if edit_action == UIAction.QUIT:
                    LOGGER.info("USER_QUIT key=q edit_mode=%s", edit_mode.name)
                    break
    finally:
        source.release()
        ui.close()
        diagnostics.log_summary(
            target_lost=person_target_manager.target_lost_count,
            target_recovered=person_target_manager.target_recovered_count,
            recovery_attempted=person_recovery.recovery_attempted_count,
            recovery_pending=person_recovery.recovery_pending_count,
            recovery_accepted=person_recovery.recovery_accepted_count,
            quality_rejected=(person_recovery.quality_rejected_count + person_gallery_recognition.quality_rejected_count),
            gallery_recognized=person_gallery_recognition.recognized_count,
        )
        stats = tracking_pipeline.stats()
        LOGGER.info(
            "%s_STATS frames=%d average_fps=%.2f yolo_inference_count=%d "
            "unique_person_tracks=%d unique_vehicle_tracks=%d yolo_ms=%.2f "
            "person_tracker_ms=%.2f vehicle_tracker_ms=%.2f "
            "person_recovery_reid_ms=%.2f person_recovery_batches=%d "
            "person_recovery_processed=%d person_recognition_reid_ms=%.2f "
            "person_recognition_batches=%d person_recognition_processed=%d "
            "person_retry_skipped=%d vehicle_recovery_reid_ms=%.2f "
            "vehicle_recovery_batches=%d vehicle_recovery_processed=%d "
            "vehicle_recognition_reid_ms=%.2f vehicle_recognition_batches=%d "
                "vehicle_recognition_processed=%d vehicle_retry_skipped=%d",
            "PC7" if is_torch_pc else "ATLAS2B",
            stats.frames,
            stats.frames / pc7_elapsed_seconds
            if pc7_elapsed_seconds > 0.0
            else 0.0,
            stats.yolo_inference_count,
            stats.unique_person_tracks,
            stats.unique_vehicle_tracks,
            stats.yolo_ms_total / max(1, stats.frames),
            stats.person_tracker_ms_total / max(1, stats.frames),
            stats.vehicle_tracker_ms_total / max(1, stats.frames),
            person_recovery_reid_ms,
            person_recovery_batches,
            person_recovery_processed,
            person_recognition_reid_ms,
            person_recognition_batches,
            person_recognition_processed,
            person_retry_skipped,
            vehicle_recovery_reid_ms,
            vehicle_recovery_batches,
            vehicle_recovery_processed,
            vehicle_recognition_reid_ms,
            vehicle_recognition_batches,
            vehicle_recognition_processed,
            vehicle_retry_skipped,
        )
        LOGGER.info(
            "REID_DOMAIN_STATS person_new_embeddings=%d person_max_new_embeddings=%d "
            "person_budget_deferred=%d vehicle_new_embeddings=%d "
            "vehicle_max_new_embeddings=%d vehicle_budget_deferred=%d",
            person_new_embeddings_total,
            person_new_embeddings_max,
            person_budget_deferred,
            vehicle_new_embeddings_total,
            vehicle_new_embeddings_max,
            vehicle_budget_deferred,
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
