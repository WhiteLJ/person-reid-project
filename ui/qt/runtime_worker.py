"""Single-threaded application runtime owned by a Qt worker thread."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from logging import getLogger
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from src.active_identity_guard import ActiveIdentityGuard
from src.ascend_runtime import AscendRuntime
from src.config import AppConfig, parse_source
from src.database import GalleryRepository
from src.frame_source import FrameSource
from src.gallery import TargetGallery, format_person_id
from src.gallery_recognition import GalleryRecognitionCoordinator
from src.gallery_service import GalleryPersistenceService
from src.models import Track
from src.reid import ReIDExtractor
from src.reid_frame_cache import ReIDFrameCache
from src.roi_selector import find_track_by_roi, roi_xyxy_to_xywh
from src.source_factory import create_frame_source
from src.target_manager import TargetManager
from src.target_recovery import TargetRecoveryCoordinator
from src.vehicle_gallery import VehicleTargetGallery, format_vehicle_id
from src.vehicle_gallery_recognition import VehicleGalleryRecognitionCoordinator
from src.vehicle_gallery_service import VehicleGalleryPersistenceService
from src.vehicle_recovery import VehicleRecoveryCoordinator
from src.visualization import draw_multiclass_tracks
from ui.qt.models import GalleryRecordDTO, RuntimeFrame, SelectionResultDTO


LOGGER = getLogger(__name__)


class RuntimeWorker(QObject):
    """Own all mutable runtime/business objects outside the GUI thread."""

    runtime_ready = pyqtSignal()
    frame_ready = pyqtSignal(object)
    selection_ready = pyqtSignal(object)
    selection_failed = pyqtSignal(str)
    edit_started = pyqtSignal(str)
    edit_finished = pyqtSignal()
    gallery_rows_ready = pyqtSignal(object)
    gallery_mutation_done = pyqtSignal(str)
    status_message = pyqtSignal(str)
    fatal_error = pyqtSignal(str)
    shutdown_complete = pyqtSignal()

    def __init__(self, config: AppConfig, source_override: str | None = None) -> None:
        super().__init__()
        self.config = config
        if source_override is not None:
            self.config = replace(
                config, video=replace(config.video, source=parse_source(source_override))
            )
        self._timer: QTimer | None = None
        self._source: FrameSource | None = None
        self._ascend_runtime: AscendRuntime | None = None
        self._tracking_pipeline: Any = None
        self._reid_extractor: Any = None
        self._vehicle_reid_extractor: Any = None
        self._person_manager: TargetManager | None = None
        self._vehicle_manager: TargetManager | None = None
        self._person_gallery: TargetGallery | None = None
        self._vehicle_gallery: VehicleTargetGallery | None = None
        self._person_service: GalleryPersistenceService | None = None
        self._vehicle_service: VehicleGalleryPersistenceService | None = None
        self._person_repository: GalleryRepository | None = None
        self._vehicle_repository: Any = None
        self._person_recovery: TargetRecoveryCoordinator | None = None
        self._vehicle_recovery: VehicleRecoveryCoordinator | None = None
        self._person_recognition: GalleryRecognitionCoordinator | None = None
        self._vehicle_recognition: VehicleGalleryRecognitionCoordinator | None = None
        self._person_guard: ActiveIdentityGuard | None = None
        self._person_cache = ReIDFrameCache()
        self._vehicle_cache = ReIDFrameCache()
        self._vehicle_class_ids: tuple[int, ...] = ()
        self._frame_index = -1
        self._current_frame: np.ndarray | None = None
        self._person_tracks: tuple[Track, ...] = ()
        self._vehicle_tracks: tuple[Track, ...] = ()
        self._last_annotated: np.ndarray | None = None
        self._last_fps = 0.0
        self._frame_times: deque[float] = deque(maxlen=30)
        self._paused = False
        self._editing = False
        self._edit_mode: str | None = None
        self._frozen_frame: np.ndarray | None = None
        self._frozen_tracks: tuple[Track, ...] = ()
        self._pending_selection: tuple[str, Any, Track, np.ndarray] | None = None
        self._closing = False

    @pyqtSlot()
    def start(self) -> None:
        try:
            self._initialize_runtime()
            self._source = create_frame_source(self.config)
            self._source.open()
            self._timer = QTimer(self)
            self._timer.setInterval(1)
            self._timer.timeout.connect(self.process_next_frame)
            self._timer.start()
            self._publish_gallery_rows()
            self.runtime_ready.emit()
            self.status_message.emit("运行中")
        except Exception as exc:
            LOGGER.exception("QT_RUNTIME_START_FAILED")
            self.fatal_error.emit(str(exc))

    def _initialize_runtime(self) -> None:
        config = self.config
        is_torch_pc = config.inference.backend == "torch"
        person_repository = GalleryRepository(config.database.path)
        person_repository.initialize()
        person_gallery = TargetGallery()
        person_service = GalleryPersistenceService(
            person_gallery, person_repository, enrichment_config=config.gallery_enrichment
        )
        person_service.load()

        person_manager = TargetManager()
        self._person_repository = person_repository
        self._person_gallery = person_gallery
        self._person_service = person_service
        self._person_manager = person_manager

        try:
            if is_torch_pc:
                from src.pc_multiclass_tracking import MultiClassTrackingPipeline
                from src.vehicle_reid import VehicleReIDExtractor

                self._tracking_pipeline = MultiClassTrackingPipeline(config)
                self._reid_extractor = ReIDExtractor(
                    config.reid, config.model.device, backend="torch"
                )
                self._vehicle_reid_extractor = VehicleReIDExtractor(config.vehicle_reid)
            else:
                from src.ascend_multiclass_tracking import AscendMultiClassTrackingPipeline
                from src.ascend_vehicle_reid import AscendVehicleReIDExtractor

                self._ascend_runtime = AscendRuntime(config.ascend.device_id)
                self._tracking_pipeline = AscendMultiClassTrackingPipeline(
                    config, self._ascend_runtime
                )
                self._reid_extractor = ReIDExtractor(
                    config.reid,
                    config.model.device,
                    backend=config.inference.backend,
                    ascend_config=config.ascend,
                    ascend_runtime=self._ascend_runtime,
                )
                self._vehicle_reid_extractor = AscendVehicleReIDExtractor(
                    config.vehicle_reid,
                    config.ascend.vehicle_reid_model,
                    config.ascend.vehicle_reid_dynamic_batches,
                    self._ascend_runtime,
                )

            from src.vehicle_database import VehicleGalleryRepository

            self._vehicle_class_ids = tuple(config.multiclass_tracking.vehicle_class_ids)
            vehicle_manager = TargetManager()
            vehicle_repository = VehicleGalleryRepository(config.vehicle_database.path)
            vehicle_repository.initialize()
            vehicle_gallery = VehicleTargetGallery()
            vehicle_service = VehicleGalleryPersistenceService(
                vehicle_gallery,
                vehicle_repository,
                enrichment_config=config.vehicle_gallery_enrichment,
            )
            vehicle_service.load()
            self._vehicle_manager = vehicle_manager
            self._vehicle_repository = vehicle_repository
            self._vehicle_gallery = vehicle_gallery
            self._vehicle_service = vehicle_service
        except Exception:
            self._close_runtime()
            raise

        self._person_recovery = TargetRecoveryCoordinator(
            target_manager=person_manager,
            reid_extractor=self._reid_extractor,
            reid_config=config.reid,
            recovery_config=config.reid_recovery,
            embedding_cache=self._person_cache,
            quality_config=config.reid_quality,
            person_class_id=config.model.person_class_id,
        )
        self._person_recognition = GalleryRecognitionCoordinator(
            target_manager=person_manager,
            gallery=person_gallery,
            reid_extractor=self._reid_extractor,
            reid_config=config.reid,
            recognition_config=config.gallery_recognition,
            recovery_config=config.reid_recovery,
            person_class_id=config.model.person_class_id,
            embedding_cache=self._person_cache,
            quality_config=config.reid_quality,
        )
        self._person_guard = ActiveIdentityGuard(
            target_manager=person_manager,
            reid_extractor=self._reid_extractor,
            reid_config=config.reid,
            recovery_config=config.reid_recovery,
            quality_config=config.reid_quality,
            guard_config=config.active_identity_guard,
            embedding_cache=self._person_cache,
            person_class_id=config.model.person_class_id,
        )
        self._vehicle_recovery = VehicleRecoveryCoordinator(
            target_manager=self._vehicle_manager,
            reid_extractor=self._vehicle_reid_extractor,
            reid_config=config.vehicle_reid,
            recovery_config=config.vehicle_recovery,
            quality_config=config.vehicle_reid_quality,
            vehicle_class_ids=self._vehicle_class_ids,
            embedding_cache=self._vehicle_cache,
        )
        self._vehicle_recognition = VehicleGalleryRecognitionCoordinator(
            target_manager=self._vehicle_manager,
            gallery=self._vehicle_gallery,
            reid_extractor=self._vehicle_reid_extractor,
            reid_config=config.vehicle_reid,
            recognition_config=config.vehicle_gallery_recognition,
            recovery_config=config.vehicle_recovery,
            quality_config=config.vehicle_reid_quality,
            vehicle_class_ids=self._vehicle_class_ids,
            embedding_cache=self._vehicle_cache,
        )

    @pyqtSlot()
    def process_next_frame(self) -> None:
        if self._closing or self._paused or self._editing or self._source is None:
            return
        frame_started = perf_counter()
        frame = self._source.read()
        if frame is None:
            self.request_shutdown()
            return
        self._frame_index += 1
        self._current_frame = frame.copy()
        result = self._tracking_pipeline.process(frame)
        self._person_tracks = tuple(result.person_tracks)
        self._vehicle_tracks = tuple(result.vehicle_tracks)

        guard_result = self._person_guard.process_frame(
            frame, self._person_tracks, self._frame_index
        )
        self._person_recovery.process_frame(
            frame,
            self._person_tracks,
            self._frame_index,
            reference_update_blocked_target_ids=guard_result.blocked_reference_update_target_ids,
        )
        self._vehicle_recovery.process_frame(
            frame, self._vehicle_tracks, self._frame_index
        )

        self._person_service.update_runtime_state(
            self._person_manager.targets.values(), self._frame_index
        )
        self._person_service.enrich_reference_updates(
            self._person_recovery.drain_reference_updates()
        )
        self._vehicle_service.update_runtime_state(
            self._vehicle_manager.targets.values(), self._frame_index
        )
        self._vehicle_service.enrich_reference_updates(
            self._vehicle_recovery.drain_reference_updates()
        )
        self._person_recognition.process_frame(
            frame,
            self._person_tracks,
            self._frame_index,
            protected_track_ids=self._person_recovery.last_recovered_track_ids,
        )
        recognized = self._vehicle_recognition.process_frame(
            frame,
            self._vehicle_tracks,
            self._frame_index,
            protected_track_ids=self._vehicle_recovery.last_recovered_track_ids,
        )
        for match in recognized:
            target = self._vehicle_manager.target_for_track(match.candidate.track.track_id)
            if target is not None:
                self._vehicle_service.mark_auto_recognized(target.target_id)

        annotated = self._render(frame)
        elapsed = perf_counter() - frame_started
        self._frame_times.append(elapsed)
        self._last_fps = 1.0 / (sum(self._frame_times) / len(self._frame_times)) if self._frame_times else 0.0
        self._last_annotated = annotated
        self.frame_ready.emit(self._runtime_frame(annotated))

    def _render(self, frame: np.ndarray) -> np.ndarray:
        person_labels = {}
        for target in self._person_manager.active_targets():
            if target.current_track_id is None:
                continue
            person = self._person_gallery.person_for_session_target(target.target_id)
            if person is not None:
                person_labels[target.current_track_id] = format_person_id(person.person_id)
        vehicle_labels = {}
        for target in self._vehicle_manager.active_targets():
            if target.current_track_id is None:
                continue
            vehicle = self._vehicle_gallery.vehicle_for_session_target(target.target_id)
            if vehicle is not None:
                vehicle_labels[target.target_id] = format_vehicle_id(vehicle.vehicle_id)
        return draw_multiclass_tracks(
            frame,
            self._person_tracks,
            self._vehicle_tracks,
            self._person_manager,
            self._vehicle_manager,
            class_name=self._tracking_pipeline.class_name,
            show_class_name=self.config.ui.show_class_name,
            show_track_id=self.config.tracking.show_track_id,
            show_confidence=self.config.ui.show_confidence,
            show_unselected_tracks=self.config.ui.show_unselected_tracks,
            person_gallery_labels_by_track=person_labels,
            vehicle_gallery_labels_by_target=vehicle_labels,
        )

    def _runtime_frame(self, annotated: np.ndarray) -> RuntimeFrame:
        active_people = len(self._person_manager.active_targets())
        active_vehicles = len(self._vehicle_manager.active_targets())
        return RuntimeFrame(
            frame_index=self._frame_index,
            annotated_bgr=annotated.copy(),
            source_width=int(annotated.shape[1]),
            source_height=int(annotated.shape[0]),
            person_tracks=self._person_tracks,
            vehicle_tracks=self._vehicle_tracks,
            person_active_count=active_people,
            person_lost_count=len(self._person_manager.lost_targets()),
            vehicle_active_count=active_vehicles,
            vehicle_lost_count=len(self._vehicle_manager.lost_targets()),
            person_gallery_count=len(self._person_gallery.all_people()),
            vehicle_gallery_count=len(self._vehicle_gallery.all_vehicles()),
            fps=self._last_fps,
            source_label=self._source.source_label,
            backend=self.config.inference.backend.upper(),
        )

    @pyqtSlot(str)
    def start_edit(self, mode: str) -> None:
        if self._current_frame is None or self._editing:
            self.selection_failed.emit("当前没有可编辑的冻结画面")
            return
        if mode not in {"select", "remove"}:
            self.selection_failed.emit("不支持的编辑模式")
            return
        self._editing = True
        self._edit_mode = mode
        self._frozen_frame = self._current_frame.copy()
        visible = (*self._person_tracks, *self._vehicle_tracks)
        if mode == "remove":
            visible = tuple(
                track
                for track in visible
                if (
                    self._person_manager.target_for_track(track.track_id)
                    if track.class_id == self.config.model.person_class_id
                    else self._vehicle_manager.target_for_track(track.track_id)
                )
                is not None
            )
        self._frozen_tracks = tuple(visible)
        self.status_message.emit("\u6846\u9009\u6a21\u5f0f" if mode == "select" else "\u64a4\u9500\u6a21\u5f0f")
        self.edit_started.emit(mode)

    @pyqtSlot()
    def finish_edit(self) -> None:
        self._editing = False
        self._edit_mode = None
        self._frozen_frame = None
        self._frozen_tracks = ()
        self._pending_selection = None
        self.status_message.emit("\u8fd0\u884c\u4e2d")
        self.edit_finished.emit()

    @pyqtSlot()
    def cancel_edit(self) -> None:
        self.finish_edit()

    @pyqtSlot(object)
    def submit_roi(self, roi) -> None:
        if not self._editing or self._frozen_frame is None:
            return
        try:
            roi_xywh = roi_xyxy_to_xywh(roi)
        except ValueError:
            self.selection_failed.emit("\u672a\u5339\u914d\u5230\u76ee\u6807\uff0c\u8bf7\u91cd\u65b0\u6846\u9009")
            return
        track = find_track_by_roi(roi_xywh, self._frozen_tracks, min_iou=self.config.selection.min_iou)
        if track is None:
            self.selection_failed.emit("\u672a\u5339\u914d\u5230\u76ee\u6807\uff0c\u8bf7\u91cd\u65b0\u6846\u9009")
            return
        is_person = track.class_id == self.config.model.person_class_id
        is_vehicle = track.class_id in self._vehicle_class_ids
        if not is_person and not is_vehicle:
            self.selection_failed.emit("目标类别不支持")
            return
        manager = self._person_manager if is_person else self._vehicle_manager
        if self._edit_mode == "remove":
            target = manager.target_for_track(track.track_id)
            if target is None:
                self.selection_failed.emit("只能撤销当前SessionTarget")
                return
            if is_person:
                self._person_gallery.detach_session_target(target.target_id)
                self._person_recognition.notify_gallery_changed()
            else:
                self._vehicle_service.detach_session_target(target.target_id)
                self._vehicle_recognition.notify_gallery_changed()
            manager.deselect(track)
            self.status_message.emit("目标已撤销")
            self._emit_frozen_frame()
            return

        recovery = self._person_recovery if is_person else self._vehicle_recovery
        domain_tracks = self._person_tracks if is_person else self._vehicle_tracks
        target = recovery.select_from_track(
            self._frozen_frame, track, self._frame_index, tracks=domain_tracks
        )
        if target is None:
            self.selection_failed.emit("\u76ee\u6807\u8d28\u91cf\u4e0d\u8db3\uff0c\u8bf7\u91cd\u65b0\u6846\u9009")
            return
        domain = "person" if is_person else "vehicle"
        self._pending_selection = (domain, target, track, self._frozen_frame.copy())
        self.selection_ready.emit(
            SelectionResultDTO(domain, target.target_id, track.track_id, track.bbox)
        )
        self._emit_frozen_frame()

    @pyqtSlot()
    def save_pending_selection(self) -> None:
        if self._pending_selection is None:
            return
        domain, target, track, frozen = self._pending_selection
        try:
            if domain == "person":
                identity = self._person_service.enroll_with_snapshot(target, frozen, track.bbox)
                display_id = format_person_id(identity.person_id)
                self._person_recognition.notify_gallery_changed()
            else:
                identity = self._vehicle_service.enroll_with_snapshot(target, frozen, track.bbox)
                display_id = format_vehicle_id(identity.vehicle_id)
                self._vehicle_recognition.notify_gallery_changed()
            self.status_message.emit(f"已保存 {display_id}")
            self._pending_selection = None
            self._publish_gallery_rows()
            self._emit_frozen_frame()
        except Exception as exc:
            LOGGER.exception("QT_GALLERY_ENROLL_FAILED")
            self.selection_failed.emit(f"保存失败: {exc}")

    @pyqtSlot()
    def discard_pending_selection(self) -> None:
        if self._pending_selection is not None:
            self._pending_selection = None
            self.status_message.emit("目标已保留，未保存到数据库")
            self._emit_frozen_frame()

    @pyqtSlot()
    def clear_session(self) -> None:
        person_ids = tuple(self._person_manager.targets)
        self._person_gallery.detach_all_session_targets(person_ids)
        self._person_manager.clear()
        self._person_recognition.notify_gallery_changed()
        vehicle_ids = tuple(self._vehicle_manager.targets)
        self._vehicle_service.detach_all_session_targets(vehicle_ids)
        self._vehicle_manager.clear()
        self._vehicle_recognition.notify_gallery_changed()
        self.status_message.emit("SessionTarget 已清空")
        self._emit_frozen_frame()

    @pyqtSlot(str, object, bool)
    def delete_gallery(self, domain: str, ids, batch: bool = False) -> None:
        del batch
        values = [int(ids)] if isinstance(ids, int) else [int(value) for value in ids]
        service = self._person_service if domain == "person" else self._vehicle_service
        service.remove_many(values)
        self._publish_gallery_rows()
        self.gallery_mutation_done.emit("数据库已更新")

    @pyqtSlot()
    def pause_toggle(self) -> None:
        self._paused = not self._paused
        self.status_message.emit("已暂停" if self._paused else "运行中")

    @pyqtSlot()
    def request_shutdown(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self._timer is not None:
            self._timer.stop()
        if self._source is not None:
            self._source.release()
        self._close_runtime()
        self.shutdown_complete.emit()

    def _close_runtime(self) -> None:
        if self._ascend_runtime is not None:
            self._ascend_runtime.close()
            self._ascend_runtime = None

    def _emit_frozen_frame(self) -> None:
        if self._frozen_frame is not None:
            annotated = self._render(self._frozen_frame)
            self._last_annotated = annotated
            self.frame_ready.emit(self._runtime_frame(annotated))

    def _publish_gallery_rows(self) -> None:
        people = self._person_repository.load_snapshots()
        vehicles = self._vehicle_repository.load_snapshots()
        rows = []
        for person in self._person_gallery.all_people():
            rows.append(
                GalleryRecordDTO("person", person.person_id, format_person_id(person.person_id), people.get(person.person_id))
            )
        for vehicle in self._vehicle_gallery.all_vehicles():
            rows.append(
                GalleryRecordDTO("vehicle", vehicle.vehicle_id, format_vehicle_id(vehicle.vehicle_id), vehicles.get(vehicle.vehicle_id))
            )
        self.gallery_rows_ready.emit(tuple(rows))
