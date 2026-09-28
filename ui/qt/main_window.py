"""Formal Qt Widgets main window for MVP-10."""

from __future__ import annotations

from PyQt5.QtCore import QPoint, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget

from src.config import AppConfig
from .gallery_page import GalleryPage
from .models import GalleryRecordDTO, RuntimeFrame, SelectionResultDTO
from .runtime_worker import RuntimeWorker
from .widgets.video_canvas import VideoCanvas


class TargetDecisionPopup(QFrame):
    saveRequested = pyqtSignal()
    discardRequested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.Tool)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        self.save_button = QPushButton("\u4fdd\u5b58\u76ee\u6807")
        self.discard_button = QPushButton("\u4e0d\u4fdd\u5b58")
        layout.addWidget(self.save_button)
        layout.addWidget(self.discard_button)
        self.save_button.clicked.connect(self.saveRequested.emit)
        self.discard_button.clicked.connect(self.discardRequested.emit)


class QtMainWindow(QMainWindow):
    startEditRequested = pyqtSignal(str)
    submitRoiRequested = pyqtSignal(object)
    finishEditRequested = pyqtSignal()
    savePendingRequested = pyqtSignal()
    discardPendingRequested = pyqtSignal()
    deleteGalleryRequested = pyqtSignal(str, object, bool)
    shutdownRequested = pyqtSignal()

    def __init__(self, config: AppConfig, source_override: str | None = None, parent=None, *, start_worker: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("Person + Vehicle Tracking System")
        self._shutdown_requested = False
        self._popup: TargetDecisionPopup | None = None
        self._last_frame: RuntimeFrame | None = None
        self._edit_mode: str | None = None
        self._edit_request_mode: str | None = None

        self.stack = QStackedWidget(self)
        self.main_page = QWidget()
        self.gallery_page = GalleryPage()
        self.stack.addWidget(self.main_page)
        self.stack.addWidget(self.gallery_page)
        self._build_main_page()
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.thread = QThread(self)
        self.worker = RuntimeWorker(config, source_override)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.worker.runtime_ready.connect(self._runtime_ready)
        self.worker.frame_ready.connect(self._frame_ready)
        self.worker.selection_ready.connect(self._selection_ready)
        self.worker.selection_failed.connect(self._selection_failed)
        self.worker.edit_started.connect(self._edit_started)
        self.worker.edit_finished.connect(self._edit_finished)
        self.worker.gallery_rows_ready.connect(self._gallery_rows_ready)
        self.worker.gallery_mutation_done.connect(self._gallery_mutation_done)
        self.worker.status_message.connect(self._set_status)
        self.worker.fatal_error.connect(self._fatal_error)
        self.worker.shutdown_complete.connect(self._worker_shutdown_complete)
        self.startEditRequested.connect(self.worker.start_edit)
        self.submitRoiRequested.connect(self.worker.submit_roi)
        self.finishEditRequested.connect(self.worker.finish_edit)
        self.savePendingRequested.connect(self.worker.save_pending_selection)
        self.discardPendingRequested.connect(self.worker.discard_pending_selection)
        self.deleteGalleryRequested.connect(self.worker.delete_gallery)
        self.shutdownRequested.connect(self.worker.request_shutdown)
        self._update_edit_controls()
        if start_worker:
            self.thread.start()

    def _build_main_page(self) -> None:
        root = QVBoxLayout(self.main_page)
        root.setContentsMargins(16, 12, 16, 12)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        self.select_button = QPushButton("\u6846\u9009")
        self.remove_button = QPushButton("\u64a4\u9500")
        self.database_button = QPushButton("\u6570\u636e\u5e93")
        self.quit_button = QPushButton("\u9000\u51fa")
        for button in (self.select_button, self.remove_button, self.database_button, self.quit_button):
            button.setMinimumHeight(56)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            toolbar.addWidget(button, 1)
        self.select_button.setCheckable(True)
        self.remove_button.setCheckable(True)
        active_style = "QPushButton:checked { background-color: #2f75b5; color: white; }"
        self.select_button.setStyleSheet(active_style)
        self.remove_button.setStyleSheet(active_style)
        root.addLayout(toolbar)
        self.video_canvas = VideoCanvas()
        root.addWidget(self.video_canvas, 1)
        self.status_label = QLabel("\u6b63\u5728\u542f\u52a8...")
        self.status_label.setMinimumHeight(28)
        root.addWidget(self.status_label)
        self.select_button.clicked.connect(lambda _checked=False: self._toggle_edit("select"))
        self.remove_button.clicked.connect(lambda _checked=False: self._toggle_edit("remove"))
        self.database_button.clicked.connect(self._open_gallery)
        self.quit_button.clicked.connect(self._confirm_quit)
        self.video_canvas.roiSelected.connect(self.submitRoiRequested.emit)
        self.gallery_page.backRequested.connect(self._back_to_tracking)
        self.gallery_page.deleteRequested.connect(self._confirm_gallery_delete)

    def _runtime_ready(self) -> None:
        self._set_status("\u8fd0\u884c\u4e2d")

    def _frame_ready(self, runtime_frame: RuntimeFrame) -> None:
        self._last_frame = runtime_frame
        self.video_canvas.set_frame(runtime_frame.annotated_bgr)
        if self._edit_mode is None and self._edit_request_mode is None:
            self._set_status(
                f"{runtime_frame.backend} | {runtime_frame.source_label} | FPS {runtime_frame.fps:.1f} | "
                f"P {runtime_frame.person_active_count}/{runtime_frame.person_lost_count} "
                f"G{runtime_frame.person_gallery_count} | "
                f"V {runtime_frame.vehicle_active_count}/{runtime_frame.vehicle_lost_count} "
                f"G{runtime_frame.vehicle_gallery_count}"
            )

    def _toggle_edit(self, mode: str) -> None:
        if mode not in {"select", "remove"}:
            return
        if self._edit_mode is None:
            if self._popup is not None:
                return
            if self._edit_request_mode is not None:
                return
            self._edit_request_mode = mode
            self._update_edit_controls()
            self.startEditRequested.emit(mode)
            return
        if self._edit_mode != mode:
            return
        self._request_finish_edit()

    def _begin_edit(self, mode: str) -> None:
        """Backward-compatible internal alias for the mouse toggle action."""

        self._toggle_edit(mode)

    def _request_finish_edit(self) -> None:
        if self._popup is not None:
            self._set_status("\u8bf7\u5148\u9009\u62e9 \u4fdd\u5b58\u76ee\u6807 \u6216 \u4e0d\u4fdd\u5b58")
            self._update_edit_controls()
            return
        if self._edit_mode is not None:
            self.finishEditRequested.emit()
            self._update_edit_controls()

    def _edit_started(self, mode: str) -> None:
        self._edit_request_mode = None
        self._edit_mode = mode
        text = (
            "\u6846\u9009\u6a21\u5f0f\uff1a\u62d6\u52a8\u9f20\u6807\u9009\u62e9\u76ee\u6807\uff0c\u518d\u6b21\u70b9\u51fb\u201c\u5b8c\u6210\u6846\u9009\u201d\u7ed3\u675f"
            if mode == "select"
            else "\u64a4\u9500\u6a21\u5f0f\uff1a\u62d6\u52a8\u9f20\u6807\u9009\u62e9\u76ee\u6807\uff0c\u518d\u6b21\u70b9\u51fb\u201c\u5b8c\u6210\u64a4\u9500\u201d\u7ed3\u675f"
        )
        self.video_canvas.begin_edit(text)
        self._update_edit_controls()

    def _edit_finished(self) -> None:
        self.video_canvas.end_edit()
        self._edit_mode = None
        self._edit_request_mode = None
        self._update_edit_controls()

    def _selection_ready(self, result: SelectionResultDTO) -> None:
        self._set_status(f"\u5df2\u9009\u62e9 {result.domain} T{result.track_id}\uff0c\u8bf7\u9009\u62e9\u662f\u5426\u4fdd\u5b58")
        self._close_popup()
        self.video_canvas.setEnabled(False)
        self._popup = TargetDecisionPopup(self)
        self._popup.saveRequested.connect(self._save_popup_target)
        self._popup.discardRequested.connect(self._discard_popup_target)
        transform = self.video_canvas.transform
        if transform is not None:
            rect = transform.source_roi_to_widget(result.bbox)
            self._popup.adjustSize()
            anchor = self.video_canvas.mapToGlobal(
                QPoint(int(round(rect.right())), int(round(rect.bottom())))
            )
            screen = QApplication.screenAt(anchor) or self.screen()
            available = screen.availableGeometry()
            x = min(max(available.left(), anchor.x()), available.right() - self._popup.width())
            y = min(max(available.top(), anchor.y()), available.bottom() - self._popup.height())
            self._popup.move(x, y)
        self._popup.show()

    def _save_popup_target(self) -> None:
        self._close_popup()
        self.savePendingRequested.emit()

    def _discard_popup_target(self) -> None:
        self._close_popup()
        self.discardPendingRequested.emit()

    def _close_popup(self) -> None:
        if self._popup is not None:
            self._popup.close()
            self._popup.deleteLater()
            self._popup = None
        self.video_canvas.setEnabled(True)

    def _selection_failed(self, message: str) -> None:
        if self._edit_request_mode is not None and self._edit_mode is None:
            self._edit_request_mode = None
            self._update_edit_controls()
        self._set_status(message)

    def _update_edit_controls(self) -> None:
        active = self._edit_mode
        pending = self._edit_request_mode is not None
        select_active = active == "select"
        remove_active = active == "remove"
        self.select_button.setChecked(select_active)
        self.remove_button.setChecked(remove_active)
        self.select_button.setText("\u5b8c\u6210\u6846\u9009" if select_active else "\u6846\u9009")
        self.remove_button.setText("\u5b8c\u6210\u64a4\u9500" if remove_active else "\u64a4\u9500")
        self.select_button.setEnabled(not pending and not remove_active)
        self.remove_button.setEnabled(not pending and not select_active)
        self.database_button.setEnabled(not pending and active is None)
        self.quit_button.setEnabled(True)

    def _open_gallery(self) -> None:
        self.stack.setCurrentWidget(self.gallery_page)

    def _back_to_tracking(self) -> None:
        self.stack.setCurrentWidget(self.main_page)

    def _gallery_rows_ready(self, records: tuple[GalleryRecordDTO, ...]) -> None:
        self.gallery_page.set_records(records)

    def _confirm_gallery_delete(self, domain: str, ids, batch: bool) -> None:
        values = [int(ids)] if isinstance(ids, int) else [int(value) for value in ids]
        if not values:
            return
        title = "\u6279\u91cf\u5220\u9664" if batch else "\u5220\u9664\u76ee\u6807"
        text = f"\u786e\u8ba4\u5220\u9664 {len(values)} \u4e2a\u6301\u4e45\u8eab\u4efd\u5417?\nSessionTarget \u4e0d\u4f1a\u505c\u6b62\u8ddf\u8e2a."
        if QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self.deleteGalleryRequested.emit(domain, values, batch)

    def _gallery_mutation_done(self, message: str) -> None:
        self._set_status(message)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _fatal_error(self, message: str) -> None:
        self._set_status(f"\u8fd0\u884c\u5931\u8d25: {message}")
        QMessageBox.critical(self, "\u8fd0\u884c\u5931\u8d25", message)
        self._request_shutdown()

    def _confirm_quit(self) -> None:
        if QMessageBox.question(self, "\u9000\u51fa", "\u786e\u8ba4\u9000\u51fa\u7a0b\u5e8f\u5417?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self._request_shutdown()

    def _request_shutdown(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self.shutdownRequested.emit()

    def _worker_shutdown_complete(self) -> None:
        self.thread.quit()
        self.thread.wait(5000)
        QApplication.instance().quit()

    def closeEvent(self, event) -> None:
        if self._shutdown_requested:
            event.accept()
            return
        if QMessageBox.question(self, "\u9000\u51fa", "\u786e\u8ba4\u9000\u51fa\u7a0b\u5e8f\u5417?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self._request_shutdown()
            event.accept()
        else:
            event.ignore()


MainWindow = QtMainWindow


def launch_qt(config: AppConfig, source_override: str | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    window = QtMainWindow(config, source_override)
    window.showFullScreen()
    return app.exec_()
