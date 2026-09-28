"""Aspect-ratio-preserving Qt video canvas with source/display ROI mapping."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import QWidget


@dataclass(frozen=True)
class CanvasTransform:
    source_width: int
    source_height: int
    widget_width: int
    widget_height: int
    scale: float
    offset_x: float
    offset_y: float

    @property
    def display_width(self) -> float:
        return self.source_width * self.scale

    @property
    def display_height(self) -> float:
        return self.source_height * self.scale

    def source_to_widget(self, x: float, y: float) -> tuple[float, float]:
        return self.offset_x + x * self.scale, self.offset_y + y * self.scale

    def widget_to_source(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale

    def source_roi_to_widget(self, roi: tuple[float, float, float, float]) -> QRectF:
        x1, y1 = self.source_to_widget(roi[0], roi[1])
        x2, y2 = self.source_to_widget(roi[2], roi[3])
        return QRectF(x1, y1, x2 - x1, y2 - y1)

    def widget_roi_to_source(
        self, rect: QRectF, source_shape: tuple[int, int, int]
    ) -> tuple[int, int, int, int] | None:
        x1, y1 = self.widget_to_source(rect.left(), rect.top())
        x2, y2 = self.widget_to_source(rect.right(), rect.bottom())
        width, height = source_shape[1], source_shape[0]
        x1 = max(0, min(width, int(round(min(x1, x2)))))
        y1 = max(0, min(height, int(round(min(y1, y2)))))
        x2 = max(0, min(width, int(round(max(x1, x2)))))
        y2 = max(0, min(height, int(round(max(y1, y2)))))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2


class VideoCanvas(QWidget):
    """Draw a source image and optionally collect continuous ROI drags."""

    roiSelected = pyqtSignal(tuple)
    editFinished = pyqtSignal()
    editCancelled = pyqtSignal()
    quitRequested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self._source_bgr: np.ndarray | None = None
        self._image: QImage | None = None
        self._transform: CanvasTransform | None = None
        self._editing = False
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._mode_text = ""

    @property
    def transform(self) -> CanvasTransform | None:
        return self._transform

    @property
    def source_frame(self) -> np.ndarray | None:
        return None if self._source_bgr is None else self._source_bgr.copy()

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("VideoCanvas expects an HxWx3 BGR frame")
        self._source_bgr = np.ascontiguousarray(frame_bgr.copy())
        rgb = cv2.cvtColor(self._source_bgr, cv2.COLOR_BGR2RGB)
        self._image = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            int(rgb.strides[0]),
            QImage.Format_RGB888,
        ).copy()
        self._update_transform()
        self.update()

    def begin_edit(self, mode_text: str) -> None:
        self._editing = True
        self._mode_text = mode_text
        self._drag_start = None
        self._drag_current = None
        self.setFocus(Qt.OtherFocusReason)
        self.update()

    def end_edit(self) -> None:
        self._editing = False
        self._mode_text = ""
        self._drag_start = None
        self._drag_current = None
        self.update()

    def resizeEvent(self, event) -> None:
        self._update_transform()
        super().resizeEvent(event)

    def _update_transform(self) -> None:
        if self._source_bgr is None:
            self._transform = None
            return
        source_height, source_width = self._source_bgr.shape[:2]
        scale = min(self.width() / source_width, self.height() / source_height)
        scale = max(scale, 1e-9)
        display_width = source_width * scale
        display_height = source_height * scale
        self._transform = CanvasTransform(
            source_width,
            source_height,
            self.width(),
            self.height(),
            scale,
            (self.width() - display_width) / 2.0,
            (self.height() - display_height) / 2.0,
        )

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        if self._image is not None and self._transform is not None:
            rect = QRectF(
                self._transform.offset_x,
                self._transform.offset_y,
                self._transform.display_width,
                self._transform.display_height,
            )
            painter.drawImage(rect, self._image)
        if self._editing:
            painter.setPen(QPen(QColor(255, 215, 0), 2))
            painter.drawText(16, 28, self._mode_text)
            if self._drag_start is not None and self._drag_current is not None:
                rect = QRectF(self._drag_start, self._drag_current).normalized()
                painter.drawRect(rect)
        painter.end()

    def mousePressEvent(self, event) -> None:
        if self._editing and event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
            self._drag_current = event.pos()
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._editing and self._drag_start is not None:
            self._drag_current = event.pos()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._editing and event.button() == Qt.LeftButton and self._drag_start is not None:
            self._drag_current = event.pos()
            rect = QRectF(self._drag_start, self._drag_current).normalized()
            transform = self._transform
            source = self._source_bgr
            self._drag_start = None
            self._drag_current = None
            if transform is not None and source is not None:
                roi = transform.widget_roi_to_source(rect, source.shape)
                if roi is not None and roi[2] - roi[0] >= 2 and roi[3] - roi[1] >= 2:
                    self.roiSelected.emit(roi)
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if self._editing:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
                self.editFinished.emit()
                event.accept()
                return
            if event.key() == Qt.Key_Escape:
                self._drag_start = None
                self._drag_current = None
                self.editCancelled.emit()
                event.accept()
                return
            if event.key() == Qt.Key_Q:
                self.quitRequested.emit()
                event.accept()
                return
        super().keyPressEvent(event)
