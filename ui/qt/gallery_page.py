"""Qt model/view Gallery management page."""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtCore import QAbstractTableModel, QEvent, QModelIndex, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QMouseEvent, QPainter, QPixmap
from PyQt6.QtWidgets import QAbstractItemView, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton, QStyle, QStyleOptionButton, QStyledItemDelegate, QTableView, QVBoxLayout, QWidget

from .models import GalleryRecordDTO


class ThumbnailDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index) -> None:
        pixmap = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(pixmap, QPixmap) and not pixmap.isNull():
            scaled = pixmap.scaled(
                option.rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(option.rect.x() + (option.rect.width() - scaled.width()) // 2, option.rect.y() + (option.rect.height() - scaled.height()) // 2, scaled)
            return
        super().paint(painter, option, index)


class RecognitionDelegate(QStyledItemDelegate):
    """Paint and toggle the recognition option as centered ``\u5426\u2610``/``\u662f\u2611``."""

    def paint(self, painter: QPainter, option, index) -> None:
        painter.save()
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())

        enabled = index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked
        text = "\u662f" if enabled else "\u5426"
        metrics = painter.fontMetrics()
        box_size = max(16, metrics.height() - 2)
        spacing = max(3, metrics.averageCharWidth() // 4)
        text_width = metrics.horizontalAdvance(text)
        total_width = text_width + spacing + box_size
        start_x = option.rect.left() + (option.rect.width() - total_width) // 2
        text_rect = option.rect.adjusted(
            start_x - option.rect.left(),
            0,
            start_x - option.rect.left(),
            0,
        )
        text_rect.setWidth(text_width)
        painter.setPen(
            option.palette.highlightedText().color()
            if option.state & QStyle.StateFlag.State_Selected
            else option.palette.text().color()
        )
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, text)

        checkbox = QStyleOptionButton()
        checkbox.rect = option.rect.__class__(
            start_x + text_width + spacing,
            option.rect.top() + (option.rect.height() - box_size) // 2,
            box_size,
            box_size,
        )
        checkbox.state = QStyle.StateFlag.State_Enabled
        if enabled:
            checkbox.state |= QStyle.StateFlag.State_On
        if option.widget is None:
            painter.restore()
            return
        style = option.widget.style()
        style.drawPrimitive(
            QStyle.PrimitiveElement.PE_IndicatorCheckBox,
            checkbox,
            painter,
            option.widget,
        )
        painter.restore()

    def editorEvent(self, event, model, option, index) -> bool:
        if (
            event.type() != QEvent.Type.MouseButtonRelease
            or not isinstance(event, QMouseEvent)
            or event.button() != Qt.MouseButton.LeftButton
        ):
            return False
        current = index.data(Qt.ItemDataRole.CheckStateRole)
        next_state = (
            Qt.CheckState.Unchecked
            if current == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )
        return bool(model.setData(index, next_state, Qt.ItemDataRole.CheckStateRole))


class GalleryTableModel(QAbstractTableModel):
    HEADERS = ("\u76ee\u6807 ID", "\u76ee\u6807\u56fe\u50cf", "\u8f66\u724c\u53f7", "\u5165\u5e93\u65f6\u95f4", "\u53c2\u4e0e\u8bc6\u522b", "\u5220\u9664")
    recognitionEnabledChanged = pyqtSignal(str, int, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._records: list[GalleryRecordDTO] = []

    def set_records(self, records: tuple[GalleryRecordDTO, ...] | list[GalleryRecordDTO]) -> None:
        self.beginResetModel()
        self._records = list(records)
        self.endResetModel()

    def record_at(self, row: int) -> GalleryRecordDTO | None:
        return self._records[row] if 0 <= row < len(self._records) else None

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._records)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else 6

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        record = self.record_at(index.row())
        if record is None:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return record.display_id
            if index.column() == 2:
                return (
                    record.plate_number
                    if record.plate_number is not None
                    else ("\u2014" if record.domain == "person" else "NULL")
                )
            if index.column() == 3:
                return record.created_at or "\u672a\u77e5"
            if index.column() == 4:
                return "\u662f" if record.recognition_enabled else "\u5426"
            if index.column() == 5:
                return "\u5220\u9664"
            return "\u65e0\u5feb\u7167" if record.snapshot_jpeg is None else ""
        if role == Qt.ItemDataRole.CheckStateRole and index.column() == 4:
            return (
                Qt.CheckState.Checked
                if record.recognition_enabled
                else Qt.CheckState.Unchecked
            )
        if role == Qt.ItemDataRole.DecorationRole and index.column() == 1 and record.snapshot_jpeg:
            pixmap = QPixmap()
            pixmap.loadFromData(record.snapshot_jpeg, "JPG")
            return pixmap
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() in {0, 1, 2, 3, 4, 5}:
            return Qt.AlignmentFlag.AlignCenter
        return None

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid() and index.column() == 4:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if (
            not index.isValid()
            or index.column() != 4
            or role not in {
                Qt.ItemDataRole.CheckStateRole,
                Qt.ItemDataRole.EditRole,
            }
        ):
            return False
        record = self.record_at(index.row())
        if record is None:
            return False
        try:
            if isinstance(value, bool):
                enabled = value
            else:
                state = value if isinstance(value, Qt.CheckState) else Qt.CheckState(value)
                enabled = state == Qt.CheckState.Checked
        except (TypeError, ValueError):
            return False
        if enabled == record.recognition_enabled:
            return True
        self._records[index.row()] = replace(record, recognition_enabled=enabled)
        self.dataChanged.emit(
            index,
            index,
            [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.CheckStateRole],
        )
        self.recognitionEnabledChanged.emit(record.domain, record.identity_id, enabled)
        return True


class SnapshotPreviewDialog(QDialog):
    def __init__(self, record: GalleryRecordDTO, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(record.display_id)
        layout = QVBoxLayout(self)
        image = QLabel("\u65e0\u5feb\u7167")
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if record.snapshot_jpeg:
            pixmap = QPixmap()
            pixmap.loadFromData(record.snapshot_jpeg, "JPG")
            if not pixmap.isNull():
                image.setPixmap(
                    pixmap.scaled(
                        800,
                        600,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        layout.addWidget(image)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_font = QFont(self.font())
        button_font.setPointSize(18)
        buttons.setFont(button_font)
        buttons.setMinimumHeight(56)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class GalleryPage(QWidget):
    backRequested = pyqtSignal()
    deleteRequested = pyqtSignal(str, object, bool)
    importRequested = pyqtSignal(str)
    appearanceLogRequested = pyqtSignal()
    recognitionEnabledChanged = pyqtSignal(str, int, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._all_records: tuple[GalleryRecordDTO, ...] = ()
        self._domain = "person"
        self._batch_mode = False
        root = QVBoxLayout(self)
        toolbar = QGridLayout()
        toolbar.setHorizontalSpacing(16)
        toolbar.setColumnStretch(0, 1)
        toolbar.setColumnStretch(1, 1)
        toolbar.setColumnStretch(2, 1)
        self.people_button = QPushButton("\u4eba\u5458\u5e93")
        self.vehicle_button = QPushButton("\u8f66\u8f86\u5e93")
        self.appearance_log_button = QPushButton("\u51fa\u73b0\u8bb0\u5f55")
        self.import_button = QPushButton("\u8bfb\u53d6\u5916\u90e8\u6587\u4ef6")
        self.batch_button = QPushButton("\u6279\u91cf\u7ba1\u7406")
        self.back_button = QPushButton("\u8fd4\u56de")
        for button in (
            self.people_button,
            self.vehicle_button,
            self.appearance_log_button,
            self.import_button,
            self.batch_button,
            self.back_button,
        ):
            button_font = QFont(button.font())
            button_font.setPointSize(18)
            button.setFont(button_font)
            button.setMinimumHeight(60)
        domain_buttons = QHBoxLayout()
        domain_buttons.setSpacing(10)
        domain_buttons.addWidget(self.people_button)
        domain_buttons.addWidget(self.vehicle_button)
        domain_buttons.addWidget(self.appearance_log_button)
        toolbar.addLayout(domain_buttons, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        management_buttons = QHBoxLayout()
        management_buttons.setSpacing(10)
        management_buttons.addWidget(self.import_button)
        management_buttons.addWidget(self.batch_button)
        toolbar.addLayout(management_buttons, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        toolbar.addWidget(self.back_button, 0, 2, alignment=Qt.AlignmentFlag.AlignRight)
        root.addLayout(toolbar)
        self.table = QTableView()
        self.model = GalleryTableModel(self)
        self.table.setModel(self.model)
        self.table.setItemDelegateForColumn(1, ThumbnailDelegate(self.table))
        self.table.setItemDelegateForColumn(4, RecognitionDelegate(self.table))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        table_font = QFont(self.table.font())
        table_font.setPointSize(18)
        self.table.setFont(table_font)
        header = self.table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        header_font = QFont(header.font())
        header_font.setPointSize(18)
        header_font.setBold(True)
        header.setFont(header_font)
        for column in range(6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(150)
        root.addWidget(self.table, 1)
        self.people_button.clicked.connect(lambda: self.set_domain("person"))
        self.vehicle_button.clicked.connect(lambda: self.set_domain("vehicle"))
        self.appearance_log_button.clicked.connect(self.appearanceLogRequested.emit)
        self.import_button.clicked.connect(lambda: self.importRequested.emit(self._domain))
        self.batch_button.clicked.connect(self._toggle_batch)
        self.back_button.clicked.connect(self.backRequested)
        self.table.clicked.connect(self._table_clicked)
        self.table.doubleClicked.connect(self._preview_clicked)
        self.model.recognitionEnabledChanged.connect(self.recognitionEnabledChanged)

    def set_records(self, records: tuple[GalleryRecordDTO, ...]) -> None:
        self._all_records = tuple(records)
        self._refresh_model()

    def set_domain(self, domain: str) -> None:
        if domain not in {"person", "vehicle"}:
            raise ValueError("domain must be person or vehicle")
        self._domain = domain
        self._refresh_model()

    def _refresh_model(self) -> None:
        self.model.set_records(tuple(record for record in self._all_records if record.domain == self._domain))
        self.people_button.setEnabled(self._domain != "person")
        self.vehicle_button.setEnabled(self._domain != "vehicle")

    def _toggle_batch(self) -> None:
        self._batch_mode = not self._batch_mode
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
            if self._batch_mode
            else QAbstractItemView.SelectionMode.SingleSelection
        )
        self.batch_button.setText("\u786e\u8ba4\u6279\u91cf\u5220\u9664" if self._batch_mode else "\u6279\u91cf\u7ba1\u7406")
        if self._batch_mode:
            return
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        ids = [record.identity_id for row in rows if (record := self.model.record_at(row)) is not None]
        if ids:
            self.deleteRequested.emit(self._domain, ids, True)

    def _table_clicked(self, index) -> None:
        if index.column() == 5 and not self._batch_mode:
            record = self.model.record_at(index.row())
            if record is not None:
                self.deleteRequested.emit(record.domain, record.identity_id, False)

    def _preview_clicked(self, index) -> None:
        if index.column() == 1:
            record = self.model.record_at(index.row())
            if record is not None:
                SnapshotPreviewDialog(record, self).exec()
