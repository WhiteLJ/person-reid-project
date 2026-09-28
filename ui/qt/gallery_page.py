"""Qt model/view Gallery management page."""

from __future__ import annotations

from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton, QStyledItemDelegate, QTableView, QVBoxLayout, QWidget

from .models import GalleryRecordDTO


class ThumbnailDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index) -> None:
        pixmap = index.data(Qt.DecorationRole)
        if isinstance(pixmap, QPixmap) and not pixmap.isNull():
            scaled = pixmap.scaled(option.rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(option.rect.x() + (option.rect.width() - scaled.width()) // 2, option.rect.y() + (option.rect.height() - scaled.height()) // 2, scaled)
            return
        super().paint(painter, option, index)


class GalleryTableModel(QAbstractTableModel):
    HEADERS = ("\u76ee\u6807 ID", "\u76ee\u6807\u56fe\u50cf", "\u5220\u9664")

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
        return 0 if parent.isValid() else 3

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        record = self.record_at(index.row())
        if record is None:
            return None
        if role == Qt.DisplayRole:
            if index.column() == 0:
                return record.display_id
            if index.column() == 2:
                return "\u5220\u9664"
            return "\u65e0\u5feb\u7167" if record.snapshot_jpeg is None else ""
        if role == Qt.DecorationRole and index.column() == 1 and record.snapshot_jpeg:
            pixmap = QPixmap()
            pixmap.loadFromData(record.snapshot_jpeg, "JPG")
            return pixmap
        if role == Qt.TextAlignmentRole and index.column() in {0, 1, 2}:
            return int(Qt.AlignCenter)
        return None


class SnapshotPreviewDialog(QDialog):
    def __init__(self, record: GalleryRecordDTO, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(record.display_id)
        layout = QVBoxLayout(self)
        image = QLabel("\u65e0\u5feb\u7167")
        image.setAlignment(Qt.AlignCenter)
        if record.snapshot_jpeg:
            pixmap = QPixmap()
            pixmap.loadFromData(record.snapshot_jpeg, "JPG")
            if not pixmap.isNull():
                image.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(image)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class GalleryPage(QWidget):
    backRequested = pyqtSignal()
    deleteRequested = pyqtSignal(str, object, bool)

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
        self.batch_button = QPushButton("\u6279\u91cf\u7ba1\u7406")
        self.back_button = QPushButton("\u8fd4\u56de")
        domain_buttons = QHBoxLayout()
        domain_buttons.setSpacing(10)
        domain_buttons.addWidget(self.people_button)
        domain_buttons.addWidget(self.vehicle_button)
        toolbar.addLayout(domain_buttons, 0, 0, alignment=Qt.AlignLeft)
        toolbar.addWidget(self.batch_button, 0, 1, alignment=Qt.AlignCenter)
        toolbar.addWidget(self.back_button, 0, 2, alignment=Qt.AlignRight)
        root.addLayout(toolbar)
        self.table = QTableView()
        self.model = GalleryTableModel(self)
        self.table.setModel(self.model)
        self.table.setItemDelegateForColumn(1, ThumbnailDelegate(self.table))
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        for column in range(3):
            header.setSectionResizeMode(column, QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(150)
        root.addWidget(self.table, 1)
        self.people_button.clicked.connect(lambda: self.set_domain("person"))
        self.vehicle_button.clicked.connect(lambda: self.set_domain("vehicle"))
        self.batch_button.clicked.connect(self._toggle_batch)
        self.back_button.clicked.connect(self.backRequested)
        self.table.clicked.connect(self._table_clicked)
        self.table.doubleClicked.connect(self._preview_clicked)

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
        self.table.setSelectionMode(QAbstractItemView.MultiSelection if self._batch_mode else QAbstractItemView.SingleSelection)
        self.batch_button.setText("\u786e\u8ba4\u6279\u91cf\u5220\u9664" if self._batch_mode else "\u6279\u91cf\u7ba1\u7406")
        if self._batch_mode:
            return
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        ids = [record.identity_id for row in rows if (record := self.model.record_at(row)) is not None]
        if ids:
            self.deleteRequested.emit(self._domain, ids, True)

    def _table_clicked(self, index) -> None:
        if index.column() == 2 and not self._batch_mode:
            record = self.model.record_at(index.row())
            if record is not None:
                self.deleteRequested.emit(record.domain, record.identity_id, False)

    def _preview_clicked(self, index) -> None:
        if index.column() == 1:
            record = self.model.record_at(index.row())
            if record is not None:
                SnapshotPreviewDialog(record, self).exec_()
