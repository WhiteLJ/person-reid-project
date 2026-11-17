"""Qt read-only page for Person/Vehicle appearance history."""

from __future__ import annotations

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .models import AppearanceLogRecordDTO


class AppearanceLogTableModel(QAbstractTableModel):
    HEADERS = (
        "\u65f6\u95f4",
        "\u7c7b\u578b",
        "\u76ee\u6807ID",
        "\u4e8b\u4ef6",
        "\u8f66\u724c\u53f7",
        "\u76f8\u4f3c\u5ea6",
        "\u6765\u6e90",
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._records: tuple[AppearanceLogRecordDTO, ...] = ()

    def set_records(self, records) -> None:
        # Newest events are shown first while the repository keeps its
        # deterministic oldest-to-newest load_all() contract.
        ordered = tuple(reversed(tuple(records)))
        self.beginResetModel()
        self._records = ordered
        self.endResetModel()

    def record_at(self, row: int) -> AppearanceLogRecordDTO | None:
        return self._records[row] if 0 <= row < len(self._records) else None

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._records)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            if index.isValid() and role == Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignCenter
            return None
        record = self.record_at(index.row())
        if record is None:
            return None
        if index.column() == 0:
            return record.event_time
        if index.column() == 1:
            return "\u4eba\u5458" if record.domain == "person" else "\u8f66\u8f86"
        if index.column() == 2:
            return record.display_id
        if index.column() == 3:
            return "\u8bc6\u522b" if record.event_type == "recognition" else "\u6062\u590d"
        if index.column() == 4:
            if record.domain == "person":
                return "\u2014"
            return record.plate_number or "NULL"
        if index.column() == 5:
            return "\u2014" if record.similarity is None else f"{record.similarity:.4f}"
        if index.column() == 6:
            return record.source_label or "\u2014"
        return None


class AppearanceLogPage(QWidget):
    """Read-only appearance log page; all data arrives as DTO snapshots."""

    backRequested = pyqtSignal()
    refreshRequested = pyqtSignal()
    exportRequested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)

        toolbar = QHBoxLayout()
        title = QLabel("\u51fa\u73b0\u8bb0\u5f55")
        title_font = QFont(title.font())
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        toolbar.addWidget(title)
        toolbar.addStretch(1)

        self.refresh_button = QPushButton("\u5237\u65b0")
        self.export_button = QPushButton("\u5bfc\u51faTXT")
        self.back_button = QPushButton("\u8fd4\u56de")
        for button in (self.export_button, self.refresh_button, self.back_button):
            button_font = QFont(button.font())
            button_font.setPointSize(18)
            button.setFont(button_font)
            button.setMinimumHeight(60)
            button.setMinimumWidth(140)
        toolbar.addWidget(self.export_button)
        toolbar.addWidget(self.refresh_button)
        toolbar.addWidget(self.back_button)
        root.addLayout(toolbar)

        self.table = QTableView()
        self.model = AppearanceLogTableModel(self)
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
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
        for column in range(len(AppearanceLogTableModel.HEADERS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(48)
        root.addWidget(self.table, 1)

        self.export_button.clicked.connect(self.exportRequested.emit)
        self.refresh_button.clicked.connect(self.refreshRequested.emit)
        self.back_button.clicked.connect(self.backRequested.emit)

    def set_records(self, records) -> None:
        self.model.set_records(records)
