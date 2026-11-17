"""In-memory Qt activity feed for recent enrollment/recovery/recognition."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .models import ActivityEventDTO


class ActivityCard(QFrame):
    def __init__(self, event: ActivityEventDTO, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        image = QLabel()
        image.setFixedSize(100, 76)
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if event.snapshot_jpeg:
            pixmap = QPixmap()
            pixmap.loadFromData(event.snapshot_jpeg, "JPG")
            if not pixmap.isNull():
                image.setPixmap(
                    pixmap.scaled(
                        image.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        root.addWidget(image)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        title = QLabel(event.display_id)
        title_font = QFont(title.font())
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        text_layout.addWidget(title)
        domain = "人员" if event.domain == "person" else "车辆"
        event_name = {
            "enrollment": "入库",
            "recovery": "恢复",
            "recognition": "识别",
        }.get(event.event_type, event.event_type)
        text_layout.addWidget(QLabel(f"{domain}  {event_name}"))
        text_layout.addWidget(QLabel(event.event_time))
        if event.domain == "vehicle":
            text_layout.addWidget(QLabel(event.plate_number or "NULL"))
        root.addLayout(text_layout, 1)


class ActivityFeed(QWidget):
    """Bounded, newest-first activity cards kept only in Qt memory."""

    def __init__(self, max_events: int = 100, parent=None) -> None:
        super().__init__(parent)
        if int(max_events) < 1:
            raise ValueError("max_events must be positive")
        self.max_events = int(max_events)
        self._events: list[ActivityEventDTO] = []
        self._cards: list[ActivityCard] = []
        self.setMinimumWidth(320)
        self.setMaximumWidth(400)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 0, 0, 0)
        title = QLabel("目标动态")
        title_font = QFont(title.font())
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._card_container = QWidget()
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setContentsMargins(0, 0, 0, 0)
        self._card_layout.setSpacing(8)
        self._card_layout.addStretch(1)
        self.scroll_area.setWidget(self._card_container)
        root.addWidget(self.scroll_area, 1)

    @property
    def events(self) -> tuple[ActivityEventDTO, ...]:
        return tuple(self._events)

    def add_event(self, event: ActivityEventDTO) -> None:
        if not isinstance(event, ActivityEventDTO):
            raise TypeError("ActivityFeed expects ActivityEventDTO")
        self._events.insert(0, event)
        card = ActivityCard(event, self._card_container)
        self._cards.insert(0, card)
        self._card_layout.insertWidget(0, card)
        while len(self._cards) > self.max_events:
            old_card = self._cards.pop()
            self._events.pop()
            old_card.setParent(None)
            old_card.deleteLater()

    def clear_events(self) -> None:
        while self._cards:
            card = self._cards.pop()
            card.setParent(None)
            card.deleteLater()
        self._events.clear()
