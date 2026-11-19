"""In-memory Qt activity feed with silent visual attention alerts."""

from __future__ import annotations

from time import monotonic

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .models import ActivityEventDTO


class ActivityCard(QFrame):
    """One activity event and, when required, its UI-only alarm state."""

    acknowledged = pyqtSignal(int)

    _NORMAL_STYLE = (
        "QFrame { border: 1px solid #c8c8c8; background-color: #ffffff; "
        "border-radius: 4px; }"
    )
    _ALERT_STYLE = (
        "QFrame { border: 3px solid #d32f2f; background-color: #fff3f3; "
        "border-radius: 4px; }"
    )
    _ALERT_ALT_STYLE = (
        "QFrame { border: 3px solid #ef6c00; background-color: #fff8ee; "
        "border-radius: 4px; }"
    )

    def __init__(
        self,
        event: ActivityEventDTO,
        *,
        alarm_flash_seconds: float = 8.0,
        alarm_flash_interval_ms: int = 500,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.event = event
        self._requires_attention = bool(event.requires_attention)
        self._acknowledged = False
        self._flash_state = False
        self._flash_deadline: float | None = None
        self._flash_timer: QTimer | None = None
        self._status_label: QLabel | None = None
        self._acknowledge_button: QPushButton | None = None
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            self._ALERT_STYLE if self._requires_attention else self._NORMAL_STYLE
        )

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

        if self._requires_attention:
            attention_row = QHBoxLayout()
            self._status_label = QLabel("未确认")
            self._status_label.setStyleSheet("color: #b71c1c; border: 0px;")
            attention_row.addWidget(self._status_label)
            attention_row.addStretch(1)
            self._acknowledge_button = QPushButton("确认")
            self._acknowledge_button.setMinimumHeight(40)
            self._acknowledge_button.clicked.connect(self._acknowledge)
            attention_row.addWidget(self._acknowledge_button)
            text_layout.addLayout(attention_row)

        root.addLayout(text_layout, 1)

        if self._requires_attention and alarm_flash_seconds > 0:
            self._flash_deadline = monotonic() + float(alarm_flash_seconds)
            self._flash_timer = QTimer(self)
            self._flash_timer.setInterval(max(100, int(alarm_flash_interval_ms)))
            self._flash_timer.timeout.connect(self._toggle_flash)
            self._flash_timer.start()

    @property
    def is_unacknowledged(self) -> bool:
        return self._requires_attention and not self._acknowledged

    @property
    def flash_timer_active(self) -> bool:
        return self._flash_timer is not None and self._flash_timer.isActive()

    @property
    def acknowledged_state(self) -> bool:
        return self._acknowledged

    def _toggle_flash(self) -> None:
        if self._acknowledged:
            self._stop_flashing()
            return
        if self._flash_deadline is not None and monotonic() >= self._flash_deadline:
            self._stop_flashing()
            return
        self._flash_state = not self._flash_state
        self.setStyleSheet(
            self._ALERT_ALT_STYLE if self._flash_state else self._ALERT_STYLE
        )

    def _stop_flashing(self) -> None:
        if self._flash_timer is not None:
            self._flash_timer.stop()
        if self._requires_attention and not self._acknowledged:
            # The alert remains visually prominent after the short attention
            # phase; only the flashing stops.
            self.setStyleSheet(self._ALERT_STYLE)

    def _acknowledge(self) -> None:
        if self._acknowledged:
            return
        self._acknowledged = True
        self._stop_flashing()
        if self._status_label is not None:
            self._status_label.setText("已确认")
            self._status_label.setStyleSheet("color: #555555; border: 0px;")
        if self._acknowledge_button is not None:
            self._acknowledge_button.setVisible(False)
        self.setStyleSheet(self._NORMAL_STYLE)
        self.acknowledged.emit(int(self.event.sequence_id))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self._flash_timer is not None:
            self._flash_timer.stop()
        super().closeEvent(event)


class ActivityFeed(QWidget):
    """Bounded, newest-first activity cards kept only in Qt memory."""

    def __init__(
        self,
        max_events: int = 100,
        *,
        alarm_flash_seconds: float = 8.0,
        alarm_flash_interval_ms: int = 500,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if int(max_events) < 1:
            raise ValueError("max_events must be positive")
        if float(alarm_flash_seconds) < 0.0:
            raise ValueError("alarm_flash_seconds must be non-negative")
        if int(alarm_flash_interval_ms) < 100:
            raise ValueError("alarm_flash_interval_ms must be at least 100")
        self.max_events = int(max_events)
        self.alarm_flash_seconds = float(alarm_flash_seconds)
        self.alarm_flash_interval_ms = int(alarm_flash_interval_ms)
        self._events: list[ActivityEventDTO] = []
        self._cards: list[ActivityCard] = []
        self.setMinimumWidth(320)
        self.setMaximumWidth(400)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 0, 0, 0)
        self._title = QLabel()
        title_font = QFont(self._title.font())
        title_font.setPointSize(18)
        title_font.setBold(True)
        self._title.setFont(title_font)
        root.addWidget(self._title)

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
        self._update_title()

    @property
    def events(self) -> tuple[ActivityEventDTO, ...]:
        return tuple(self._events)

    @property
    def unacknowledged_count(self) -> int:
        return sum(card.is_unacknowledged for card in self._cards)

    def _update_title(self) -> None:
        count = self.unacknowledged_count if hasattr(self, "_cards") else 0
        self._title.setText(
            "目标动态" if count == 0 else f"目标动态（未确认 {count}）"
        )

    def _on_card_acknowledged(self, _sequence_id: int) -> None:
        self._update_title()

    def add_event(self, event: ActivityEventDTO) -> None:
        if not isinstance(event, ActivityEventDTO):
            raise TypeError("ActivityFeed expects ActivityEventDTO")
        self._events.insert(0, event)
        card = ActivityCard(
            event,
            alarm_flash_seconds=self.alarm_flash_seconds,
            alarm_flash_interval_ms=self.alarm_flash_interval_ms,
            parent=self._card_container,
        )
        card.acknowledged.connect(self._on_card_acknowledged)
        self._cards.insert(0, card)
        self._card_layout.insertWidget(0, card)
        while len(self._cards) > self.max_events:
            old_card = self._cards.pop()
            self._events.pop()
            old_card.close()
            old_card.setParent(None)
            old_card.deleteLater()
        self._update_title()
        QTimer.singleShot(0, lambda: self.scroll_area.verticalScrollBar().setValue(0))

    def clear_events(self) -> None:
        while self._cards:
            card = self._cards.pop()
            card.close()
            card.setParent(None)
            card.deleteLater()
        self._events.clear()
        self._update_title()
