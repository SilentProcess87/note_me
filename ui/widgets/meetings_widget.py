from __future__ import annotations

import datetime
import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)


class MeetingsWidget(QWidget):
    session_selected = pyqtSignal(int)   # session_id
    refresh_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_data: list[tuple[dict, QListWidgetItem]] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)

        # Search + refresh
        top = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search meetings…")
        self._search.textChanged.connect(self._filter)
        refresh_btn = QPushButton("↻  Refresh")
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self.refresh_requested)
        top.addWidget(self._search)
        top.addWidget(refresh_btn)
        root.addLayout(top)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        root.addWidget(self._list, stretch=1)

        hint = QLabel("Double-click a meeting to view transcript, summary, and Q&A")
        hint.setStyleSheet("color:#888; font-size:11px;")
        root.addWidget(hint)

        btn_row = QHBoxLayout()
        open_btn = QPushButton("Open Selected Meeting")
        open_btn.clicked.connect(self._open_selected)
        btn_row.addStretch()
        btn_row.addWidget(open_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------ #

    def load_sessions(self, sessions: list[dict]) -> None:
        self._list.clear()
        self._all_data.clear()
        for s in sessions:
            item = self._make_item(s)
            self._list.addItem(item)
            self._all_data.append((s, item))

    def _make_item(self, s: dict) -> QListWidgetItem:
        icon = "🗣" if s.get("mode") == "speech_coach" else "🤝"
        title = s.get("title", "Untitled")
        start = s.get("start_time", "")
        if isinstance(start, datetime.datetime):
            start = start.strftime("%Y-%m-%d  %H:%M")
        duration = s.get("duration_str", "")
        item = QListWidgetItem(f"{icon}  {title}\n    {start}   {duration}")
        item.setData(Qt.ItemDataRole.UserRole, s.get("id"))
        return item

    def _filter(self, text: str) -> None:
        needle = text.lower()
        for data, item in self._all_data:
            haystack = (data.get("title", "") + str(data.get("start_time", ""))).lower()
            item.setHidden(needle not in haystack)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        sid = item.data(Qt.ItemDataRole.UserRole)
        if sid is not None:
            self.session_selected.emit(sid)

    def _open_selected(self) -> None:
        selected = self._list.selectedItems()
        if selected:
            sid = selected[0].data(Qt.ItemDataRole.UserRole)
            if sid is not None:
                self.session_selected.emit(sid)
