from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon

log = logging.getLogger(__name__)


class SystemTray(QObject):
    show_requested = pyqtSignal()
    start_recording_requested = pyqtSignal()
    stop_recording_requested = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self, icon: QIcon, parent=None):
        super().__init__(parent)
        self._tray = QSystemTrayIcon(icon, parent)
        self._build_menu()
        self._tray.setToolTip("NoteMe — Ready")
        self._tray.show()

    # ------------------------------------------------------------------ #

    def _build_menu(self) -> None:
        menu = QMenu()

        action_open = QAction("Open NoteMe", menu)
        action_open.triggered.connect(self.show_requested)
        menu.addAction(action_open)

        menu.addSeparator()

        self._action_start = QAction("▶  Start Recording", menu)
        self._action_start.triggered.connect(self.start_recording_requested)
        menu.addAction(self._action_start)

        self._action_stop = QAction("⏹  Stop Recording", menu)
        self._action_stop.triggered.connect(self.stop_recording_requested)
        self._action_stop.setEnabled(False)
        menu.addAction(self._action_stop)

        menu.addSeparator()

        action_quit = QAction("Exit", menu)
        action_quit.triggered.connect(self.quit_requested)
        menu.addAction(action_quit)

        self._tray.setContextMenu(menu)

    def set_recording(self, recording: bool) -> None:
        self._action_start.setEnabled(not recording)
        self._action_stop.setEnabled(recording)
        self._tray.setToolTip("NoteMe — 🔴 Recording…" if recording else "NoteMe — Ready")

    def show_message(self, title: str, message: str) -> None:
        self._tray.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, 3000)
