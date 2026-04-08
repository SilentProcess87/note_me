"""
ZoomWatcher — polls the Windows window list for active Zoom meeting windows.

Uses only ctypes (built-in), no third-party dependencies.

Zoom creates distinctive window class names during an active meeting:
  ZPMeetingFrame        main meeting window
  ZPFloatVideoWnd       floating video strip
  ZPToolBarParentWnd    meeting control toolbar

Detecting any of these means a meeting is in progress.
"""
from __future__ import annotations

import ctypes
import logging
import sys
import time
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

# Window class names that only exist while a Zoom meeting is active.
_ZOOM_MEETING_CLASSES: frozenset[str] = frozenset({
    "ZPMeetingFrame",
    "ZPFloatVideoWnd",
    "ZPToolBarParentWnd",
})

# Also watch for Microsoft Teams meeting window title patterns (optional bonus)
_TEAMS_MEETING_TITLE_KEYWORDS = ("| Microsoft Teams",)

if sys.platform == "win32":
    _EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_long)
else:
    _EnumWindowsProc = None  # type: ignore[assignment]


def zoom_meeting_active() -> bool:
    """Return True if a Zoom meeting window is currently open on this machine."""
    if sys.platform != "win32":
        return False

    found: list[int] = []

    def _callback(hwnd: int, _lparam: int) -> bool:
        cls_buf = ctypes.create_unicode_buffer(256)
        ctypes.windll.user32.GetClassNameW(hwnd, cls_buf, 256)  # type: ignore[attr-defined]
        if cls_buf.value in _ZOOM_MEETING_CLASSES:
            found.append(hwnd)
        return True

    try:
        ctypes.windll.user32.EnumWindows(  # type: ignore[attr-defined]
            _EnumWindowsProc(_callback), 0
        )
    except Exception as exc:
        log.debug("zoom_meeting_active: EnumWindows failed: %s", exc)

    return bool(found)


class ZoomWatcher(QThread):
    """
    Background thread that emits ``meeting_started`` / ``meeting_ended``
    when a Zoom meeting begins or ends.

    Parameters
    ----------
    poll_sec:
        How often (in seconds) to check for a Zoom meeting window.
        Default 3 s — low enough to feel responsive, high enough not to waste CPU.
    """

    meeting_started = pyqtSignal()   # Zoom meeting just began
    meeting_ended   = pyqtSignal()   # Zoom meeting just ended

    def __init__(self, poll_sec: float = 3.0, parent=None) -> None:
        super().__init__(parent)
        self._poll_sec = poll_sec
        self._stop_flag = False
        self._in_meeting = False

    # ------------------------------------------------------------------
    def run(self) -> None:
        log.info("ZoomWatcher started (poll %.1f s)", self._poll_sec)
        while not self._stop_flag:
            try:
                active = zoom_meeting_active()
                if active and not self._in_meeting:
                    self._in_meeting = True
                    log.info("Zoom meeting detected — emitting meeting_started")
                    self.meeting_started.emit()
                elif not active and self._in_meeting:
                    self._in_meeting = False
                    log.info("Zoom meeting ended — emitting meeting_ended")
                    self.meeting_ended.emit()
            except Exception as exc:
                log.debug("ZoomWatcher poll error: %s", exc)
            time.sleep(self._poll_sec)
        log.info("ZoomWatcher stopped")

    def stop_watching(self) -> None:
        """Signal the watcher thread to exit on its next poll cycle."""
        self._stop_flag = True

    @property
    def currently_in_meeting(self) -> bool:
        return self._in_meeting
