"""
NoteMe — OS Meeting Agent
Run with:  python main.py
"""
from __future__ import annotations

import os
import sys
import logging

# In PyInstaller windowed (console=False) builds, sys.stdout and sys.stderr
# are None.  Libraries like tqdm (used by huggingface_hub) crash with
# "'NoneType' object has no attribute 'write'" on their first print.
# Redirect both to the null device so those writes are silently discarded.
# Real output goes to the log file via setup_logger().
if getattr(sys, "frozen", False):
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")  # noqa: WPS515
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")  # noqa: WPS515

# Disable tqdm progress bars globally — they crash when used from a QThread
# (tqdm's internal threading lock becomes None in non-main threads).
# Setting env vars is not enough because tqdm may already be imported.
# The only reliable fix is to replace the tqdm class itself with a no-op.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"


class _NoOpTqdm:
    """Drop-in tqdm replacement that does nothing (thread-safe)."""
    def __init__(self, iterable=None, *args, **kwargs):
        self._it = iterable
    def __iter__(self):
        return iter(self._it) if self._it is not None else iter([])
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def update(self, n=1): pass
    def close(self): pass
    def set_description(self, *a, **kw): pass
    def set_postfix(self, *a, **kw): pass
    def write(self, *a, **kw): pass
    def refresh(self, *a, **kw): pass


try:
    import tqdm as _tqdm_mod
    import tqdm.std as _tqdm_std
    import tqdm.auto as _tqdm_auto
    # Patch every known entry point that libraries use to get tqdm
    _tqdm_mod.tqdm = _NoOpTqdm          # type: ignore
    _tqdm_std.tqdm = _NoOpTqdm          # type: ignore
    _tqdm_auto.tqdm = _NoOpTqdm         # type: ignore
    # Some libs do `from tqdm import tqdm` which binds the class directly,
    # so also patch the module-level attribute that the class lives on.
    for attr in ("tqdm", "tqdm_gui", "tqdm_notebook", "tqdm_pandas"):
        if hasattr(_tqdm_mod, attr):
            setattr(_tqdm_mod, attr, _NoOpTqdm)
except ImportError:
    pass

from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QBrush, QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication


def _make_icon(size: int = 64) -> QIcon:
    """Generate a simple purple 'N' icon without needing an external file."""
    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QBrush(QColor("#8e44ad")))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(2, 2, size - 4, size - 4)
    p.setPen(QColor("white"))
    font = p.font()
    font.setPixelSize(int(size * 0.52))
    font.setBold(True)
    p.setFont(font)
    p.drawText(QRect(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, "N")
    p.end()
    return QIcon(px)


def main() -> None:
# ── CUDA DLL bootstrap (must run before faster_whisper is imported) ──
    from utils.cuda_setup import bootstrap as _cuda_bootstrap
    _cuda_bootstrap()

    # Fix for tqdm in environments where sys.stdout/stderr might be None
    import tqdm
    try:
        tqdm.tqdm._lock = None
    except AttributeError:
        pass

    # ── Bootstrap ────────────────────────────────────────────────────
    from utils.config import get_config
    from utils.logger import setup_logger
    from core.storage.database import init_db

    config = get_config()
    storage_path = config.app.resolved_storage_path
    storage_path.mkdir(parents=True, exist_ok=True)

    setup_logger(storage_path)
    log = logging.getLogger(__name__)
    log.info("NoteMe starting…")

    init_db(storage_path)

    # ── Qt application ───────────────────────────────────────────────
    app = QApplication(sys.argv)
    app.setApplicationName("NoteMe")
    app.setQuitOnLastWindowClosed(False)   # keep running in tray

    icon = _make_icon()
    app.setWindowIcon(icon)

    # ── Windows + tray ───────────────────────────────────────────────
    from ui.main_window import MainWindow
    from ui.tray import SystemTray

    window = MainWindow(config)
    tray = SystemTray(icon)

    # Connect tray → window
    tray.show_requested.connect(window.show)
    tray.show_requested.connect(window.raise_)
    tray.show_requested.connect(window.activateWindow)

    # Tray quick-record uses the config defaults
    tray.start_recording_requested.connect(
        lambda: window._start_recording(
            config.audio.default_source,
            config.transcription.default_language,
        )
    )
    tray.stop_recording_requested.connect(window._stop_recording)
    tray.quit_requested.connect(app.quit)

    # Keep tray menu in sync with recording state
    window.recording_state_changed.connect(tray.set_recording)

    tray.show_message("NoteMe", "NoteMe is running in the system tray.\nClick the icon to open.")
    window.show()

    log.info("Event loop started.")
    app.exec()
    # Ensure all threads are properly terminated before exit
    # (Add thread cleanup code here if applicable)
    sys.exit()


if __name__ == "__main__":
    main()
