from __future__ import annotations

import datetime
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
)

from .widgets.meetings_widget import MeetingsWidget
from .widgets.meeting_detail import MeetingDetailDialog
from .widgets.recording_widget import RecordingWidget
from .widgets.settings_widget import SettingsWidget
from .widgets.speech_coach import SpeechCoachWidget

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Model loader (runs in background so the UI stays responsive)
# ------------------------------------------------------------------ #

class _ModelLoader(QThread):
    model_ready = pyqtSignal(object, str, str)  # WhisperModel, model_name, device_used
    error = pyqtSignal(str)

    def __init__(self, model_name: str, device: str, compute_type: str, storage_path: Path, parent=None):
        super().__init__(parent)
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._storage_path = storage_path

    def run(self) -> None:
        # Disable tqdm progress bars — they crash when run inside a QThread
        # (tqdm's internal lock becomes None in non-main threads)
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        preferred = self._model_name
        candidates = [preferred]
        if preferred == "large-v3" and self._device == "cpu":
            # On CPU, _start_recording will use "base" for live transcription,
            # so load "base" first to avoid a model mismatch (and costly reload).
            candidates = ["base", "turbo", "large-v3"]
        elif preferred == "large-v3":
            candidates += ["turbo", "base"]
        elif preferred == "turbo":
            candidates += ["base"]

        first_error = None
        for name in candidates:
            try:
                model = self._try_load_model(name)
                log.info("Whisper model loaded: %s (%s/%s)", name, self._device, self._compute_type)
                self.model_ready.emit(model, name, self._device)
                return
            except Exception as exc:
                exc_lower = str(exc).lower()
                if first_error is None:
                    first_error = exc
                log.warning("Whisper model '%s' failed to load: %s", name, exc)

                # Detect missing CUDA runtime DLLs and fall back to CPU automatically.
                # This happens when cuBLAS/cuDNN libraries are not installed.
                _cuda_dll_missing = (
                    self._device != "cpu"
                    and any(
                        kw in exc_lower
                        for kw in ("cublas", "cudnn", "cublaslt",
                                   "cannot be loaded", "not found or cannot")
                    )
                )
                if _cuda_dll_missing:
                    log.warning(
                        "CUDA runtime DLL missing — switching to CPU (int8) and retrying '%s'.\n"
                        "To enable GPU, run:\n"
                        "  pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 "
                        '"nvidia-cudnn-cu12>=9,<10"',
                        name,
                    )
                    self._device = "cpu"
                    self._compute_type = "int8"
                    try:
                        model = self._try_load_model(name)
                        log.info("Whisper model loaded (CPU fallback): %s", name)
                        self.model_ready.emit(model, name, self._device)
                        return
                    except Exception as cpu_exc:
                        log.warning("CPU fallback for '%s' also failed: %s", name, cpu_exc)
                    continue

                self._repair_corrupt_snapshot_if_needed(exc)
                continue

        self.error.emit(str(first_error) if first_error else "Unknown Whisper model load error")

    def _try_load_model(self, model_name: str):
        from faster_whisper import WhisperModel
        resolved = self._resolve_model_source(model_name)
        log.info("Loading Whisper model '%s' from '%s'…", model_name, resolved)
        return WhisperModel(str(resolved), device=self._device, compute_type=self._compute_type)

    def _resolve_model_source(self, model_name: str) -> Path | str:
        path = Path(model_name)
        if path.exists():
            return path

        repo_id = self._repo_id_for_model(model_name)
        target = self._storage_path / "models" / repo_id.replace("/", "--")
        target.mkdir(parents=True, exist_ok=True)

        # If model.bin already exists, skip the download entirely.
        # snapshot_download uses tqdm which crashes in QThread.
        if (target / "model.bin").is_file():
            log.info("Model already cached at %s", target)
            return target

        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            allow_patterns=[
                "model.bin",
                "config.json",
                "tokenizer.json",
                "vocabulary.*",
                "preprocessor_config.json",
            ],
        )
        return target

    # Hebrew-specialized model (fine-tuned on Hebrew speech data)
    HEBREW_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"

    @staticmethod
    def _repo_id_for_model(model_name: str) -> str:
        builtin = {
            "tiny": "Systran/faster-whisper-tiny",
            "base": "Systran/faster-whisper-base",
            "small": "Systran/faster-whisper-small",
            "medium": "Systran/faster-whisper-medium",
            "large-v3": "Systran/faster-whisper-large-v3",
            "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
            "hebrew": "ivrit-ai/whisper-large-v3-turbo-ct2",
        }
        return builtin.get(model_name, model_name)

    def _repair_corrupt_snapshot_if_needed(self, exc: Exception) -> None:
        # Kept for compatibility with older failures; model loading now uses
        # a normal local directory instead of HF cache snapshots on Windows.
        return


# ------------------------------------------------------------------ #
# Main window
# ------------------------------------------------------------------ #

class MainWindow(QMainWindow):

    # Emitted so the tray icon can sync its menu state
    recording_state_changed = pyqtSignal(bool)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config
        self._active_session = None
        self._whisper_model = None
        self._whisper_model_hebrew = None   # Dedicated Hebrew model (loaded on demand)
        # Track the device/compute_type/model actually used after any fallback.
        # These may differ from config if CUDA DLLs were missing at load time.
        self._loaded_device: str = config.transcription.device
        self._loaded_compute_type: str = config.transcription.compute_type
        self._loaded_model_name: str = config.transcription.model
        self._ollama_client = None
        self._grammar_service = None
        self._qa_service = None
        self._model_loader: Optional[_ModelLoader] = None
        self._hebrew_loader: Optional[_ModelLoader] = None
        # Zoom auto-record
        self._zoom_watcher = None   # ZoomWatcher | None
        self._zoom_started_recording: bool = False  # True if watcher auto-started recording

        self._setup_ui()
        self._init_services()
        self._load_meetings()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        self.setWindowTitle("NoteMe")
        self.resize(960, 700)

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.West)

        # Tab 0 — Record
        self._record_tab = RecordingWidget()
        self._record_tab.start_requested.connect(self._start_recording)
        self._record_tab.stop_requested.connect(self._stop_recording)
        self._tabs.addTab(self._record_tab, "🎙  Record")

        # Tab 1 — Meetings history
        self._meetings_tab = MeetingsWidget()
        self._meetings_tab.session_selected.connect(self._open_meeting_detail)
        self._meetings_tab.refresh_requested.connect(self._load_meetings)
        self._tabs.addTab(self._meetings_tab, "📋  Meetings")

        # Tab 2 — Speech Coach
        self._coach_tab = SpeechCoachWidget()
        self._tabs.addTab(self._coach_tab, "🗣  Speech Coach")

        # Tab 3 — Settings
        self._settings_tab = SettingsWidget(self._config)
        self._settings_tab.settings_saved.connect(self._on_settings_saved)
        self._settings_tab.refresh_devices_requested.connect(self._refresh_devices)
        self._tabs.addTab(self._settings_tab, "⚙  Settings")

        self.setCentralWidget(self._tabs)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Loading Whisper model…")

    # ------------------------------------------------------------------ #
    # Service initialisation
    # ------------------------------------------------------------------ #

    def _init_services(self) -> None:
        self._init_llm_client()
        self._record_tab.set_qa_service(self._qa_service)

    def _init_llm_client(self) -> None:
        """(Re)create the LLM client and dependent services from current config."""
        from core.llm.provider import create_llm_client
        from core.llm.grammar import GrammarService
        from core.llm.qa import QAService
        from utils.secrets import decrypt_key

        cfg = self._config.llm
        api_key = decrypt_key(cfg.api_key_encrypted)
        self._ollama_client = create_llm_client(
            provider=cfg.provider,
            api_key=api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout_sec,
        )
        self._grammar_service = GrammarService(self._ollama_client, cfg.model)
        self._qa_service = QAService(self._ollama_client, cfg.model)

        # Load Whisper model in background
        self._model_loader = _ModelLoader(
            self._config.transcription.model,
            self._config.transcription.device,
            self._config.transcription.compute_type,
            self._config.app.resolved_storage_path,
            parent=self,
        )
        self._model_loader.model_ready.connect(self._on_model_ready)
        self._model_loader.error.connect(self._on_model_error)
        self._model_loader.start()

        self._refresh_devices()
        self._init_zoom_watcher()
        # Hebrew model is loaded after the main model finishes (see _on_model_ready)
        # to avoid concurrent snapshot_download calls that crash tqdm.

    def _load_hebrew_model(self) -> None:
        """Load the Hebrew-specialized Whisper model in background."""
        if self._whisper_model_hebrew is not None or self._hebrew_loader is not None:
            return
        self._hebrew_loader = _ModelLoader(
            "hebrew",
            self._config.transcription.device,
            self._config.transcription.compute_type,
            self._config.app.resolved_storage_path,
            parent=self,
        )
        self._hebrew_loader.model_ready.connect(self._on_hebrew_model_ready)
        self._hebrew_loader.error.connect(
            lambda e: log.warning("Hebrew model failed to load: %s", e)
        )
        self._hebrew_loader.start()

    @pyqtSlot(object, str, str)
    def _on_hebrew_model_ready(self, model, model_name: str, device_used: str) -> None:
        self._whisper_model_hebrew = model
        log.info("Hebrew Whisper model loaded (%s/%s)", device_used, model_name)

    def _init_zoom_watcher(self) -> None:
        """Start or stop the ZoomWatcher based on current config."""
        # Stop any existing watcher
        if self._zoom_watcher is not None:
            self._zoom_watcher.stop_watching()
            self._zoom_watcher.wait(4_000)
            self._zoom_watcher = None

        if not self._config.app.zoom_auto_record:
            return

        from core.zoom_watcher import ZoomWatcher
        self._zoom_watcher = ZoomWatcher(parent=self)
        self._zoom_watcher.meeting_started.connect(self._on_zoom_meeting_started)
        self._zoom_watcher.meeting_ended.connect(self._on_zoom_meeting_ended)
        self._zoom_watcher.start()
        log.info("ZoomWatcher started — will auto-record Zoom meetings.")

    @pyqtSlot()
    def _on_zoom_meeting_started(self) -> None:
        """Automatically start recording when a Zoom meeting is detected."""
        if self._active_session:
            # Already recording (manually started) — don't interfere
            log.info("Zoom meeting detected but recording already active — skipping auto-start.")
            return
        if self._whisper_model is None:
            log.warning("Zoom meeting detected but Whisper model not ready yet.")
            self._status_bar.showMessage(
                "Zoom meeting detected — Whisper model still loading, recording will start shortly.", 5000
            )
            # Retry once the model is ready (poll every second for up to 30s)
            self._zoom_retry_countdown = 30
            QTimer.singleShot(1000, self._zoom_retry_start)
            return
        self._zoom_started_recording = True
        # Always capture both system audio + mic for Zoom meetings,
        # regardless of the default_source setting.
        language = self._config.transcription.default_language
        self._start_recording("both", language)
        self._status_bar.showMessage("\U0001f7e2  Auto-recording started: Zoom meeting detected.", 5000)
        log.info("Auto-recording started for Zoom meeting.")

    def _zoom_retry_start(self) -> None:
        """Retry starting the recording if the model wasn't ready yet."""
        if self._active_session:
            return  # already recording now
        if self._whisper_model is not None:
            self._on_zoom_meeting_started()
            return
        self._zoom_retry_countdown -= 1
        if self._zoom_retry_countdown > 0:
            QTimer.singleShot(1000, self._zoom_retry_start)

    @pyqtSlot()
    def _on_zoom_meeting_ended(self) -> None:
        """Automatically stop recording when the Zoom meeting ends."""
        if not self._zoom_started_recording:
            return  # recording was started manually, don't auto-stop
        if self._active_session:
            self._zoom_started_recording = False
            self._stop_recording()
            self._status_bar.showMessage("\U0001f534  Auto-recording stopped: Zoom meeting ended.", 5000)
            log.info("Auto-recording stopped: Zoom meeting ended.")

    @pyqtSlot(object, str, str)
    def _on_model_ready(self, model, model_name: str, device_used: str) -> None:
        self._whisper_model = model
        self._loaded_model_name = model_name
        self._loaded_device = device_used
        self._loaded_compute_type = (
            self._config.transcription.compute_type
            if device_used == self._config.transcription.device
            else "int8"  # CPU fallback always uses int8
        )
        self._coach_tab.update_services(model, self._grammar_service)

        fallback_parts = []
        if model_name != self._config.transcription.model:
            fallback_parts.append(f"model '{model_name}'")
        if device_used != self._config.transcription.device:
            fallback_parts.append(
                f"CPU mode (CUDA DLLs missing — "
                f"run: pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 "
                f"\"nvidia-cudnn-cu12>=9,<10\")"
            )

        if fallback_parts:
            self._status_bar.showMessage(
                f"NoteMe ready. Fallback: {', '.join(fallback_parts)}.", 10_000
            )
        else:
            device_label = "GPU" if device_used == "cuda" else "CPU"
            self._status_bar.showMessage(f"NoteMe ready  [{device_label}]", 5000)

        # Now that the main model is loaded, start the Hebrew model if needed.
        # Must be sequential to avoid concurrent tqdm/snapshot_download crashes.
        lang = self._config.transcription.default_language
        if lang in ("he", "auto"):
            self._load_hebrew_model()

    @pyqtSlot(str)
    def _on_model_error(self, error: str) -> None:
        self._status_bar.showMessage(f"Model error: {error}")
        log.error("Whisper model load failed: %s", error)

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    @pyqtSlot(str, str)
    def _start_recording(self, source: str, language: str) -> None:
        if self._active_session:
            return
        if self._whisper_model is None:
            QMessageBox.information(
                self, "NoteMe",
                "The Whisper model is still loading.\nPlease wait a moment and try again."
            )
            return

        from core.session import MeetingSession

        # Use the device/compute_type the model actually loaded with (may differ
        # from config if a CPU fallback was triggered at startup).
        live_model = self._loaded_model_name
        live_device = self._loaded_device
        live_compute = self._loaded_compute_type
        live_chunk_duration = self._config.audio.chunk_duration_sec

        # On CPU, large-v3 is too heavy for live transcription.
        # The _ModelLoader already loads a lighter model (base) in this case,
        # so we just verify and adjust chunk duration as a safety net.
        if live_device == "cpu" and live_model == "large-v3":
            live_model = "base"
            live_chunk_duration = max(5.0, live_chunk_duration)
            self._status_bar.showMessage(
                "Using lighter live model (base) for meeting recording on CPU.",
                5000,
            )

        # Use the Hebrew-specialized model when language is Hebrew
        if language == "he" and self._whisper_model_hebrew is not None:
            preloaded = self._whisper_model_hebrew
            log.info("Using Hebrew-specialized Whisper model.")
        elif live_model == self._loaded_model_name:
            preloaded = self._whisper_model
        else:
            preloaded = None

        self._active_session = MeetingSession(
            source=source,
            language=language,
            system_device_index=self._config.audio.system_device_index,
            mic_device_index=self._config.audio.mic_device_index,
            chunk_duration_sec=live_chunk_duration,
            whisper_model=live_model,
            whisper_compute_type=live_compute,
            whisper_device=live_device,
            preloaded_whisper_model=preloaded,
            ollama_client=self._ollama_client,
            ollama_model=self._config.llm.model,
            storage_path=str(self._config.app.resolved_storage_path),
            vocabulary_hint=self._config.transcription.vocabulary_hint,
        )
        self._active_session.segment_ready.connect(self._on_segment)
        self._active_session.summary_ready.connect(self._on_summary)
        self._active_session.recording_stopped.connect(self._on_recording_stopped)
        self._active_session.levels_updated.connect(self._record_tab.update_levels)
        self._active_session.transcription_progress.connect(self._record_tab.update_transcription_progress)
        self._active_session.error.connect(self._on_session_error)
        self._active_session.start()

        self._record_tab.set_recording(True)
        self._status_bar.showMessage("Recording…")
        self.recording_state_changed.emit(True)

    @pyqtSlot()
    def _stop_recording(self) -> None:
        if self._active_session:
            self._active_session.stop()
            self._record_tab.set_recording(False)
            self._record_tab.set_status("Processing… generating summary")
            self._status_bar.showMessage("Processing recording…")
            self.recording_state_changed.emit(False)

    @pyqtSlot(object)
    def _on_segment(self, segment) -> None:
        self._record_tab.append_segment(segment.start_sec, segment.text)

    @pyqtSlot(str, str)
    def _on_summary(self, summary: str, action_items: str) -> None:
        self._record_tab.set_status("Summary ready! ✓")
        self._status_bar.showMessage("Summary generated.", 5000)
        self._load_meetings()

    @pyqtSlot(int)
    def _on_recording_stopped(self, session_id: int) -> None:
        self._active_session = None
        self._load_meetings()

    @pyqtSlot(str)
    def _on_session_error(self, error: str) -> None:
        QMessageBox.warning(self, "Recording Error", error)
        self._record_tab.set_recording(False)
        self._active_session = None
        self.recording_state_changed.emit(False)

    # ------------------------------------------------------------------ #
    # Meeting detail
    # ------------------------------------------------------------------ #

    @pyqtSlot(int)
    def _open_meeting_detail(self, session_id: int) -> None:
        from core.storage.database import get_db
        from core.storage.models import Session as DbSession

        db = get_db()
        try:
            rec = db.get(DbSession, session_id)
            if not rec:
                return

            lines = []
            for seg in rec.segments:
                m, s = divmod(int(seg.start_sec), 60)
                lines.append(f"[{m:02d}:{s:02d}] {seg.text}")
            transcript = "\n".join(lines)

            summary_text = ""
            action_items = ""
            if rec.summaries:
                latest = rec.summaries[-1]
                summary_text = latest.summary_text
                action_items = latest.action_items or ""

            data = {
                "id": rec.id,
                "title": rec.title,
                "language": rec.language or "auto",
                "transcript": transcript,
                "summary": summary_text,
                "action_items": action_items,
                "audio_path": rec.audio_path or "",
            }
        finally:
            db.close()

        dlg = MeetingDetailDialog(session_data=data, qa_service=self._qa_service, parent=self)
        dlg.exec()

    # ------------------------------------------------------------------ #
    # Meeting list
    # ------------------------------------------------------------------ #

    def _load_meetings(self) -> None:
        from core.storage.database import get_db
        from core.storage.models import Session as DbSession

        db = get_db()
        try:
            records = db.query(DbSession).order_by(DbSession.start_time.desc()).all()
            sessions = []
            for r in records:
                if r.end_time and r.start_time:
                    secs = int((r.end_time - r.start_time).total_seconds())
                    m, s = divmod(secs, 60)
                    duration = f"({m}m {s}s)"
                else:
                    duration = "(in progress)"
                sessions.append({
                    "id": r.id,
                    "title": r.title,
                    "start_time": r.start_time,
                    "duration_str": duration,
                    "mode": r.mode,
                })
            self._meetings_tab.load_sessions(sessions)
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    # Devices refresh
    # ------------------------------------------------------------------ #

    def _refresh_devices(self) -> None:
        from core.audio.devices import get_loopback_devices, get_mic_devices
        loopback = [{"index": d.index, "name": d.name} for d in get_loopback_devices()]
        mics = [{"index": d.index, "name": d.name} for d in get_mic_devices()]
        self._settings_tab.populate_devices(loopback, mics)
        self._sync_coach_devices(loopback, mics)

    def _sync_coach_devices(self, loopback: list[dict], mics: list[dict]) -> None:
        mic_name = "Auto (Default)"
        system_name = "Auto (Default)"
        for d in mics:
            if d["index"] == self._config.audio.mic_device_index:
                mic_name = d["name"]
                break
        for d in loopback:
            if d["index"] == self._config.audio.system_device_index:
                system_name = d["name"]
                break
        self._coach_tab.update_audio_devices(
            mic_device_index=self._config.audio.mic_device_index,
            mic_device_name=mic_name,
            system_device_name=system_name,
        )

    # ------------------------------------------------------------------ #
    # Settings save
    # ------------------------------------------------------------------ #

    @pyqtSlot(dict)
    def _on_settings_saved(self, settings: dict) -> None:
        for section, values in settings.items():
            for key, value in values.items():
                section_obj = getattr(self._config, section, None)
                if section_obj is not None:
                    setattr(section_obj, key, value)
        self._config.save()

        if settings.get("app", {}).get("startup_with_windows") is not None:
            self._set_startup(settings["app"]["startup_with_windows"])

        # Restart Zoom watcher if the setting changed
        self._init_zoom_watcher()

        # Re-init LLM services with new provider/model/key
        self._init_llm_client()
        if self._whisper_model:
            self._coach_tab.update_services(self._whisper_model, self._grammar_service)
        self._record_tab.set_qa_service(self._qa_service)
        self._refresh_devices()

        QMessageBox.information(
            self, "Settings Saved",
            "Settings saved successfully.\n"
            "Whisper model changes require a restart to take effect."
        )

    def _set_startup(self, enabled: bool) -> None:
        try:
            import sys
            import winreg
            # When frozen by PyInstaller sys.executable IS the .exe
            if getattr(sys, "frozen", False):
                app_path = f'"{sys.executable}"'
            else:
                app_path = f'"{sys.executable}" "{sys.argv[0]}"'
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0, winreg.KEY_SET_VALUE,
            )
            if enabled:
                winreg.SetValueEx(key, "NoteMe", 0, winreg.REG_SZ, app_path)
            else:
                try:
                    winreg.DeleteValue(key, "NoteMe")
                except FileNotFoundError:
                    pass
            winreg.CloseKey(key)
        except Exception as exc:
            log.warning("Could not update startup entry: %s", exc)

    # ------------------------------------------------------------------ #
    # Close → hide to tray
    # ------------------------------------------------------------------ #

    def closeEvent(self, event: QCloseEvent) -> None:
        event.ignore()
        self.hide()
