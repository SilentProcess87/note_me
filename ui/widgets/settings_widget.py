from __future__ import annotations

import logging
import time

from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)

_WHISPER_MODELS = [
    "tiny", "base", "small", "medium", "large-v3", "turbo",
    "ivrit-ai/whisper-large-v3-turbo-ct2",
]
_COMPUTE_TYPES = ["int8", "float16", "float32"]
_DEVICES = ["cpu", "cuda"]
_LANGUAGES = [("Auto Detect", "auto"), ("English", "en"), ("Hebrew (עברית)", "he")]
_SOURCE_MAP = {0: "both", 1: "system", 2: "mic"}
_SOURCE_IDX = {"both": 0, "system": 1, "mic": 2}


class _MicTestWorker(QThread):
    level_changed = pyqtSignal(int)
    completed = pyqtSignal(bool, str)

    def __init__(self, device_index: int, parent=None):
        super().__init__(parent)
        self._device_index = None if device_index == -1 else device_index

    def run(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd

            dev_info = sd.query_devices(self._device_index, "input")
            samplerate = int(dev_info["default_samplerate"])
            detected = False

            def _callback(indata, frames, time_info, status):
                nonlocal detected
                mono = indata[:, 0]
                rms = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
                level = max(0, min(100, int(rms * 500)))
                if level > 3:
                    detected = True
                self.level_changed.emit(level)

            with sd.InputStream(
                samplerate=samplerate,
                device=self._device_index,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=_callback,
            ):
                time.sleep(3.0)

            if detected:
                self.completed.emit(True, "Microphone signal detected successfully.")
            else:
                self.completed.emit(False, "No microphone signal detected. Speak louder or check the selected device.")
        except Exception as exc:
            self.completed.emit(False, f"Microphone test failed: {exc}")


class _SystemAudioTestWorker(QThread):
    level_changed = pyqtSignal(int)
    completed = pyqtSignal(bool, str)

    def __init__(self, device_index: int, parent=None):
        super().__init__(parent)
        self._device_index = device_index

    def run(self) -> None:
        try:
            import numpy as np
            import pyaudiowpatch as pyaudio
            import sounddevice as sd

            p = pyaudio.PyAudio()
            try:
                if self._device_index >= 0:
                    device = p.get_device_info_by_index(self._device_index)
                else:
                    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                    default_out = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                    device = None
                    for lb in p.get_loopback_device_info_generator():
                        if default_out["name"] in lb["name"]:
                            device = lb
                            break
                    if device is None:
                        self.completed.emit(False, "No WASAPI loopback device found.")
                        return

                samplerate = int(device["defaultSampleRate"])
                channels = max(1, device["maxInputChannels"])
                detected = False

                def _callback(in_data, frame_count, time_info, status):
                    nonlocal detected
                    raw = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                    if channels > 1:
                        raw = raw.reshape(-1, channels).mean(axis=1)
                    rms = float(np.sqrt(np.mean(raw ** 2))) if len(raw) else 0.0
                    level = max(0, min(100, int(rms * 500)))
                    if level > 3:
                        detected = True
                    self.level_changed.emit(level)
                    return (in_data, pyaudio.paContinue)

                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=samplerate,
                    input=True,
                    input_device_index=device["index"],
                    frames_per_buffer=1024,
                    stream_callback=_callback,
                )
                try:
                    stream.start_stream()
                    tone_sr = 48_000
                    duration = 1.5
                    t = np.linspace(0, duration, int(tone_sr * duration), endpoint=False)
                    tone = (0.15 * np.sin(2 * np.pi * 523.25 * t)).astype(np.float32)
                    sd.play(tone, tone_sr, blocking=False)
                    time.sleep(3.0)
                    try:
                        sd.stop()
                    except Exception:
                        pass
                finally:
                    stream.stop_stream()
                    stream.close()

                if detected:
                    self.completed.emit(True, "System audio signal detected successfully. You should also have heard a short test tone.")
                else:
                    self.completed.emit(False, "No system audio detected from the selected loopback source. If you heard the test tone, choose a different system-audio device and try again.")
            finally:
                p.terminate()
        except Exception as exc:
            self.completed.emit(False, f"System audio test failed: {exc}")


class SettingsWidget(QWidget):
    settings_saved = pyqtSignal(dict)
    refresh_devices_requested = pyqtSignal()

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config
        self._test_worker = None
        self._setup_ui()
        self._load_from_config()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        audio_grp = QGroupBox("Audio Devices")
        audio_form = QFormLayout(audio_grp)

        sys_row = QHBoxLayout()
        self._sys_combo = QComboBox()
        sys_row.addWidget(self._sys_combo, stretch=1)
        sys_ref = QPushButton("↻")
        sys_ref.setFixedWidth(30)
        sys_ref.clicked.connect(self.refresh_devices_requested)
        sys_row.addWidget(sys_ref)
        self._btn_test_system = QPushButton("Test")
        self._btn_test_system.setFixedWidth(60)
        self._btn_test_system.clicked.connect(self._start_system_test)
        sys_row.addWidget(self._btn_test_system)
        audio_form.addRow("System Audio (Loopback):", sys_row)

        mic_row = QHBoxLayout()
        self._mic_combo = QComboBox()
        mic_row.addWidget(self._mic_combo, stretch=1)
        mic_ref = QPushButton("↻")
        mic_ref.setFixedWidth(30)
        mic_ref.clicked.connect(self.refresh_devices_requested)
        mic_row.addWidget(mic_ref)
        self._btn_test_mic = QPushButton("Test")
        self._btn_test_mic.setFixedWidth(60)
        self._btn_test_mic.clicked.connect(self._start_mic_test)
        mic_row.addWidget(self._btn_test_mic)
        audio_form.addRow("Microphone:", mic_row)

        self._src_combo = QComboBox()
        self._src_combo.addItems(["Both (System + Mic)", "System Audio Only", "Microphone Only"])
        audio_form.addRow("Default Source:", self._src_combo)

        self._audio_test_status = QLabel("Use the Test buttons to verify your selected devices.")
        self._audio_test_status.setWordWrap(True)
        self._audio_test_status.setStyleSheet("color:#888; font-size:11px;")
        audio_form.addRow("Audio Test:", self._audio_test_status)

        self._audio_test_meter = QProgressBar()
        self._audio_test_meter.setRange(0, 100)
        self._audio_test_meter.setValue(0)
        self._audio_test_meter.setTextVisible(False)
        self._audio_test_meter.setFixedHeight(10)
        audio_form.addRow("Level:", self._audio_test_meter)
        root.addWidget(audio_grp)

        trans_grp = QGroupBox("Transcription  (Whisper)")
        trans_form = QFormLayout(trans_grp)

        self._model_combo = QComboBox()
        self._model_combo.addItems(_WHISPER_MODELS)
        self._model_combo.setEditable(True)
        trans_form.addRow("Model:", self._model_combo)

        self._compute_combo = QComboBox()
        self._compute_combo.addItems(_COMPUTE_TYPES)
        trans_form.addRow("Compute Type:", self._compute_combo)

        self._device_combo = QComboBox()
        self._device_combo.addItems(_DEVICES)
        trans_form.addRow("Device:", self._device_combo)

        self._lang_combo = QComboBox()
        for label, val in _LANGUAGES:
            self._lang_combo.addItem(label, val)
        trans_form.addRow("Default Language:", self._lang_combo)

        self._vocab_edit = QLineEdit()
        self._vocab_edit.setPlaceholderText("Kubernetes, pods, Docker, API, CI/CD, ...")
        self._vocab_edit.setToolTip(
            "Comma-separated list of technical terms and proper nouns.\n"
            "Helps Whisper correctly recognize these words when spoken\n"
            "in mixed-language conversations (e.g. Hebrew with English tech terms)."
        )
        trans_form.addRow("Vocabulary Hint:", self._vocab_edit)

        vocab_hint_lbl = QLabel(
            "Add tech terms, names, or acronyms that Whisper should recognize.\n"
            "Especially useful for Hebrew meetings with English tech vocabulary."
        )
        vocab_hint_lbl.setStyleSheet("color:#888; font-size:10px;")
        vocab_hint_lbl.setWordWrap(True)
        trans_form.addRow("", vocab_hint_lbl)
        root.addWidget(trans_grp)

        llm_grp = QGroupBox("LLM  (Summarization & Q&A)")
        llm_form = QFormLayout(llm_grp)

        self._provider_combo = QComboBox()
        from core.llm.provider import PROVIDERS
        for pid, label in PROVIDERS:
            self._provider_combo.addItem(label, pid)
        self._provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        llm_form.addRow("Provider:", self._provider_combo)

        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("sk-... or gsk_...")
        self._api_key_lbl = QLabel("API Key:")
        llm_form.addRow(self._api_key_lbl, self._api_key_edit)

        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("http://localhost:11434")
        self._url_lbl = QLabel("URL:")
        llm_form.addRow(self._url_lbl, self._url_edit)

        model_row = QHBoxLayout()
        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("llama3.1:8b")
        model_row.addWidget(self._model_edit, stretch=1)
        self._test_llm_btn = QPushButton("Test")
        self._test_llm_btn.setFixedWidth(60)
        self._test_llm_btn.clicked.connect(self._test_llm)
        model_row.addWidget(self._test_llm_btn)
        llm_form.addRow("Model:", model_row)

        self._llm_hint = QLabel("")
        self._llm_hint.setWordWrap(True)
        self._llm_hint.setStyleSheet("color:#888; font-size:10px;")
        llm_form.addRow("", self._llm_hint)
        root.addWidget(llm_grp)

        store_grp = QGroupBox("Storage")
        store_form = QFormLayout(store_grp)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        path_row.addWidget(self._path_edit, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(browse_btn)
        store_form.addRow("Storage Path:", path_row)

        self._startup_check = QCheckBox("Start NoteMe automatically with Windows")
        store_form.addRow("", self._startup_check)
        root.addWidget(store_grp)

        # ── Zoom / Meeting Integration ──────────────────────────────────
        zoom_grp = QGroupBox("Meeting Integration")
        zoom_form = QFormLayout(zoom_grp)

        self._zoom_check = QCheckBox(
            "Auto-record when a Zoom meeting starts  "
            "(captures system audio + microphone — stops automatically when meeting ends)"
        )
        zoom_form.addRow("", self._zoom_check)

        self._zoom_status = QLabel("Status: disabled")
        self._zoom_status.setStyleSheet("color:#888; font-size:11px;")
        zoom_form.addRow("Status:", self._zoom_status)
        self._zoom_check.stateChanged.connect(self._on_zoom_toggle)

        # Refresh the Zoom status label every 4 s so the user can see meeting detection
        from PyQt6.QtCore import QTimer as _QTimer
        self._zoom_poll = _QTimer(self)
        self._zoom_poll.setInterval(4000)
        self._zoom_poll.timeout.connect(self._on_zoom_toggle)
        self._zoom_poll.start()

        root.addWidget(zoom_grp)

        root.addStretch()
        save_btn = QPushButton("💾  Save Settings")
        save_btn.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;border-radius:5px;padding:8px 24px;}"
            "QPushButton:hover{background:#219a52;}"
        )
        save_btn.clicked.connect(self._save)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(save_btn)
        root.addLayout(row)

    def _load_from_config(self) -> None:
        cfg = self._config

        # LLM provider
        for i in range(self._provider_combo.count()):
            if self._provider_combo.itemData(i) == cfg.llm.provider:
                self._provider_combo.setCurrentIndex(i)
                break
        self._url_edit.setText(cfg.llm.base_url)
        self._model_edit.setText(cfg.llm.model)
        # Decrypt and load the API key for display
        from utils.secrets import decrypt_key
        self._api_key_edit.setText(decrypt_key(cfg.llm.api_key_encrypted))
        self._on_provider_changed()  # set field visibility

        self._path_edit.setText(str(cfg.app.resolved_storage_path))
        self._startup_check.setChecked(cfg.app.startup_with_windows)
        self._zoom_check.setChecked(cfg.app.zoom_auto_record)
        self._on_zoom_toggle()  # set initial status label

        idx = self._model_combo.findText(cfg.transcription.model)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)
        else:
            self._model_combo.setCurrentText(cfg.transcription.model)

        idx = self._compute_combo.findText(cfg.transcription.compute_type)
        if idx >= 0:
            self._compute_combo.setCurrentIndex(idx)

        idx = self._device_combo.findText(cfg.transcription.device)
        if idx >= 0:
            self._device_combo.setCurrentIndex(idx)

        for i in range(self._lang_combo.count()):
            if self._lang_combo.itemData(i) == cfg.transcription.default_language:
                self._lang_combo.setCurrentIndex(i)
                break

        self._src_combo.setCurrentIndex(_SOURCE_IDX.get(cfg.audio.default_source, 0))
        self._vocab_edit.setText(cfg.transcription.vocabulary_hint)

    def populate_devices(self, loopback_devices: list[dict], mic_devices: list[dict]) -> None:
        cfg = self._config

        self._sys_combo.clear()
        self._sys_combo.addItem("Auto (Default)", -1)
        for d in loopback_devices:
            self._sys_combo.addItem(d["name"], d["index"])
        for i in range(self._sys_combo.count()):
            if self._sys_combo.itemData(i) == cfg.audio.system_device_index:
                self._sys_combo.setCurrentIndex(i)
                break

        self._mic_combo.clear()
        self._mic_combo.addItem("Auto (Default)", -1)
        for d in mic_devices:
            self._mic_combo.addItem(d["name"], d["index"])
        for i in range(self._mic_combo.count()):
            if self._mic_combo.itemData(i) == cfg.audio.mic_device_index:
                self._mic_combo.setCurrentIndex(i)
                break

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Storage Folder", self._path_edit.text())
        if path:
            self._path_edit.setText(path)

    def _encrypt_api_key(self) -> str:
        """Encrypt the API key from the input field using DPAPI."""
        raw = self._api_key_edit.text().strip()
        if not raw:
            return ""
        from utils.secrets import encrypt_key
        return encrypt_key(raw)

    def _on_provider_changed(self) -> None:
        """Show/hide API key and URL fields based on selected provider."""
        provider = self._provider_combo.currentData() or "ollama"
        from core.llm.provider import needs_api_key, default_model_for_provider
        show_key = needs_api_key(provider)
        self._api_key_lbl.setVisible(show_key)
        self._api_key_edit.setVisible(show_key)

        # URL: show for ollama and custom only
        show_url = provider in ("ollama", "custom")
        self._url_lbl.setVisible(show_url)
        self._url_edit.setVisible(show_url)

        # Hint text
        hints = {
            "ollama": "Ollama runs locally. Install from ollama.com, then: ollama serve",
            "groq": "Free tier: 30 req/min. Get your key at console.groq.com",
            "openai": "Get your key at platform.openai.com/api-keys",
            "together": "Free $5 credit. Get key at api.together.xyz",
            "mistral": "EU-hosted. Get key at console.mistral.ai",
            "custom": "Enter any OpenAI-compatible endpoint URL above.",
        }
        self._llm_hint.setText(hints.get(provider, ""))

        # Set default model placeholder
        self._model_edit.setPlaceholderText(default_model_for_provider(provider))

    def _test_llm(self) -> None:
        """Test the currently configured LLM provider."""
        provider = self._provider_combo.currentData() or "ollama"
        from core.llm.provider import create_llm_client
        client = create_llm_client(
            provider=provider,
            api_key=self._api_key_edit.text(),
            base_url=self._url_edit.text(),
            timeout=10,
        )
        if client.is_available():
            models = client.list_models()
            model_list = "\n".join(models[:15]) or "(no models listed)"
            QMessageBox.information(self, "LLM Test", f"\u2713  Connected!\n\nModels:\n{model_list}")
        else:
            QMessageBox.warning(self, "LLM Test", "\u2717  Could not connect. Check your provider, URL, and API key.")

    def _on_zoom_toggle(self) -> None:
        """Update the Zoom status label based on current checkbox state."""
        if self._zoom_check.isChecked():
            from core.zoom_watcher import zoom_meeting_active
            try:
                in_meeting = zoom_meeting_active()
                self._zoom_status.setText(
                    "\u25cf  Meeting in progress" if in_meeting
                    else "\u25cb  Zoom running, not in meeting — waiting for meeting to start"
                )
                self._zoom_status.setStyleSheet(
                    "color:#27ae60; font-size:11px;" if in_meeting
                    else "color:#888; font-size:11px;"
                )
            except Exception:
                self._zoom_status.setText("Enabled — will auto-record when Zoom meeting starts")
                self._zoom_status.setStyleSheet("color:#888; font-size:11px;")
        else:
            self._zoom_status.setText("Disabled")
            self._zoom_status.setStyleSheet("color:#888; font-size:11px;")

    def _set_test_buttons_enabled(self, enabled: bool) -> None:
        self._btn_test_mic.setEnabled(enabled)
        self._btn_test_system.setEnabled(enabled)

    def _start_mic_test(self) -> None:
        if self._test_worker and self._test_worker.isRunning():
            return
        self._audio_test_meter.setValue(0)
        self._audio_test_status.setText("Testing microphone for 3 seconds — speak now.")
        self._set_test_buttons_enabled(False)
        self._test_worker = _MicTestWorker(self._mic_combo.currentData())
        self._test_worker.level_changed.connect(self._audio_test_meter.setValue)
        self._test_worker.completed.connect(self._on_test_completed)
        self._test_worker.start()

    def _start_system_test(self) -> None:
        if self._test_worker and self._test_worker.isRunning():
            return
        self._audio_test_meter.setValue(0)
        self._audio_test_status.setText("Testing system audio for 3 seconds — play some audio now.")
        self._set_test_buttons_enabled(False)
        self._test_worker = _SystemAudioTestWorker(self._sys_combo.currentData())
        self._test_worker.level_changed.connect(self._audio_test_meter.setValue)
        self._test_worker.completed.connect(self._on_test_completed)
        self._test_worker.start()

    @pyqtSlot(bool, str)
    def _on_test_completed(self, ok: bool, message: str) -> None:
        self._set_test_buttons_enabled(True)
        self._audio_test_status.setText(("✓  " if ok else "✗  ") + message)
        if not ok:
            self._audio_test_meter.setValue(0)

    def _save(self) -> None:
        settings = {
            "audio": {
                "system_device_index": self._sys_combo.currentData() if self._sys_combo.currentData() is not None else -1,
                "mic_device_index": self._mic_combo.currentData() if self._mic_combo.currentData() is not None else -1,
                "default_source": _SOURCE_MAP.get(self._src_combo.currentIndex(), "both"),
            },
            "transcription": {
                "model": self._model_combo.currentText(),
                "compute_type": self._compute_combo.currentText(),
                "device": self._device_combo.currentText(),
                "default_language": self._lang_combo.currentData(),
                "vocabulary_hint": self._vocab_edit.text(),
            },
            "llm": {
                "provider": self._provider_combo.currentData() or "ollama",
                "base_url": self._url_edit.text() or "http://localhost:11434",
                "model": self._model_edit.text() or self._model_edit.placeholderText(),
                "api_key_encrypted": self._encrypt_api_key(),
            },
            "app": {
                "storage_path": self._path_edit.text(),
                "startup_with_windows": self._startup_check.isChecked(),
                "zoom_auto_record": self._zoom_check.isChecked(),
            },
        }
        self.settings_saved.emit(settings)
