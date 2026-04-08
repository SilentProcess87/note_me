from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)

_LANGUAGES = [("Auto Detect", "auto"), ("English", "en"), ("Hebrew (עברית)", "he")]

_STYLE_RECORD = """
QPushButton { background-color:#8e44ad; color:white; border-radius:7px; font-size:13pt; font-weight:bold; }
QPushButton:hover { background-color:#7d3c98; }
"""
_STYLE_STOP = """
QPushButton { background-color:#e74c3c; color:white; border-radius:7px; font-size:13pt; font-weight:bold; }
QPushButton:hover { background-color:#c0392b; }
"""
_STYLE_COPY = """
QPushButton { background-color:#2980b9; color:white; border-radius:5px; font-size:11pt; font-weight:bold; padding:6px; }
QPushButton:hover { background-color:#2471a3; }
QPushButton:disabled { background-color:#555; color:#888; }
"""


class _CoachWorker(QThread):
    transcription_ready = pyqtSignal(str, str)
    improved_ready = pyqtSignal(str, str)
    error = pyqtSignal(str)

    def __init__(self, audio: np.ndarray, language: str, whisper_model, grammar_service, parent=None):
        super().__init__(parent)
        self._audio = audio
        self._language = language
        self._whisper = whisper_model
        self._grammar = grammar_service

    def run(self) -> None:
        try:
            lang_arg = self._language if self._language != "auto" else None
            segments, info = self._whisper.transcribe(
                self._audio,
                language=lang_arg,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            detected = getattr(info, "language", self._language or "en")
        except Exception as exc:
            self.error.emit(f"Transcription failed: {exc}")
            return

        if not text:
            self.error.emit("No speech detected. Please speak clearly and try again.")
            return

        self.transcription_ready.emit(text, detected)

        if self._grammar:
            try:
                result = self._grammar.improve(text, detected)
                self.improved_ready.emit(result.improved, result.notes)
            except Exception as exc:
                self.error.emit(f"LLM improvement failed: {exc}")


class SpeechCoachWidget(QWidget):
    def __init__(self, whisper_model=None, grammar_service=None, parent=None):
        super().__init__(parent)
        self._whisper_model = whisper_model
        self._grammar_service = grammar_service
        self._mic_device_index: int = -1
        self._mic_device_name: str = "Auto (Default)"
        self._system_device_name: str = "Auto (Default)"
        self._is_recording = False
        self._mic_sr: int = 16_000
        self._audio_chunks: list[np.ndarray] = []
        self._sd_stream = None
        self._worker: Optional[_CoachWorker] = None
        self._copy_timer = QTimer(self)
        self._copy_timer.setSingleShot(True)
        self._copy_timer.timeout.connect(self._reset_copy_label)

        self._recording_timer = QTimer(self)
        self._recording_timer.setInterval(80)
        self._recording_timer.timeout.connect(self._update_recording_feedback)
        self._record_started_at: float = 0.0
        self._current_level: int = 0

        self._setup_ui()

    def update_services(self, whisper_model=None, grammar_service=None) -> None:
        self._whisper_model = whisper_model
        self._grammar_service = grammar_service

    def update_audio_devices(
        self,
        mic_device_index: int = -1,
        mic_device_name: str = "Auto (Default)",
        system_device_name: str = "Auto (Default)",
    ) -> None:
        self._mic_device_index = mic_device_index
        self._mic_device_name = mic_device_name
        self._system_device_name = system_device_name
        self._device_lbl.setText(
            f"Using microphone: {self._mic_device_name}\n"
            f"System audio source: {self._system_device_name}"
        )

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)

        hdr = QLabel("Speech Coach")
        hdr_font = QFont()
        hdr_font.setPointSize(16)
        hdr_font.setBold(True)
        hdr.setFont(hdr_font)
        root.addWidget(hdr)

        desc = QLabel(
            "Record yourself speaking. NoteMe will transcribe your words and generate a "
            "polished, grammar-corrected version you can copy with one click."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#999;")
        root.addWidget(desc)

        self._device_lbl = QLabel("Using microphone: Auto (Default)\nSystem audio source: Auto (Default)")
        self._device_lbl.setWordWrap(True)
        self._device_lbl.setStyleSheet("color:#8aa; font-size:11px;")
        root.addWidget(self._device_lbl)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:"))
        self._lang_combo = QComboBox()
        for label, val in _LANGUAGES:
            self._lang_combo.addItem(label, val)
        lang_row.addWidget(self._lang_combo)
        lang_row.addStretch()
        root.addLayout(lang_row)

        self._btn = QPushButton("🎙  Start Speaking")
        self._btn.setFixedHeight(54)
        self._btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._btn.setStyleSheet(_STYLE_RECORD)
        self._btn.clicked.connect(self._toggle)
        root.addWidget(self._btn)

        self._status = QLabel("Press the button and start speaking")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color:#aaa; font-style:italic;")
        root.addWidget(self._status)

        feedback_row = QHBoxLayout()
        self._record_dot = QLabel("⚫")
        self._record_dot.setStyleSheet("color:#777; font-size:18px;")
        feedback_row.addWidget(self._record_dot)

        self._elapsed_lbl = QLabel("00:00")
        self._elapsed_lbl.setStyleSheet("color:#bbb; font-weight:bold;")
        feedback_row.addWidget(self._elapsed_lbl)

        feedback_row.addWidget(QLabel("Mic Level:"))
        self._level_bar = QProgressBar()
        self._level_bar.setRange(0, 100)
        self._level_bar.setValue(0)
        self._level_bar.setTextVisible(False)
        self._level_bar.setFixedHeight(12)
        feedback_row.addWidget(self._level_bar, stretch=1)
        root.addLayout(feedback_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedHeight(4)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_grp = QGroupBox("What You Said  (Original)")
        left_layout = QVBoxLayout(left_grp)
        self._original_text = QTextEdit()
        self._original_text.setReadOnly(True)
        self._original_text.setPlaceholderText("Your transcription will appear here…")
        left_layout.addWidget(self._original_text)
        copy_orig_btn = QPushButton("Copy")
        copy_orig_btn.setFixedWidth(80)
        copy_orig_btn.clicked.connect(self._copy_original)
        orig_btn_row = QHBoxLayout()
        orig_btn_row.addStretch()
        orig_btn_row.addWidget(copy_orig_btn)
        left_layout.addLayout(orig_btn_row)
        splitter.addWidget(left_grp)

        right_grp = QGroupBox("Improved Version")
        right_layout = QVBoxLayout(right_grp)
        self._improved_text = QTextEdit()
        self._improved_text.setReadOnly(True)
        self._improved_text.setPlaceholderText("The grammar-corrected, polished version will appear here…")
        right_layout.addWidget(self._improved_text)

        self._notes_lbl = QLabel("")
        self._notes_lbl.setWordWrap(True)
        self._notes_lbl.setStyleSheet("color:#888; font-size:10px;")
        self._notes_lbl.setVisible(False)
        right_layout.addWidget(self._notes_lbl)

        self._copy_btn = QPushButton("📋  Copy Improved Text")
        self._copy_btn.setFixedHeight(42)
        self._copy_btn.setStyleSheet(_STYLE_COPY)
        self._copy_btn.clicked.connect(self._copy_improved)
        right_layout.addWidget(self._copy_btn)
        splitter.addWidget(right_grp)

        splitter.setSizes([420, 420])
        root.addWidget(splitter, stretch=1)

    def _toggle(self) -> None:
        if not self._is_recording:
            self._start_recording()
        else:
            self._stop_and_process()

    def _start_recording(self) -> None:
        self._audio_chunks.clear()
        self._original_text.clear()
        self._improved_text.clear()
        self._notes_lbl.setVisible(False)
        self._level_bar.setValue(0)
        self._current_level = 0
        self._is_recording = True
        self._record_started_at = time.time()

        self._btn.setText("⏹  Stop Speaking")
        self._btn.setStyleSheet(_STYLE_STOP)
        self._record_dot.setText("🔴")
        self._status.setText("Recording has started — speak now, and watch the mic level meter.")
        self._recording_timer.start()

        try:
            import sounddevice as sd

            selected_device = None if self._mic_device_index == -1 else self._mic_device_index
            dev_info = sd.query_devices(selected_device, "input")
            self._mic_sr = int(dev_info["default_samplerate"])
            active_name = self._mic_device_name if self._mic_device_index != -1 else dev_info["name"]
            self._device_lbl.setText(
                f"Using microphone: {active_name}\n"
                f"System audio source: {self._system_device_name}"
            )

            def _cb(indata, frames, time_info, status):
                mono = indata[:, 0].copy().astype(np.float32)
                self._audio_chunks.append(mono)
                rms = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
                self._current_level = max(0, min(100, int(rms * 500)))

            self._sd_stream = sd.InputStream(
                samplerate=self._mic_sr,
                device=selected_device,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=_cb,
            )
            self._sd_stream.start()
        except Exception as exc:
            self._recording_timer.stop()
            self._status.setText(f"Microphone error: {exc}")
            self._record_dot.setText("⚫")
            self._is_recording = False
            self._btn.setText("🎙  Start Speaking")
            self._btn.setStyleSheet(_STYLE_RECORD)

    def _stop_and_process(self) -> None:
        self._is_recording = False
        self._recording_timer.stop()
        self._record_dot.setText("⚫")
        self._btn.setText("🎙  Start Speaking")
        self._btn.setStyleSheet(_STYLE_RECORD)

        if self._sd_stream:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None

        if not self._audio_chunks:
            self._status.setText("No audio recorded.")
            self._level_bar.setValue(0)
            return

        raw = np.concatenate(self._audio_chunks).astype(np.float32)
        self._audio_chunks.clear()

        if self._mic_sr != 16_000:
            from core.audio.resample import resample_to_16k
            audio = resample_to_16k(raw, self._mic_sr)
        else:
            audio = raw

        if self._whisper_model is None:
            self._status.setText("Whisper model still loading — please wait a moment and try again.")
            self._level_bar.setValue(0)
            return

        self._status.setText("Transcribing…")
        self._progress.setVisible(True)

        language = self._lang_combo.currentData()
        self._worker = _CoachWorker(
            audio=audio,
            language=language,
            whisper_model=self._whisper_model,
            grammar_service=self._grammar_service,
            parent=self,
        )
        self._worker.transcription_ready.connect(self._on_transcription)
        self._worker.improved_ready.connect(self._on_improved)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(lambda: self._progress.setVisible(False))
        self._worker.start()

    def _update_recording_feedback(self) -> None:
        if not self._is_recording:
            return
        elapsed = int(time.time() - self._record_started_at)
        minutes, seconds = divmod(elapsed, 60)
        self._elapsed_lbl.setText(f"{minutes:02d}:{seconds:02d}")
        self._level_bar.setValue(self._current_level)

    @pyqtSlot(str, str)
    def _on_transcription(self, text: str, language: str) -> None:
        self._original_text.setPlainText(text)
        if self._grammar_service:
            self._status.setText("Transcribed. Generating improved version…")
        else:
            self._status.setText("Done. (Start Ollama for grammar improvement)")

    @pyqtSlot(str, str)
    def _on_improved(self, improved: str, notes: str) -> None:
        self._improved_text.setPlainText(improved)
        if notes:
            self._notes_lbl.setText(f"Changes made: {notes}")
            self._notes_lbl.setVisible(True)
        self._status.setText("✓  Ready — copy the improved text using the button below")

    @pyqtSlot(str)
    def _on_error(self, error: str) -> None:
        self._status.setText(f"Error: {error}")
        self._progress.setVisible(False)

    def _copy_original(self) -> None:
        text = self._original_text.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self._status.setText("Original text copied to clipboard!")

    def _copy_improved(self) -> None:
        text = self._improved_text.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self._copy_btn.setText("✓  Copied!")
            self._status.setText("✓  Improved text copied to clipboard!")
            self._copy_timer.start(2000)

    def _reset_copy_label(self) -> None:
        self._copy_btn.setText("📋  Copy Improved Text")
