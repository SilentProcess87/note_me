from __future__ import annotations

import logging

from PyQt6.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)

_LANGUAGES = [("Auto Detect", "auto"), ("English", "en"), ("Hebrew (עברית)", "he")]
_SOURCES = [
    ("System Audio  (captures meetings)", "system"),
    ("Microphone", "mic"),
    ("Both  (System + Mic)", "both"),
]

_BTN_START = "⏺  Start Recording"
_BTN_STOP = "⏹  Stop Recording"

_STYLE_START = """
QPushButton { background-color:#27ae60; color:white; border-radius:6px; padding:8px 20px; }
QPushButton:hover { background-color:#219a52; }
"""
_STYLE_STOP = """
QPushButton { background-color:#e74c3c; color:white; border-radius:6px; padding:8px 20px; }
QPushButton:hover { background-color:#c0392b; }
"""


class _LiveQAWorker(QThread):
    """Runs a Q&A query against the live transcript in the background."""
    answer_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, question: str, qa_service, transcript: str, language: str, parent=None):
        super().__init__(parent)
        self._question = question
        self._qa = qa_service
        self._transcript = transcript
        self._language = language

    def run(self) -> None:
        try:
            ans = self._qa.answer(self._question, self._transcript, self._language)
            self.answer_ready.emit(ans)
        except Exception as exc:
            self.error.emit(str(exc))


class RecordingWidget(QWidget):
    start_requested = pyqtSignal(str, str)   # source, language
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_recording = False
        self._qa_service = None
        self._qa_workers: list[_LiveQAWorker] = []
        self._setup_ui()

    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # Settings group
        grp = QGroupBox("Recording Settings")
        grp_layout = QVBoxLayout(grp)

        grp_layout.addWidget(QLabel("Audio Source:"))
        self._source_group = QButtonGroup(self)
        src_row = QHBoxLayout()
        self._source_radios: dict[str, QRadioButton] = {}
        for label, val in _SOURCES:
            rb = QRadioButton(label)
            rb.setProperty("val", val)
            self._source_radios[val] = rb
            self._source_group.addButton(rb)
            src_row.addWidget(rb)
        self._source_radios["both"].setChecked(True)
        grp_layout.addLayout(src_row)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:"))
        self._lang_combo = QComboBox()
        for label, val in _LANGUAGES:
            self._lang_combo.addItem(label, val)
        lang_row.addWidget(self._lang_combo)
        lang_row.addStretch()
        grp_layout.addLayout(lang_row)
        root.addWidget(grp)

        # Record button
        btn_row = QHBoxLayout()
        self._btn = QPushButton(_BTN_START)
        self._btn.setFixedHeight(50)
        fnt = QFont()
        fnt.setPointSize(13)
        fnt.setBold(True)
        self._btn.setFont(fnt)
        self._btn.setStyleSheet(_STYLE_START)
        self._btn.clicked.connect(self._toggle)
        btn_row.addStretch()
        btn_row.addWidget(self._btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Status + transcription progress row
        status_row = QHBoxLayout()
        self._status = QLabel("Ready to record")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color:#aaa; font-style:italic;")
        status_row.addWidget(self._status, stretch=1)

        self._progress_label = QLabel("")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setStyleSheet("color:#f39c12; font-size:11px;")
        self._progress_label.setVisible(False)
        status_row.addWidget(self._progress_label)
        root.addLayout(status_row)

        # Audio level meters
        levels_row = QHBoxLayout()
        levels_row.setSpacing(8)

        levels_row.addWidget(QLabel("🎤"))
        self._mic_bar = QProgressBar()
        self._mic_bar.setRange(0, 100)
        self._mic_bar.setValue(0)
        self._mic_bar.setTextVisible(False)
        self._mic_bar.setFixedHeight(10)
        self._mic_bar.setStyleSheet(
            "QProgressBar { border:1px solid #555; border-radius:4px; background:#222; }"
            "QProgressBar::chunk { background:#8e44ad; border-radius:3px; }"
        )
        levels_row.addWidget(self._mic_bar, stretch=1)

        levels_row.addSpacing(12)
        levels_row.addWidget(QLabel("🔊"))
        self._sys_bar = QProgressBar()
        self._sys_bar.setRange(0, 100)
        self._sys_bar.setValue(0)
        self._sys_bar.setTextVisible(False)
        self._sys_bar.setFixedHeight(10)
        self._sys_bar.setStyleSheet(
            "QProgressBar { border:1px solid #555; border-radius:4px; background:#222; }"
            "QProgressBar::chunk { background:#2980b9; border-radius:3px; }"
        )
        levels_row.addWidget(self._sys_bar, stretch=1)

        root.addLayout(levels_row)

        # Splitter: live transcript (top) + live Q&A (bottom)
        splitter = QSplitter(Qt.Orientation.Vertical)

        # -- Live transcript
        t_grp = QGroupBox("Live Transcript")
        t_layout = QVBoxLayout(t_grp)
        self._transcript = QTextEdit()
        self._transcript.setReadOnly(True)
        self._transcript.setPlaceholderText("Transcript appears here as you record…")
        self._transcript.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        t_layout.addWidget(self._transcript)
        splitter.addWidget(t_grp)

        # -- Live Q&A panel
        qa_grp = QGroupBox("💬  Ask about this meeting (live)")
        qa_layout = QVBoxLayout(qa_grp)
        self._qa_chat = QTextEdit()
        self._qa_chat.setReadOnly(True)
        self._qa_chat.setPlaceholderText(
            "Ask questions about the meeting while it’s in progress…\n"
            "e.g. \"What did they say about pods?\" or \"Summarize the last 5 minutes\""
        )
        self._qa_chat.setMaximumHeight(180)
        qa_layout.addWidget(self._qa_chat)

        qa_input_row = QHBoxLayout()
        self._qa_input = QLineEdit()
        self._qa_input.setPlaceholderText("Type your question and press Enter…")
        self._qa_input.returnPressed.connect(self._submit_live_qa)
        qa_input_row.addWidget(self._qa_input)
        self._qa_btn = QPushButton("Ask")
        self._qa_btn.setFixedWidth(60)
        self._qa_btn.clicked.connect(self._submit_live_qa)
        qa_input_row.addWidget(self._qa_btn)
        qa_layout.addLayout(qa_input_row)
        splitter.addWidget(qa_grp)

        splitter.setSizes([500, 180])
        root.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------ #
    # Public slots
    # ------------------------------------------------------------------ #

    def set_recording(self, recording: bool) -> None:
        self._is_recording = recording
        if recording:
            self._btn.setText(_BTN_STOP)
            self._btn.setStyleSheet(_STYLE_STOP)
            self._status.setText("🔴  Recording…")
            self._transcript.clear()
            self._qa_chat.clear()
            self._progress_label.setText("Waiting for first transcription...")
            self._progress_label.setStyleSheet("color:#f39c12; font-size:11px;")
            self._progress_label.setVisible(True)
        else:
            self._btn.setText(_BTN_START)
            self._btn.setStyleSheet(_STYLE_START)
            self._status.setText("Processing… generating summary")
            self._progress_label.setVisible(False)
            self._mic_bar.setValue(0)
            self._sys_bar.setValue(0)

    def update_levels(self, mic_rms: float, sys_rms: float) -> None:
        """Called ~12 Hz while recording to animate the level meters."""
        self._mic_bar.setValue(min(100, int(mic_rms * 600)))
        self._sys_bar.setValue(min(100, int(sys_rms * 600)))

    def append_segment(self, start_sec: float, text: str) -> None:
        m, s = divmod(int(start_sec), 60)
        self._transcript.append(
            f'<span style="color:#888;">[{m:02d}:{s:02d}]</span> {text}'
        )
        cursor = self._transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._transcript.setTextCursor(cursor)

    def update_transcription_progress(self, chunks_done: int, chunks_remaining: int) -> None:
        """Show live transcription processing progress."""
        if not self._is_recording and chunks_remaining == 0:
            self._progress_label.setVisible(False)
            return
        self._progress_label.setVisible(True)
        if chunks_remaining > 0:
            self._progress_label.setText(
                f"Transcribing... {chunks_done} done, {chunks_remaining} queued"
            )
            self._progress_label.setStyleSheet("color:#f39c12; font-size:11px;")
        else:
            self._progress_label.setText(f"Transcribed {chunks_done} chunks")
            self._progress_label.setStyleSheet("color:#27ae60; font-size:11px;")

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_qa_service(self, qa_service) -> None:
        """Set the QA service so live questions can be answered."""
        self._qa_service = qa_service

    # ------------------------------------------------------------------ #
    # Live Q&A
    # ------------------------------------------------------------------ #

    def _submit_live_qa(self) -> None:
        question = self._qa_input.text().strip()
        if not question:
            return
        if not self._qa_service:
            self._qa_chat.append('<span style="color:#e74c3c;">Ollama is not running.</span>')
            return

        # Grab the current live transcript text
        transcript = self._transcript.toPlainText()
        if not transcript.strip():
            self._qa_chat.append('<span style="color:#888;">No transcript yet — wait for some speech.</span>')
            return

        self._qa_input.clear()
        self._qa_btn.setEnabled(False)
        self._qa_chat.append(f'<b>You:</b> {question}')
        self._qa_chat.append('<i style="color:#888;">Thinking…</i>')

        worker = _LiveQAWorker(question, self._qa_service, transcript, "auto", self)
        worker.answer_ready.connect(self._on_live_qa_answer)
        worker.error.connect(self._on_live_qa_error)
        self._qa_workers.append(worker)
        worker.start()

    @pyqtSlot(str)
    def _on_live_qa_answer(self, answer: str) -> None:
        # Replace the "Thinking…" line
        cursor = self._qa_chat.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        self._qa_chat.setTextCursor(cursor)
        self._qa_chat.append(f'<b>NoteMe:</b> {answer}<br>')
        self._qa_btn.setEnabled(True)

    @pyqtSlot(str)
    def _on_live_qa_error(self, err: str) -> None:
        self._qa_chat.append(f'<span style="color:#e74c3c;">Error: {err}</span><br>')
        self._qa_btn.setEnabled(True)

    # ------------------------------------------------------------------ #

    def _toggle(self) -> None:
        if not self._is_recording:
            source = next(
                (rb.property("val") for rb in self._source_radios.values() if rb.isChecked()),
                "both",
            )
            language = self._lang_combo.currentData()
            self.start_requested.emit(source, language)
        else:
            self.stop_requested.emit()
