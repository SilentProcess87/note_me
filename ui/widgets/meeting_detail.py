from __future__ import annotations

import logging

from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_DECISIONS_SEP    = "<<<DECISIONS>>>"
_PARTICIPANTS_SEP = "<<<PARTICIPANTS>>>"

log = logging.getLogger(__name__)


class _QAWorker(QThread):
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


class _SummaryGenWorker(QThread):
    """Run summarization from the meeting detail dialog (on-demand)."""
    done = pyqtSignal(str, str)  # summary_text, action_items
    error = pyqtSignal(str)

    def __init__(self, transcript: str, language: str, llm_client, llm_model: str, session_id: int, parent=None):
        super().__init__(parent)
        self._transcript = transcript
        self._language = language
        self._client = llm_client
        self._model = llm_model
        self._session_id = session_id

    def run(self) -> None:
        try:
            from core.llm.summarizer import Summarizer
            s = Summarizer(self._client, self._model)
            result = s.summarize(self._transcript, self._language)
            # Save to DB
            from core.storage.database import get_db
            from core.storage.models import Summary, Session as DbSession
            db = get_db()
            try:
                db.add(Summary(
                    session_id=self._session_id,
                    summary_text=result.summary,
                    action_items=result.action_items,
                    ollama_model=self._model,
                ))
                if result.title:
                    rec = db.get(DbSession, self._session_id)
                    if rec:
                        date_str = rec.start_time.strftime("%Y-%m-%d %H:%M") if rec.start_time else ""
                        rec.title = f"{result.title}  ({date_str})" if date_str else result.title
                db.commit()
            finally:
                db.close()
            self.done.emit(result.summary, result.action_items)
        except Exception as exc:
            self.error.emit(str(exc))


class MeetingDetailDialog(QDialog):
    def __init__(self, session_data: dict, qa_service=None, llm_client=None, llm_model: str = "", parent=None):
        super().__init__(parent)
        self._session = session_data
        self._session_id: int = session_data.get("id", -1)
        self._qa_service = qa_service
        self._llm_client = llm_client
        self._llm_model = llm_model
        self._transcript = session_data.get("transcript", "")
        self._language = session_data.get("language", "auto")
        self._qa_workers: list[_QAWorker] = []
        self._summary_gen_worker = None
        self._summary_loaded = bool(session_data.get("summary"))
        self._poll_ticks = 0
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(2500)
        self._poll_timer.timeout.connect(self._poll_db)
        self.setWindowTitle(f"Meeting \u2014 {session_data.get('title', '')}")
        self.resize(860, 620)
        self._setup_ui()
        if not self._summary_loaded:
            self._poll_timer.start()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        tabs = QTabWidget()

        # ── Transcript tab ────────────────────────────────────────────────────
        tw = QWidget()
        tl = QVBoxLayout(tw)
        self._t_edit = QTextEdit()
        self._t_edit.setReadOnly(True)
        self._t_edit.setPlainText(self._transcript)
        tl.addWidget(self._t_edit)

        t_btn_row = QHBoxLayout()
        t_btn_row.addStretch()
        copy_t_btn = QPushButton("📋  Copy Transcript")
        copy_t_btn.setFixedHeight(32)
        copy_t_btn.setStyleSheet(
            "QPushButton{background:#2980b9;color:white;border-radius:5px;padding:4px 16px;}"
            "QPushButton:hover{background:#2471a3;}"
        )
        copy_t_btn.clicked.connect(self._copy_transcript)
        t_btn_row.addWidget(copy_t_btn)

        copy_text_btn = QPushButton("📋  Copy Text Only")
        copy_text_btn.setFixedHeight(32)
        copy_text_btn.setToolTip("Copy transcript without timestamps")
        copy_text_btn.setStyleSheet(
            "QPushButton{background:#7f8c8d;color:white;border-radius:5px;padding:4px 16px;}"
            "QPushButton:hover{background:#636e72;}"
        )
        copy_text_btn.clicked.connect(self._copy_text_only)
        t_btn_row.addWidget(copy_text_btn)

        export_btn = QPushButton("💾  Export to File")

        # Audio playback row (only if audio file exists)
        self._audio_path = self._session.get("audio_path", "")
        if self._audio_path:
            import os
            if os.path.isfile(self._audio_path):
                audio_row = QHBoxLayout()
                audio_row.addStretch()

                play_btn = QPushButton("▶️  Play Recording")
                play_btn.setFixedHeight(32)
                play_btn.setStyleSheet(
                    "QPushButton{background:#8e44ad;color:white;border-radius:5px;padding:4px 16px;}"
                    "QPushButton:hover{background:#7d3c98;}"
                )
                play_btn.clicked.connect(self._play_audio)
                audio_row.addWidget(play_btn)

                save_audio_btn = QPushButton("💾  Save Audio")
                save_audio_btn.setFixedHeight(32)
                save_audio_btn.setStyleSheet(
                    "QPushButton{background:#27ae60;color:white;border-radius:5px;padding:4px 16px;}"
                    "QPushButton:hover{background:#219a52;}"
                )
                save_audio_btn.clicked.connect(self._save_audio)
                audio_row.addWidget(save_audio_btn)

                size_mb = os.path.getsize(self._audio_path) / 1e6
                audio_label = QLabel(f"  ({size_mb:.1f} MB)")
                audio_label.setStyleSheet("color:#888; font-size:10px;")
                audio_row.addWidget(audio_label)
                tl.addLayout(audio_row)
        export_btn.setFixedHeight(32)
        export_btn.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;border-radius:5px;padding:4px 16px;}"
            "QPushButton:hover{background:#219a52;}"
        )
        export_btn.clicked.connect(self._export_transcript)
        t_btn_row.addWidget(export_btn)
        tl.addLayout(t_btn_row)

        tabs.addTab(tw, "📄  Transcript")

        # ── Summary tab ────────────────────────────────────────────────────
        summary_raw = self._session.get("summary", "") or "No summary available yet."
        action_items_raw = self._session.get("action_items", "")

        # Decode tasks / decisions / participants from the encoded action_items field
        participants_text = ""
        work = action_items_raw
        if _PARTICIPANTS_SEP in work:
            work, participants_text = work.split(_PARTICIPANTS_SEP, 1)
            participants_text = participants_text.strip()
        if _DECISIONS_SEP in work:
            tasks_text, decisions_text = work.split(_DECISIONS_SEP, 1)
            tasks_text = tasks_text.strip()
            decisions_text = decisions_text.strip()
        else:
            tasks_text = work.strip()
            decisions_text = ""

        sw = QWidget()
        sw_layout = QVBoxLayout(sw)
        sw_layout.setSpacing(10)

        # -- Participants header (shown when available)
        if participants_text or not self._summary_loaded:
            self._participants_lbl = QLabel(
                f"\U0001f465  {participants_text}" if participants_text
                else "\u23f3  Identifying participants\u2026"
            )
            self._participants_lbl.setWordWrap(True)
            self._participants_lbl.setStyleSheet(
                "color:#aaa; font-size:11px; padding:4px 0;"
            )
            sw_layout.addWidget(self._participants_lbl)
        else:
            self._participants_lbl = None

        # -- Summary section
        sum_grp = QGroupBox("📝  Meeting Summary")
        sum_grp.setStyleSheet("QGroupBox { font-weight:bold; font-size:12pt; }")
        sg = QVBoxLayout(sum_grp)
        self._sum_edit = QTextEdit()
        self._sum_edit.setReadOnly(True)
        self._sum_edit.setPlainText(
            summary_raw if self._summary_loaded
            else "\u23f3  Generating summary\u2026 (this may take up to 30 seconds)"
        )
        self._sum_edit.setMinimumHeight(140)
        sg.addWidget(self._sum_edit)
        sw_layout.addWidget(sum_grp)

        # -- Decisions section
        dec_grp = QGroupBox("\u2705  Decisions")
        dec_grp.setStyleSheet("QGroupBox { font-weight:bold; font-size:12pt; }")
        dg = QVBoxLayout(dec_grp)
        self._dec_edit = QTextEdit()
        self._dec_edit.setReadOnly(True)
        self._dec_edit.setPlainText(
            decisions_text if self._summary_loaded else "\u23f3  Pending\u2026"
        )
        self._dec_edit.setMinimumHeight(100)
        self._dec_edit.setMaximumHeight(180)
        dg.addWidget(self._dec_edit)
        sw_layout.addWidget(dec_grp)

        # -- Tasks section
        task_grp = QGroupBox("📌  Tasks & Action Items")
        task_grp.setStyleSheet("QGroupBox { font-weight:bold; font-size:12pt; }")
        tg = QVBoxLayout(task_grp)
        self._task_edit = QTextEdit()
        self._task_edit.setReadOnly(True)
        self._task_edit.setPlainText(
            tasks_text if self._summary_loaded else "\u23f3  Pending\u2026"
        )
        self._task_edit.setMinimumHeight(100)
        self._task_edit.setMaximumHeight(180)
        tg.addWidget(self._task_edit)
        sw_layout.addWidget(task_grp)

        # Action buttons row
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        # Generate / Re-generate summary button
        self._gen_btn = QPushButton("🔄  Generate Summary")
        self._gen_btn.setFixedHeight(32)
        self._gen_btn.setStyleSheet(
            "QPushButton{background:#e67e22;color:white;border-radius:5px;padding:4px 16px;}"
            "QPushButton:hover{background:#d35400;}"
        )
        self._gen_btn.clicked.connect(self._generate_summary)
        self._gen_btn.setVisible(not self._summary_loaded)  # show only when no summary
        btn_row.addWidget(self._gen_btn)

        copy_all_btn = QPushButton("📋  Copy Full Report")
        copy_all_btn.setFixedHeight(32)
        copy_all_btn.setStyleSheet(
            "QPushButton{background:#8e44ad;color:white;border-radius:5px;padding:4px 16px;}"
            "QPushButton:hover{background:#7d3c98;}"
        )
        copy_all_btn.clicked.connect(self._copy_full_report)
        btn_row.addWidget(copy_all_btn)
        sw_layout.addLayout(btn_row)

        # Wrap in a scroll area so it works for long meetings
        scroll = QScrollArea()
        scroll.setWidget(sw)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        tabs.addTab(scroll, "📝  Summary")

        # ── Q&A tab ───────────────────────────────────────────────────
        qw = QWidget()
        ql = QVBoxLayout(qw)

        self._chat = QTextEdit()
        self._chat.setReadOnly(True)
        if not self._qa_service:
            self._chat.setPlaceholderText("Q&A requires Ollama to be running.")
        else:
            self._chat.setPlaceholderText("Ask anything about this meeting…")
        ql.addWidget(self._chat, stretch=1)

        inp_row = QHBoxLayout()
        self._qa_input = QLineEdit()
        self._qa_input.setPlaceholderText("Type your question and press Enter…")
        self._qa_input.setEnabled(bool(self._qa_service))
        self._qa_input.returnPressed.connect(self._submit_question)
        self._qa_btn = QPushButton("Ask")
        self._qa_btn.setFixedWidth(70)
        self._qa_btn.setEnabled(bool(self._qa_service))
        self._qa_btn.clicked.connect(self._submit_question)
        inp_row.addWidget(self._qa_input)
        inp_row.addWidget(self._qa_btn)
        ql.addLayout(inp_row)

        tabs.addTab(qw, "💬  Q&A")
        root.addWidget(tabs)

    # ------------------------------------------------------------------ #
    # Live refresh (polls DB until summary is ready)
    # ------------------------------------------------------------------ #

    def _poll_db(self) -> None:
        """Called every 2.5 s while summary is pending. Reads from DB and updates UI."""
        self._poll_ticks += 1
        if self._poll_ticks > 12:  # ~30 s — if no summary yet, stop polling and show the button
            self._poll_timer.stop()
            self._sum_edit.setPlainText(
                "No summary was generated for this meeting.\n"
                "Click \"\U0001f504 Generate Summary\" to create one now."
            )
            self._dec_edit.setPlainText("")
            self._task_edit.setPlainText("")
            self._gen_btn.setVisible(True)
            return
        try:
            from core.storage.database import get_db
            from core.storage.models import Session as DbSession
            db = get_db()
            try:
                rec = db.get(DbSession, self._session_id)
                if not rec:
                    return

                # Refresh transcript (new segments may have arrived)
                lines = []
                for seg in rec.segments:
                    m, s = divmod(int(seg.start_sec), 60)
                    lines.append(f"[{m:02d}:{s:02d}] {seg.text}")
                new_transcript = "\n".join(lines)
                if new_transcript != self._transcript:
                    self._transcript = new_transcript
                    # Update the transcript text edit (first tab)
                    # We stored no ref; find it via the tabs widget
                    self._t_edit.setPlainText(new_transcript)

                # Refresh summary
                if rec.summaries:
                    latest = rec.summaries[-1]
                    self._apply_summary(latest.summary_text, latest.action_items or "")
                    self._poll_timer.stop()
            finally:
                db.close()
        except Exception as exc:
            log.debug("poll_db error: %s", exc)

    def _generate_summary(self) -> None:
        """Trigger on-demand summarization for this meeting."""
        if not self._llm_client or not self._transcript.strip():
            self._sum_edit.setPlainText(
                "Cannot generate summary: no LLM provider configured or transcript is empty.\n"
                "Configure a provider in Settings \u2192 LLM."
            )
            return
        self._gen_btn.setEnabled(False)
        self._gen_btn.setText("\u23f3  Generating...")
        self._sum_edit.setPlainText("\u23f3  Generating summary\u2026 this may take 15\u201360 seconds.")
        self._dec_edit.setPlainText("\u23f3  Pending\u2026")
        self._task_edit.setPlainText("\u23f3  Pending\u2026")

        self._summary_gen_worker = _SummaryGenWorker(
            self._transcript, self._language, self._llm_client,
            self._llm_model, self._session_id, parent=self,
        )
        self._summary_gen_worker.done.connect(self._on_gen_done)
        self._summary_gen_worker.error.connect(self._on_gen_error)
        self._summary_gen_worker.start()

    @pyqtSlot(str, str)
    def _on_gen_done(self, summary: str, action_items: str) -> None:
        self._apply_summary(summary, action_items)
        self._gen_btn.setText("\U0001f504  Re-generate Summary")
        self._gen_btn.setEnabled(True)

    @pyqtSlot(str)
    def _on_gen_error(self, error: str) -> None:
        self._sum_edit.setPlainText(f"Summary generation failed:\n{error}")
        self._gen_btn.setText("\U0001f504  Retry")
        self._gen_btn.setEnabled(True)

    def _apply_summary(self, summary_text: str, action_items_raw: str) -> None:
        """Populate the three summary sections (and participants) from a stored summary."""
        participants_text = ""
        work = action_items_raw
        if _PARTICIPANTS_SEP in work:
            work, participants_text = work.split(_PARTICIPANTS_SEP, 1)
            participants_text = participants_text.strip()
        if _DECISIONS_SEP in work:
            tasks_text, decisions_text = work.split(_DECISIONS_SEP, 1)
            tasks_text = tasks_text.strip()
            decisions_text = decisions_text.strip()
        else:
            tasks_text = work.strip()
            decisions_text = ""

        self._sum_edit.setPlainText(summary_text or "No summary available yet.")
        self._dec_edit.setPlainText(decisions_text or "No decisions recorded.")
        self._task_edit.setPlainText(tasks_text or "No tasks recorded.")
        if self._participants_lbl is not None:
            if participants_text:
                self._participants_lbl.setText(f"\U0001f465  {participants_text}")
            else:
                self._participants_lbl.hide()
        self._summary_loaded = True

    # ------------------------------------------------------------------ #

    def _submit_question(self) -> None:
        question = self._qa_input.text().strip()
        if not question or not self._qa_service:
            return
        self._qa_input.clear()
        self._qa_btn.setEnabled(False)
        self._chat.append(f"<b>You:</b> {question}")
        self._chat.append("<i style='color:#888;'>Thinking…</i>")

        worker = _QAWorker(question, self._qa_service, self._transcript, self._language, self)
        worker.answer_ready.connect(self._on_answer)
        worker.error.connect(self._on_qa_error)
        self._qa_workers.append(worker)
        worker.start()

    @pyqtSlot(str)
    def _on_answer(self, answer: str) -> None:
        # Replace last "Thinking…" line with the real answer
        doc = self._chat.document()
        cursor = self._chat.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()   # remove the newline left behind
        self._chat.setTextCursor(cursor)
        self._chat.append(f"<b>NoteMe:</b> {answer}<br>")
        self._qa_btn.setEnabled(True)

    @pyqtSlot(str)
    def _on_qa_error(self, err: str) -> None:
        self._chat.append(f'<span style="color:red;">Error: {err}</span><br>')
        self._qa_btn.setEnabled(True)

    # ------------------------------------------------------------------ #
    # Copy / Export
    # ------------------------------------------------------------------ #

    def _play_audio(self) -> None:
        """Open the audio file with the system's default media player."""
        import os, subprocess
        if self._audio_path and os.path.isfile(self._audio_path):
            try:
                os.startfile(self._audio_path)  # Windows: opens default player
            except Exception:
                subprocess.Popen(["start", self._audio_path], shell=True)

    def _save_audio(self) -> None:
        """Let the user save a copy of the audio file."""
        import os, shutil
        if not self._audio_path or not os.path.isfile(self._audio_path):
            return
        ext = os.path.splitext(self._audio_path)[1]  # .ogg or .wav
        title = self._session.get("title", "meeting").replace(" ", "_").replace(":", "-")
        default_name = f"{title}{ext}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio", default_name,
            f"Audio files (*{ext});;All files (*)",
        )
        if path:
            try:
                shutil.copy2(self._audio_path, path)
            except Exception as exc:
                log.error("Failed to save audio: %s", exc)

    @staticmethod
    def _strip_timestamps(text: str) -> str:
        """Remove [MM:SS] prefixes from each line."""
        import re
        return re.sub(r"^\[\d{2}:\d{2}\]\s*", "", text, flags=re.MULTILINE)

    def _copy_transcript(self) -> None:
        QApplication.clipboard().setText(self._t_edit.toPlainText())
        self.setWindowTitle(self.windowTitle().rstrip(" ✔") + " ✔")

    def _copy_text_only(self) -> None:
        """Copy transcript without timestamps."""
        QApplication.clipboard().setText(self._strip_timestamps(self._t_edit.toPlainText()))
        self.setWindowTitle(self.windowTitle().rstrip(" ✔") + " ✔")

    def _export_transcript(self) -> None:
        title = self._session.get("title", "meeting").replace(" ", "_").replace(":", "-")
        default_name = f"{title}_transcript.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Transcript", default_name,
            "Text files (*.txt);;Markdown (*.md);;All files (*)",
        )
        if not path:
            return
        # Build full report if exporting as .md, plain transcript otherwise
        content = self._build_full_report() if path.endswith(".md") else self._t_edit.toPlainText()
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self.setWindowTitle(self.windowTitle().rstrip(" ✔") + " ✔")
        except Exception as exc:
            log.error("Export failed: %s", exc)

    def _copy_full_report(self) -> None:
        QApplication.clipboard().setText(self._build_full_report())
        self.setWindowTitle(self.windowTitle().rstrip(" ✔") + " ✔")

    def _build_full_report(self) -> str:
        """Assemble a plain-text report with all sections."""
        title = self._session.get("title", "Meeting")
        parts = [f"# {title}", ""]

        participants = self._participants_lbl.text() if self._participants_lbl else ""
        if participants:
            parts.append(f"Participants: {participants.lstrip(chr(0x1f465)).strip()}")
            parts.append("")

        parts.append("## Summary")
        parts.append(self._sum_edit.toPlainText())
        parts.append("")

        dec = self._dec_edit.toPlainText()
        if dec and dec != "No decisions recorded.":
            parts.append("## Decisions")
            parts.append(dec)
            parts.append("")

        tasks = self._task_edit.toPlainText()
        if tasks and tasks != "No tasks recorded.":
            parts.append("## Tasks & Action Items")
            parts.append(tasks)
            parts.append("")

        parts.append("## Transcript")
        parts.append(self._t_edit.toPlainText())
        return "\n".join(parts)
