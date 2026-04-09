from __future__ import annotations

import datetime
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QMetaObject, QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot

from .audio.capture import AudioCaptureManager
from .llm.client import OllamaClient
from .llm.summarizer import Summarizer
from .storage.database import get_db
from .storage.models import Session as DbSession, Summary, TranscriptSegment
from .transcription.engine import Segment, TranscriptionWorker

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Background worker — LLM summarization
# ------------------------------------------------------------------ #

class SummaryWorker(QThread):
    # (summary_text, action_items, llm_title)
    summary_ready = pyqtSignal(str, str, str)
    error = pyqtSignal(str)

    def __init__(self, transcript: str, summarizer: Summarizer, language: str, parent=None):
        super().__init__(parent)
        self._transcript = transcript
        self._summarizer = summarizer
        self._language = language

    def run(self) -> None:
        try:
            result = self._summarizer.summarize(self._transcript, self._language)
            self.summary_ready.emit(result.summary, result.action_items, result.title)
        except Exception as exc:
            self.error.emit(str(exc))


# ------------------------------------------------------------------ #
# Meeting session orchestrator
# ------------------------------------------------------------------ #

class MeetingSession(QObject):
    """
    Lifecycle: start() → [live segment_ready signals] → stop()
                      → recording_stopped(session_id)
                      → [async] summary_ready(summary, action_items)
    """

    segment_ready = pyqtSignal(object)       # Segment dataclass
    summary_ready = pyqtSignal(str, str)     # summary, action_items
    recording_stopped = pyqtSignal(int)      # session_id
    levels_updated = pyqtSignal(float, float)  # mic_rms, sys_rms (0-1)
    transcription_progress = pyqtSignal(int, int)  # (chunks_done, chunks_remaining)
    error = pyqtSignal(str)

    def __init__(
        self,
        source: str = "both",
        language: str = "auto",
        system_device_index: int = -1,
        mic_device_index: int = -1,
        chunk_duration_sec: float = 3.0,
        whisper_model: str = "large-v3",
        whisper_compute_type: str = "int8",
        whisper_device: str = "cpu",
        preloaded_whisper_model=None,
        ollama_client: Optional[OllamaClient] = None,
        ollama_model: str = "mistral",
        storage_path: Optional[str] = None,
        vocabulary_hint: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._language = language
        self._ollama_client = ollama_client
        self._ollama_model = ollama_model
        self._db_session_id: Optional[int] = None
        self._segments: list[Segment] = []
        self._is_recording = False
        self._storage_path = storage_path
        self._wav_path: Optional[str] = None

        # Audio capture (save path set in start() after session ID is known)
        self._capture_kwargs = dict(
            on_audio_ready=self._on_audio_ready,
            source=source,
            system_device_index=system_device_index,
            mic_device_index=mic_device_index,
            chunk_duration_sec=chunk_duration_sec,
        )
        self._capture: Optional[AudioCaptureManager] = None

        # Transcription (reuses the app-level preloaded model)
        whisper_lang = language if language and language != "auto" else None
        self._worker = TranscriptionWorker(
            model_name=whisper_model,
            compute_type=whisper_compute_type,
            device=whisper_device,
            language=whisper_lang,
            preloaded_model=preloaded_whisper_model,
            vocabulary_hint=vocabulary_hint,
        )
        self._worker.segment_ready.connect(self._on_segment)
        self._worker.chunk_done.connect(self.transcription_progress)
        self._worker.error.connect(self.error)

        self._summary_worker: Optional[SummaryWorker] = None
        self._stop_thread: Optional[threading.Thread] = None

        # QTimer polls capture levels from the Qt thread so we don't emit
        # signals from the raw Python mixer thread.
        self._level_timer = QTimer(self)
        self._level_timer.setInterval(80)   # ~12 Hz
        self._level_timer.timeout.connect(self._poll_levels)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._is_recording:
            return
        self._is_recording = True
        self._segments.clear()

        # Persist session record
        db = get_db()
        try:
            rec = DbSession(
                title=f"Meeting {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                start_time=datetime.datetime.utcnow(),
                language=self._language,
                mode="meeting",
            )
            db.add(rec)
            db.commit()
            self._db_session_id = rec.id
        finally:
            db.close()

        # Create the audio capture with a WAV save path
        audio_dir = self._storage_path or str(Path.home() / "NoteMe")
        Path(audio_dir).mkdir(parents=True, exist_ok=True)
        self._wav_path = str(Path(audio_dir) / f"session_{self._db_session_id}.wav")
        self._capture = AudioCaptureManager(
            **self._capture_kwargs,
            save_audio_path=self._wav_path,
        )

        self._worker.reset_time()
        self._worker.start()
        self._capture.start()
        self._level_timer.start()
        log.info("MeetingSession %d started.", self._db_session_id)

    def stop(self) -> None:
        if not self._is_recording:
            return
        self._is_recording = False
        self._stop_thread = threading.Thread(target=self._finish_stop, daemon=True, name="MeetingSessionStop")
        self._stop_thread.start()

    def _finish_stop(self) -> None:
        # NOTE: this runs in a plain Python thread (not Qt main thread).
        # Never call QTimer or QObject methods directly from here.
        self._capture.stop()
        self._worker.stop_processing()
        self._worker.wait(15_000)  # wait up to 15 s for remaining transcription

        # Compress WAV → OGG (much smaller) in the background
        ogg_path = self._compress_audio()

        if self._db_session_id:
            db = get_db()
            try:
                rec = db.get(DbSession, self._db_session_id)
                if rec:
                    rec.end_time = datetime.datetime.utcnow()
                    rec.audio_path = ogg_path or self._wav_path
                    db.commit()
            finally:
                db.close()

        log.info("MeetingSession %d stopped.", self._db_session_id)

        # Emit recording_stopped so the UI can react.
        # All segment_ready signals from the TranscriptionWorker were queued
        # to the Qt event loop *before* this emit, so they will be delivered first.
        self.recording_stopped.emit(self._db_session_id or 0)

        # Schedule _on_recording_done on the Qt main thread.
        # By the time it runs, the Qt event loop will have already processed
        # every pending segment_ready event, so self._segments is fully populated.
        QMetaObject.invokeMethod(
            self, "_on_recording_done", Qt.ConnectionType.QueuedConnection
        )

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #

    @pyqtSlot()
    def _on_recording_done(self) -> None:
        """Runs on the Qt main thread after all segment events have been processed."""
        self._level_timer.stop()
        if not (self._ollama_client and self._segments):
            log.info("No segments or Ollama client — skipping summarization.")
            # Emit empty summary so MainWindow can release the session
            self.summary_ready.emit("", "")
            return
        log.info("Starting summarization for session %d (%d segments).",
                 self._db_session_id, len(self._segments))
        transcript = self._build_transcript()
        summarizer = Summarizer(self._ollama_client, self._ollama_model)
        self._summary_worker = SummaryWorker(transcript, summarizer, self._language, self)
        self._summary_worker.summary_ready.connect(self._on_summary_ready)
        self._summary_worker.error.connect(self._on_summary_error)
        self._summary_worker.start()

    def _poll_levels(self) -> None:
        """Emit current audio levels from the Qt thread (safe for UI connections)."""
        self.levels_updated.emit(self._capture.mic_rms, self._capture.sys_rms)

    def _on_audio_ready(self, audio: np.ndarray) -> None:
        if self._worker.isRunning():
            self._worker.enqueue(audio)

    @pyqtSlot(object)
    def _on_segment(self, segment: Segment) -> None:
        self._segments.append(segment)
        if self._db_session_id:
            db = get_db()
            try:
                db.add(TranscriptSegment(
                    session_id=self._db_session_id,
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                    text=segment.text,
                    language=segment.language,
                    confidence=segment.confidence,
                ))
                db.commit()
            except Exception as exc:
                log.error("Failed to save segment: %s", exc)
            finally:
                db.close()
        self.segment_ready.emit(segment)

    def _on_summary_error(self, error: str) -> None:
        log.error("Summarization error: %s", error)
        # Still emit so MainWindow releases the session
        self.summary_ready.emit("", "")

    @pyqtSlot(str, str, str)
    def _on_summary_ready(self, summary: str, action_items: str, llm_title: str) -> None:
        if self._db_session_id:
            db = get_db()
            try:
                db.add(Summary(
                    session_id=self._db_session_id,
                    summary_text=summary,
                    action_items=action_items,
                    ollama_model=self._ollama_model,
                ))
                # Update the session title if the LLM produced a non-empty one
                if llm_title:
                    rec = db.get(DbSession, self._db_session_id)
                    if rec:
                        date_str = rec.start_time.strftime("%Y-%m-%d %H:%M") if rec.start_time else ""
                        rec.title = f"{llm_title}  ({date_str})" if date_str else llm_title
                db.commit()
            except Exception as exc:
                log.error("Failed to save summary: %s", exc)
            finally:
                db.close()
        self.summary_ready.emit(summary, action_items)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _compress_audio(self) -> Optional[str]:
        """Compress WAV to OGG/Vorbis in chunks to avoid memory issues. Returns OGG path or None."""
        if not self._wav_path or not os.path.isfile(self._wav_path):
            return None
        ogg_path = self._wav_path.replace(".wav", ".ogg")
        try:
            import soundfile as sf
            CHUNK = 160_000  # 10 seconds at 16 kHz
            with sf.SoundFile(self._wav_path, "r") as src:
                with sf.SoundFile(ogg_path, "w", samplerate=src.samplerate,
                                  channels=src.channels, format="OGG", subtype="VORBIS") as dst:
                    while True:
                        buf = src.read(CHUNK, dtype="float32")
                        if len(buf) == 0:
                            break
                        dst.write(buf)
            os.remove(self._wav_path)  # delete the larger WAV
            log.info("Compressed audio: %s (%.1f MB)",
                     ogg_path, os.path.getsize(ogg_path) / 1e6)
            return ogg_path
        except ImportError:
            log.info("soundfile not available — keeping WAV")
            return None
        except Exception as exc:
            log.warning("Audio compression failed: %s — keeping WAV", exc)
            return None

    def _build_transcript(self) -> str:
        lines = []
        for seg in self._segments:
            m, s = divmod(int(seg.start_sec), 60)
            lines.append(f"[{m:02d}:{s:02d}] {seg.text}")
        return "\n".join(lines)
