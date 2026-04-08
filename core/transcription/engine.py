from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

TARGET_SR = 16_000


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    text: str
    language: str
    confidence: float


class TranscriptionWorker(QThread):
    """
    QThread that consumes 16 kHz float32 audio buffers and emits Segment objects.

    Accepts an already-loaded WhisperModel (``preloaded_model``) so the heavy
    model load only happens once in MainWindow.  If not provided it loads at
    run() time.
    """

    segment_ready = pyqtSignal(object)   # Segment dataclass
    error = pyqtSignal(str)

    # Signals for progress tracking
    chunk_queued = pyqtSignal(int)      # current queue depth
    chunk_processing = pyqtSignal(int)  # chunks remaining (including current)
    chunk_done = pyqtSignal(int, int)   # (chunks_processed_total, chunks_remaining)

    def __init__(
        self,
        model_name: str = "large-v3",
        compute_type: str = "int8",
        device: str = "cpu",
        language: Optional[str] = None,   # None or "auto" → auto-detect
        preloaded_model=None,
        vocabulary_hint: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._model_name = model_name
        self._compute_type = compute_type
        self._device = device
        self._language = language if language and language != "auto" else None
        self._model = preloaded_model
        self._vocab_hint = vocabulary_hint
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._time_offset: float = 0.0
        self._chunks_processed: int = 0
        self._prev_text: str = ""  # last segment text, fed as initial_prompt for continuity

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def enqueue(self, audio: np.ndarray) -> None:
        """Add a 16 kHz float32 mono buffer to the processing queue."""
        self._queue.put(audio)
        self.chunk_queued.emit(self._queue.qsize())

    def stop_processing(self) -> None:
        """Signal the thread to finish after draining remaining items."""
        self._queue.put(None)  # sentinel

    def reset_time(self) -> None:
        self._time_offset = 0.0
        self._prev_text = ""

    # ------------------------------------------------------------------ #
    # QThread.run
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        if self._model is None:
            if not self._load_model():
                return

        self._chunks_processed = 0
        while True:
            audio = self._queue.get()
            if audio is None:
                break
            remaining = self._queue.qsize() + 1  # +1 for current chunk
            self.chunk_processing.emit(remaining)
            self._transcribe(audio)
            self._chunks_processed += 1
            self.chunk_done.emit(self._chunks_processed, self._queue.qsize())

    def _load_model(self) -> bool:
        try:
            from faster_whisper import WhisperModel

            log.info(
                "Loading Whisper model '%s' (%s / %s)…",
                self._model_name,
                self._device,
                self._compute_type,
            )
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
            log.info("Whisper model ready.")
            return True
        except Exception as exc:
            msg = f"Could not load Whisper model '{self._model_name}': {exc}"
            log.error(msg)
            self.error.emit(msg)
            return False

    def _transcribe(self, audio: np.ndarray) -> None:
        if self._model is None:
            return
        chunk_duration = len(audio) / TARGET_SR
        # Use greedy decoding on CPU for speed; beam search on GPU for quality
        beam = 1 if self._device == "cpu" else 5
        try:
            # Build the initial prompt:
            # 1. Vocabulary hint — biases Whisper to recognize tech terms
            #    even when spoken mid-Hebrew-sentence
            # 2. Previous chunk tail — provides continuity across chunk boundaries
            parts = []
            if self._vocab_hint:
                parts.append(self._vocab_hint)
            if self._prev_text:
                parts.append(self._prev_text[-200:])
            prompt = ". ".join(parts) if parts else None

            segments, info = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=beam,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.3,
                    "min_silence_duration_ms": 200,
                    "min_speech_duration_ms": 100,
                    "speech_pad_ms": 200,
                },
                word_timestamps=False,
                condition_on_previous_text=True,
                initial_prompt=prompt,
            )
            chunk_texts = []
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                chunk_texts.append(text)
                s = Segment(
                    start_sec=self._time_offset + seg.start,
                    end_sec=self._time_offset + seg.end,
                    text=text,
                    language=getattr(info, "language", self._language or "und"),
                    confidence=getattr(seg, "avg_logprob", 0.0),
                )
                self.segment_ready.emit(s)
            if chunk_texts:
                self._prev_text = " ".join(chunk_texts)
        except Exception as exc:
            log.error("Transcription error: %s", exc)
        finally:
            self._time_offset += chunk_duration
