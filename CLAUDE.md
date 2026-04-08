# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NoteMe is a Windows-only PyQt6 desktop app that captures meeting audio (WASAPI loopback + microphone), transcribes it live with Faster-Whisper, and generates summaries via a local Ollama LLM. It runs in the system tray and supports Zoom auto-detection.

## Commands

```bash
# Run the application
python main.py

# Install dependencies
pip install -r requirements.txt

# Optional GPU support
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 "nvidia-cudnn-cu12>=9,<10"

# Build standalone exe (output: dist/NoteMe/NoteMe.exe)
pyinstaller noteme.spec
```

There are no tests, linting, or CI configured.

## Architecture

### Threading Model (critical to understand)

The app uses multiple threads with Qt signal-based communication. Never call QObject methods from non-Qt threads directly.

- **Device threads** (PyAudio/sounddevice callbacks) -> push raw PCM into thread-safe queues
- **Mixer thread** (`AudioCaptureManager._mixer_loop`) -> drains queues every 100ms, resamples to 16kHz, accumulates chunks, fires `on_audio_ready` callback
- **TranscriptionWorker** (QThread) -> consumes audio chunks from its queue, runs Whisper inference, emits `segment_ready` signals to Qt event loop
- **SummaryWorker** (QThread) -> runs after recording stops, calls Ollama REST API
- **Qt main thread** -> all UI updates, DB writes (via signal slots)
- **Level polling** -> QTimer at ~12Hz reads RMS values from mixer thread (safe: atomic float reads)

### Recording Pipeline Flow

```
Audio devices -> queues -> Mixer thread (resample+accumulate) -> 3-sec chunks
  -> TranscriptionWorker.enqueue() -> Whisper inference -> segment_ready signal
  -> MeetingSession._on_segment() -> DB insert + UI append
```

On stop: capture stops -> sentinel queued to transcriber -> wait up to 15s for drain -> `recording_stopped` signal -> `_on_recording_done` scheduled on Qt thread -> SummaryWorker launched.

### Model Loading and Fallback Chain

`_ModelLoader` (QThread in `ui/main_window.py`) loads the Whisper model at startup with a fallback chain. On CPU with `large-v3` configured, it loads `base` first (since `_start_recording` will use `base` for CPU live transcription). If CUDA DLLs are missing, it auto-falls back to CPU/int8. The preloaded model is reused across recording sessions to avoid reload.

### Key Architectural Decisions

- **Config precedence**: `%APPDATA%/NoteMe/config.yaml` overrides bundled `config.yaml`. Settings UI writes to APPDATA. Pydantic models validate config.
- **Database**: SQLite at `~/NoteMe/noteme.db` via SQLAlchemy ORM. Sessions have segments, summaries, and QA entries.
- **Whisper models**: Downloaded via `huggingface_hub.snapshot_download` to `~/NoteMe/models/`. Cached locally.
- **CUDA bootstrap**: `utils/cuda_setup.py` must run before any `faster_whisper` import to add nvidia DLL paths to `os.environ["PATH"]`.
- **PyInstaller**: `noteme.spec` bundles all native DLLs (ctranslate2, CUDA, PortAudio). UPX is disabled (breaks native DLLs). Console is disabled (windowed mode), so stdout/stderr are redirected to devnull.
- **CPU optimization**: `beam_size=1` (greedy) on CPU, `beam_size=5` on GPU. CPU forces lighter model for live transcription.
- **Close behavior**: Window close hides to tray (`closeEvent` ignores the event). `app.setQuitOnLastWindowClosed(False)`.

### Signal Wiring

`MainWindow` is the central hub connecting `RecordingWidget` signals to `MeetingSession` slots and vice versa. The `SystemTray` syncs recording state via `recording_state_changed` signal. Zoom auto-record connects `ZoomWatcher` signals to `_start_recording`/`_stop_recording`.
