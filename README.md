# NoteMe — AI Meeting Transcription & Summary Agent

NoteMe is a Windows desktop application that listens to your meetings, transcribes them in real-time, and generates intelligent summaries with decisions, tasks, and participant identification — all powered by local AI.

## Features

### Core
- **Real-time transcription** — Live speech-to-text using faster-whisper on GPU or CPU
- **Dual Whisper models** — Generic `turbo` for English + specialized `ivrit-ai` model for Hebrew
- **Vocabulary hints** — Teach Whisper to recognize technical terms (Kubernetes, pods, etc.) in mixed-language conversations
- **Audio level meters** — Visual feedback for both microphone and system audio during recording
- **Audio recording** — Every meeting is saved as a compressed OGG file for later playback

### Meeting Intelligence (LLM-powered)
- **Smart summaries** — Title, participants, summary, decisions, and tasks extracted automatically
- **Live Q&A** — Ask questions about the meeting *while it's still in progress*
- **Post-meeting Q&A** — Query any past meeting's transcript
- **Multi-provider LLM** — Choose between Ollama (local/free), Groq (free tier), OpenAI, Together AI, Mistral AI, or any OpenAI-compatible API

### Integrations
- **Zoom auto-detect** — Automatically starts recording when a Zoom meeting begins, stops when it ends
- **System tray** — Runs in background with quick start/stop controls
- **Windows startup** — Optional auto-start with Windows

### Export
- **Copy transcript** — With or without timestamps
- **Copy full report** — Summary + decisions + tasks + transcript as formatted markdown
- **Export to file** — Save as `.txt` (transcript) or `.md` (full report)
- **Play/Save audio** — Listen to or download the meeting recording

### Speech Coach
- Record yourself speaking, get grammar-corrected version via LLM
- Side-by-side original vs improved text with one-click copy

### Security
- **DPAPI-encrypted API keys** — Cloud provider API keys are encrypted with Windows credentials; cannot be extracted even from a decompiled binary
- **Audio stays local** — Whisper transcription always runs on your machine; only text is sent to cloud LLMs (if configured)

## Architecture

```
Mic / System Audio
    | (WASAPI loopback + sounddevice)
AudioCaptureManager --> 16kHz mono chunks --> WAV file
    |
TranscriptionWorker (QThread)
    | faster-whisper + Silero VAD
Live transcript segments --> SQLite DB
    | (on recording stop)
LLM Summarizer --> Title, Participants, Summary, Decisions, Tasks
    |
Meeting Detail UI <-- DB polling (auto-refresh)
```

**Two independent AI models:**

| Component | What it does | Runs where | Cost |
|-----------|-------------|-----------|------|
| **Whisper** (speech-to-text) | Converts audio to text transcript | Always local (GPU/CPU) | Free |
| **LLM** (text to summary/Q&A) | Reads transcript, generates summary and Q&A answers | Local (Ollama) or Cloud API | Free to ~$0.01/meeting |

## Prerequisites

- **Windows 10/11** (64-bit)
- **Python 3.11+**
- **NVIDIA GPU** (recommended, 6+ GB VRAM) — works on CPU too, just slower

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Enable GPU acceleration (recommended)

```bash
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 "nvidia-cudnn-cu12>=9,<10"
```

### 3. Install an LLM provider

**Option A — Local (Ollama, free, fully offline):**
```bash
# Install from https://ollama.com/download
ollama pull llama3.1:8b
```

**Option B — Cloud (Groq, free tier, faster):**
- Get a free API key at [console.groq.com](https://console.groq.com)
- Enter it in NoteMe Settings, LLM, Provider: Groq

### 4. Run

```bash
python main.py
```

On first launch, Whisper models are downloaded automatically (~1.6 GB each for turbo and Hebrew).

## Configuration

Stored in `%APPDATA%\NoteMe\config.yaml`. Editable in the Settings tab.

### Key settings

```yaml
transcription:
  model: turbo                    # tiny | base | small | medium | large-v3 | turbo
  device: cuda                    # cuda | cpu
  compute_type: float16           # float16 (GPU) | int8 (CPU)
  default_language: auto          # auto | en | he
  vocabulary_hint: "Kubernetes, pods, Docker, CI/CD, ..."

audio:
  default_source: both            # system | mic | both
  chunk_duration_sec: 8.0         # seconds per transcription chunk

llm:
  provider: ollama                # ollama | groq | openai | together | mistral | custom
  model: llama3.1:8b
  api_key_encrypted: ""           # DPAPI-encrypted; set via Settings UI
```

### Model recommendations

| Your hardware | Whisper model | Compute | Notes |
|--------------|---------------|---------|-------|
| NVIDIA GPU 8+ GB | `turbo` | `float16` | Best speed/accuracy balance |
| NVIDIA GPU 4-6 GB | `small` or `base` | `float16` | Lower VRAM usage |
| CPU only | `base` | `int8` | Slower but works |
| Hebrew meetings | Set language to Hebrew | — | Loads dedicated ivrit-ai model |
| Mixed Hebrew/English | Set to Auto + add vocab hints | — | Uses turbo with vocabulary bias |

## Building a Standalone Executable

```bash
pip install pyinstaller
pyinstaller noteme.spec
```

Output: `dist\NoteMe\NoteMe.exe` (~3.7 GB, includes CUDA DLLs and all dependencies).

The exe bundles:
- Python runtime
- CUDA DLLs (cublas, cudnn) — GPU works without any pip packages on target machine
- All Python dependencies

Whisper models are downloaded on first run to `%USERPROFILE%\NoteMe\models\`.

## Project Structure

```
note_me/
├── main.py                      # Entry point + tqdm monkey-patch
├── config.yaml                  # Default settings
├── noteme.spec                  # PyInstaller build spec
├── core/
│   ├── audio/
│   │   ├── capture.py           # WASAPI loopback + mic + WAV recording
│   │   ├── devices.py           # Audio device enumeration
│   │   └── resample.py          # 16kHz resampling
│   ├── transcription/
│   │   └── engine.py            # TranscriptionWorker (QThread + faster-whisper)
│   ├── llm/
│   │   ├── client.py            # OllamaClient
│   │   ├── openai_client.py     # OpenAI-compatible client (Groq, OpenAI, etc.)
│   │   ├── provider.py          # Provider factory
│   │   ├── summarizer.py        # Meeting summarization prompt
│   │   ├── qa.py                # Q&A over transcript
│   │   └── grammar.py           # Speech coach grammar analysis
│   ├── storage/
│   │   ├── database.py          # SQLite + SQLAlchemy
│   │   └── models.py            # Session, TranscriptSegment, Summary, QAEntry
│   ├── session.py               # MeetingSession orchestrator
│   └── zoom_watcher.py          # Zoom meeting detection (Windows API)
├── ui/
│   ├── main_window.py           # MainWindow + _ModelLoader
│   ├── tray.py                  # System tray icon
│   └── widgets/
│       ├── recording_widget.py  # Live recording + Q&A panel
│       ├── meetings_widget.py   # Meeting history list
│       ├── meeting_detail.py    # Transcript + Summary + Q&A dialog
│       ├── speech_coach.py      # Grammar coaching
│       └── settings_widget.py   # All settings + provider config
└── utils/
    ├── config.py                # Pydantic config with YAML persistence
    ├── cuda_setup.py            # CUDA DLL bootstrap for Windows
    ├── secrets.py               # DPAPI encryption for API keys
    └── logger.py                # Rotating file logger
```

## Technical Notes

### tqdm Monkey-Patch (`main.py`)

**Problem:** `faster_whisper` and `huggingface_hub` use `tqdm` internally for progress bars. When these run inside a `QThread` (as NoteMe's model loader and transcription worker do), `tqdm`'s internal threading lock (`_lock`) becomes `None`, causing:

```
'NoneType' object does not support the context manager protocol
AttributeError: 'tqdm' object has no attribute 'disable'
```

**Root cause:** `tqdm` initializes its `_lock` using `threading.Lock()` at class definition time. In CPython, when a `QThread` creates `tqdm` instances, the lock initialization can race with Qt's thread management, leaving `_lock = None`.

**Fix:** At the very top of `main.py` (before any library import), we replace `tqdm.tqdm`, `tqdm.std.tqdm`, and `tqdm.auto.tqdm` with `_NoOpTqdm` — a minimal class that implements `tqdm`'s interface but does nothing. Since NoteMe is a GUI app, progress bars are never visible anyway.

```python
class _NoOpTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self._it = iterable
    def __iter__(self):
        return iter(self._it) if self._it is not None else iter([])
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, n=1): pass
    def close(self): pass
```

Environment variables `HF_HUB_DISABLE_PROGRESS_BARS=1` and `TQDM_DISABLE=1` are also set as a belt-and-suspenders measure, but the monkey-patch is the only reliable fix since `tqdm` may be imported before the env vars take effect.

### CUDA DLL Bootstrap (`utils/cuda_setup.py`)

On Windows, Python 3.8+ restricts DLL loading to directories registered via `os.add_dll_directory()`. Additionally, `ctranslate2` (used by faster-whisper) lazy-loads CUDA kernels using native `LoadLibrary()` which only checks `PATH`.

The bootstrap:
1. Scans pip-installed `nvidia-*` packages for DLL directories
2. Falls back to system CUDA Toolkit / cuDNN install paths
3. Registers found directories with both `os.add_dll_directory()` AND `os.environ["PATH"]`
4. In PyInstaller frozen mode, registers `sys._MEIPASS` where bundled DLLs live

### Smart Audio Mixing (`core/audio/capture.py`)

When using "Both" (system + mic), the mixer detects silence:
- System silent (no meeting audio playing) — passes mic through at full volume
- Mic silent — passes system through at full volume
- Both active — mixes at 0.7x (gentler than 0.5x, preserves speech energy for VAD)

This prevents the common problem where mixing a silent source with an active one halves the signal below the VAD threshold.

### Thread-Safe Summarization (`core/session.py`)

`MeetingSession.stop()` runs in a plain Python thread, but transcript segments are delivered via Qt signals queued to the main thread. Using `QMetaObject.invokeMethod(..., QueuedConnection)` ensures summarization starts only after all segment events have been processed — fixing a race condition where `self._segments` appeared empty.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No audio captured | Check mic device in Settings; ensure "Both" source is selected |
| VAD removes all audio | Your mic may not be the default device; select it explicitly in Settings |
| Hebrew transcription poor | Set language to Hebrew (loads specialized model); or use Auto + vocabulary hints |
| English words in Hebrew wrong | Add terms to Vocabulary Hint in Settings (e.g. "Kubernetes, pods, Docker") |
| Summary empty / not generating | Ensure Ollama is running (`ollama serve`) or configure a cloud LLM provider |
| CUDA errors | Run: `pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 "nvidia-cudnn-cu12>=9,<10"` |
| App won't start (frozen exe) | Check `%USERPROFILE%\NoteMe\noteme.log` for errors |

## License

This project uses PyQt6 (GPL v3). For commercial distribution, either open-source under GPL or replace with PySide6 (LGPL) — the API is nearly identical.

Whisper models: MIT License (OpenAI).
LLM models: Varies by provider (Llama: Meta Community License; GPT: OpenAI Terms).
