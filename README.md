# NoteMe - Meeting Transcription & Summary Agent

NoteMe is a Windows desktop application that captures audio from meetings (Zoom, Teams, or any system audio) and transcribes it in real-time using OpenAI's Whisper model. After recording, it generates meeting summaries and action items using a local LLM via Ollama.

## Features

- **Live Transcription** - Real-time speech-to-text with visual audio level meters
- **Meeting Summarization** - Automatic summary and action items via local LLM (Ollama)
- **Dual Audio Capture** - System audio (WASAPI loopback) and/or microphone input
- **Zoom Auto-Record** - Detects Zoom meetings and starts recording automatically
- **Meeting History** - Searchable database of past meetings with transcripts and summaries
- **Speech Coach** - Grammar correction for transcribed text
- **Q&A** - Ask questions about meeting content
- **System Tray** - Runs in background with quick-access controls
- **GPU Acceleration** - Optional CUDA support for faster transcription

## Prerequisites

- **Windows 10/11** (64-bit)
- **Python 3.10+** (3.11 or 3.12 recommended)
- **Ollama** (for meeting summarization) - [Download Ollama](https://ollama.com/download)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/note_me.git
cd note_me
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Enable GPU Acceleration

If you have an NVIDIA GPU with CUDA support:

```bash
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 "nvidia-cudnn-cu12>=9,<10"
```

Then update `config.yaml`:

```yaml
transcription:
  device: cuda
  compute_type: float16
```

### 5. Install Ollama and Pull a Model

Ollama is required for meeting summaries, grammar correction, and Q&A features.

```bash
# After installing Ollama from https://ollama.com/download
ollama pull mistral
```

You can use any Ollama model. Popular choices:
- `mistral` - Good balance of speed and quality
- `qwen2.5-coder:7b` - Good for technical meetings
- `llama3.1` - Strong general purpose

### 6. Run the Application

```bash
python main.py
```

## Configuration

Configuration is stored in `%APPDATA%\NoteMe\config.yaml` (created on first run). You can also edit the bundled `config.yaml` for defaults.

### Audio Settings

```yaml
audio:
  chunk_duration_sec: 3.0        # Seconds of audio per transcription chunk
  default_source: both           # system | mic | both
  mic_device_index: -1           # -1 = auto-detect default microphone
  system_device_index: -1        # -1 = auto-detect default WASAPI loopback
```

### Transcription Settings

```yaml
transcription:
  model: large-v3                # tiny | base | small | medium | large-v3 | turbo
  compute_type: int8             # int8 (CPU) | float16 (GPU) | float32
  device: cpu                    # cpu | cuda
  default_language: auto         # auto | en | he | any ISO 639-1 code
```

**Model recommendations by hardware:**

| Hardware | Recommended Model | Compute Type | Notes |
|----------|------------------|--------------|-------|
| CPU only | `base` or `small` | `int8` | App auto-selects `base` for live recording on CPU |
| NVIDIA GPU (6GB+ VRAM) | `large-v3` | `float16` | Best accuracy |
| NVIDIA GPU (4GB VRAM) | `turbo` or `medium` | `float16` | Good balance |

### LLM Settings

```yaml
llm:
  base_url: http://localhost:11434   # Ollama server URL
  model: mistral                     # Ollama model name
  timeout_sec: 120                   # Request timeout
```

### App Settings

```yaml
app:
  startup_with_windows: false    # Auto-start with Windows
  storage_path: ""               # Data directory (default: ~/NoteMe)
  zoom_auto_record: false        # Auto-record when Zoom meeting detected
```

## Usage

### Recording a Meeting

1. Launch NoteMe
2. Select your audio source (System Audio, Microphone, or Both)
3. Choose the language (Auto Detect, English, or Hebrew)
4. Click **Start Recording**
5. The live transcript appears as audio is processed
6. Click **Stop Recording** when done
7. A summary and action items are generated automatically

### Viewing Past Meetings

1. Go to the **Meetings** tab
2. Click any meeting to view its transcript, summary, and action items
3. Use the **Q&A** feature to ask questions about meeting content

### System Tray

- NoteMe minimizes to the system tray when closed
- Right-click the tray icon for quick Start/Stop recording
- Double-click to open the main window

## Building a Standalone Executable

To create a portable `.exe` (no Python installation required):

```bash
pip install pyinstaller
pyinstaller noteme.spec
```

The output is in `dist\NoteMe\`. Run `NoteMe.exe` to launch.

**Note:** The first run will download the Whisper model (~150MB for `base`, ~3GB for `large-v3`). Models are cached in `~/NoteMe/models/`.

## Troubleshooting

### No audio being captured
- Make sure the correct audio device is selected in Settings
- For system audio capture, ensure your default output device supports WASAPI loopback
- Try selecting "Microphone" only to test

### Transcription is slow or not appearing
- On CPU, the app uses the `base` model for live transcription (automatic)
- The progress indicator next to the recording status shows transcription queue depth
- If the queue keeps growing, consider using a smaller model (`tiny` or `base`)
- GPU acceleration (`device: cuda`) significantly improves speed

### Ollama connection errors
- Make sure Ollama is running: `ollama serve`
- Verify the URL in Settings matches your Ollama server (default: `http://localhost:11434`)
- Pull the model first: `ollama pull mistral`

### CUDA/GPU errors
- Install CUDA pip packages: `pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 "nvidia-cudnn-cu12>=9,<10"`
- The app automatically falls back to CPU if CUDA DLLs are missing

## Project Structure

```
note_me/
├── main.py                     # Application entry point
├── config.yaml                 # Default configuration
├── requirements.txt            # Python dependencies
├── noteme.spec                 # PyInstaller build spec
├── core/
│   ├── transcription/engine.py # Faster-Whisper transcription worker
│   ├── audio/capture.py        # WASAPI + mic audio capture
│   ├── audio/devices.py        # Audio device enumeration
│   ├── session.py              # Meeting session orchestrator
│   ├── zoom_watcher.py         # Zoom meeting auto-detection
│   ├── storage/                # SQLite database (SQLAlchemy)
│   └── llm/                    # Ollama client, summarizer, grammar, Q&A
├── ui/
│   ├── main_window.py          # Main window with tabs
│   ├── tray.py                 # System tray integration
│   └── widgets/                # Recording, meetings, settings, speech coach
└── utils/
    ├── config.py               # YAML config with Pydantic validation
    ├── cuda_setup.py           # CUDA DLL bootstrap
    └── logger.py               # Rotating file logger
```

## Tech Stack

- **Speech-to-Text:** [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2-optimized Whisper)
- **Audio Capture:** PyAudioWPatch (WASAPI loopback) + sounddevice (microphone)
- **GUI:** PyQt6
- **LLM:** [Ollama](https://ollama.com/) (local inference)
- **Database:** SQLite with SQLAlchemy ORM
- **Config:** YAML with Pydantic validation

## License

This project is provided as-is for personal and educational use.
