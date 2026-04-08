"""
Bootstrap CUDA runtime DLL paths for Windows.

On Windows, Python 3.8+ uses a restricted DLL search that ignores the PATH
environment variable for extension modules.  ctranslate2 (used by
faster-whisper) needs cuBLAS 12 and cuDNN 9 DLLs registered via
os.add_dll_directory() *before* the library is imported.

Supported installation methods (checked in order):
  1. pip packages: nvidia-cublas-cu12, nvidia-cudnn-cu12
     (install with:  pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
                     pip install nvidia-cuda-runtime-cu12)
  2. Official CUDA Toolkit installer
     (C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin)
  3. Official cuDNN installer
     (C:\\Program Files\\NVIDIA\\CUDNN\\v9.x\\bin\\...)
  4. CUDA_PATH / CUDA_PATH_V12_x environment variables (set by CUDA installer)

Call bootstrap() at app startup before any other CUDA-dependent import.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _register(path: Path, out: List[Path]) -> None:
    """Call os.add_dll_directory on an existing directory and record it."""
    if not path.is_dir():
        return
    try:
        os.add_dll_directory(str(path))
        out.append(path)
        log.debug("Registered CUDA DLL dir: %s", path)
    except OSError as exc:
        log.debug("Could not register %s: %s", path, exc)


def _pip_nvidia_dirs() -> List[Path]:
    """
    Enumerate bin/ directories from pip-installed nvidia-* packages.
    These live under <site-packages>/nvidia/<pkg>/bin/*.dll
    """
    found: List[Path] = []
    for sp in sys.path:
        nvidia = Path(sp) / "nvidia"
        if not nvidia.is_dir():
            continue
        for pkg in nvidia.iterdir():
            if not pkg.is_dir():
                continue
            for sub in ("bin", "lib"):
                d = pkg / sub
                if d.is_dir() and any(d.glob("*.dll")):
                    found.append(d)
    return found


def _system_cuda_dirs() -> List[Path]:
    """
    Return bin directories from a system NVIDIA CUDA Toolkit / cuDNN install.
    """
    found: List[Path] = []

    # --- CUDA Toolkit (cuBLAS, cudart, etc.) ---
    # Primary: highest versioned dir under the toolkit root
    toolkit = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if toolkit.is_dir():
        for ver in sorted(toolkit.iterdir(), reverse=True):
            b = ver / "bin"
            if b.is_dir() and any(b.glob("cublas64_*.dll")):
                found.append(b)
                break  # use highest compatible version only

    # Secondary: CUDA_PATH / CUDA_PATH_V12_x environment variables
    for env_key in sorted(os.environ, reverse=True):
        if not (env_key == "CUDA_PATH" or env_key.startswith("CUDA_PATH_V12")):
            continue
        b = Path(os.environ[env_key]) / "bin"
        if b.is_dir() and b not in found:
            found.append(b)

    # --- cuDNN (separate NVIDIA installer) ---
    # cuDNN 9.x places DLLs under  <root>/v9.x/bin/<cuda_version>/cudnn*.dll
    cudnn_root = Path("C:/Program Files/NVIDIA/CUDNN")
    if cudnn_root.is_dir():
        for ver in sorted(cudnn_root.iterdir(), reverse=True):
            top_bin = ver / "bin"
            if not top_bin.is_dir():
                continue
            # Prefer CUDA-version-specific subdirs (cuDNN 9 layout)
            for sub in sorted(top_bin.iterdir(), reverse=True):
                if sub.is_dir() and any(sub.glob("cudnn*.dll")):
                    found.append(sub)
                    break
            # Also include the top-level bin if DLLs are there
            if any(top_bin.glob("cudnn*.dll")):
                found.append(top_bin)
            break  # highest version only

    return found


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def bootstrap() -> bool:
    """
    Register CUDA runtime DLL directories with the Windows DLL loader.

    Must be called before importing faster_whisper or ctranslate2.

    Returns True if at least one CUDA-related directory was registered
    (or if we are not on Windows where this step is unnecessary).
    """
    if sys.platform != "win32":
        return True  # Nothing needed on Linux / macOS
    if not callable(getattr(os, "add_dll_directory", None)):
        return True  # Python < 3.8 — PATH is used directly

    registered: List[Path] = []

    # When running as a PyInstaller bundle the CUDA DLLs are bundled alongside
    # the exe in _MEIPASS.  Register that directory and we're done.
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        _register(meipass, registered)
        if registered:
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = str(meipass) + (
                os.pathsep + current_path if current_path else ""
            )
            log.info("CUDA DLL bootstrap (frozen): registered %s", meipass)
        return True  # always succeed in frozen mode

    # 1. pip-installed nvidia packages (preferred — no system install required)
    for d in _pip_nvidia_dirs():
        _register(d, registered)

    # 2. System CUDA Toolkit / cuDNN installation
    if not registered:
        for d in _system_cuda_dirs():
            _register(d, registered)

    if registered:
        # os.add_dll_directory covers Python extension-module loading.
        # Also prepend to PATH so that native C++ code inside ctranslate2
        # that calls LoadLibrary() directly (e.g. for cuBLAS/cuDNN kernels
        # loaded lazily at first inference) can find the DLLs too.
        path_str = os.pathsep.join(str(p) for p in registered)
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = path_str + (os.pathsep + current_path if current_path else "")
        log.info(
            "CUDA DLL bootstrap: registered %d path(s) (AddDllDirectory + PATH): %s",
            len(registered),
            [str(p) for p in registered],
        )
        return True

    log.warning(
        "CUDA DLL bootstrap: no CUDA runtime DLLs found. "
        "GPU transcription will fail with 'cublas64_12.dll not found'.\n"
        "To fix without installing the full CUDA Toolkit, run:\n"
        "  pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 "
        "\"nvidia-cudnn-cu12>=9,<10\""
    )
    return False
