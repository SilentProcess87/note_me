from __future__ import annotations

import numpy as np

TARGET_SR = 16_000  # Whisper expects 16 kHz mono float32


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample a mono float32 array from orig_sr to 16 kHz."""
    if orig_sr == TARGET_SR:
        return audio.astype(np.float32)
    n_target = int(len(audio) * TARGET_SR / orig_sr)
    if n_target == 0:
        return np.array([], dtype=np.float32)
    indices = np.linspace(0, len(audio) - 1, n_target)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
