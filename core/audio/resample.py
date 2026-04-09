from __future__ import annotations

import numpy as np
from math import gcd

TARGET_SR = 16_000  # Whisper expects 16 kHz mono float32

# Cache for pre-computed low-pass filter kernels per sample rate
_LP_CACHE: dict[int, np.ndarray] = {}


def _lowpass_kernel(factor: int) -> np.ndarray:
    """Simple windowed-sinc low-pass filter for anti-aliasing before decimation."""
    if factor in _LP_CACHE:
        return _LP_CACHE[factor]
    # Filter length: 2 * factor + 1 (small and fast)
    N = 2 * factor + 1
    n = np.arange(N) - factor
    # Normalized cutoff at 1/factor of Nyquist
    fc = 1.0 / factor
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(n == 0, fc, np.sin(np.pi * fc * n) / (np.pi * n))
    # Hann window
    h *= 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
    h /= h.sum()  # normalize
    kernel = h.astype(np.float32)
    _LP_CACHE[factor] = kernel
    return kernel


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample a mono float32 array from orig_sr to 16 kHz.

    For integer-ratio downsampling (e.g. 48kHz, 44.1kHz, 192kHz → 16kHz)
    uses fast decimation with a lightweight anti-alias filter instead of
    the expensive np.interp approach.
    """
    if orig_sr == TARGET_SR:
        return audio.astype(np.float32)
    if len(audio) == 0:
        return np.array([], dtype=np.float32)

    # Check if we can use integer-ratio decimation
    g = gcd(orig_sr, TARGET_SR)
    up = TARGET_SR // g
    down = orig_sr // g

    if up == 1:
        # Pure downsampling (e.g. 48k→16k = factor 3, 192k→16k = factor 12)
        kernel = _lowpass_kernel(down)
        filtered = np.convolve(audio, kernel, mode="same")
        return filtered[::down].astype(np.float32)

    # Non-integer ratio (e.g. 44100→16000): use linear interpolation
    # but with a much cheaper approach than the old np.interp
    n_target = int(len(audio) * TARGET_SR / orig_sr)
    if n_target == 0:
        return np.array([], dtype=np.float32)
    indices = np.linspace(0, len(audio) - 1, n_target)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
