from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from .resample import TARGET_SR, resample_to_16k

log = logging.getLogger(__name__)


class AudioCaptureManager:
    """
    Manages system audio (WASAPI loopback) and/or microphone capture.

    Calls ``on_audio_ready(audio: np.ndarray)`` with 3-second (configurable)
    16 kHz float32 mono chunks suitable for faster-whisper.

    source: "system" | "mic" | "both"
    """

    def __init__(
        self,
        on_audio_ready: Callable[[np.ndarray], None],
        on_levels: Optional[Callable[[float, float], None]] = None,
        source: str = "both",
        system_device_index: int = -1,
        mic_device_index: int = -1,
        chunk_duration_sec: float = 3.0,
        save_audio_path: Optional[str] = None,
    ):
        self._callback = on_audio_ready
        self._on_levels = on_levels
        self._source = source
        self._sys_idx = system_device_index
        self._mic_idx = mic_device_index
        self._chunk_samples = int(TARGET_SR * chunk_duration_sec)
        self._save_path = save_audio_path
        self._wav_file = None  # wave.open handle

        # Current smoothed RMS levels (written by mixer thread, read by Qt timer)
        self.mic_rms: float = 0.0
        self.sys_rms: float = 0.0

        # Raw PCM queues fed by device callbacks
        self._sys_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._mic_queue: queue.Queue[np.ndarray] = queue.Queue()

        # Accumulation buffers (already resampled to 16 kHz)
        self._sys_buf: list[np.ndarray] = []
        self._mic_buf: list[np.ndarray] = []
        self._sys_sr: int = TARGET_SR
        self._mic_sr: int = TARGET_SR

        self._running = False
        self._mixer_thread: Optional[threading.Thread] = None
        self._pa_system = None
        self._sys_stream = None
        self._sd_stream = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._running = True
        # Open WAV file for recording if a path was provided
        if self._save_path:
            try:
                import wave
                self._wav_file = wave.open(self._save_path, "wb")
                self._wav_file.setnchannels(1)
                self._wav_file.setsampwidth(2)  # 16-bit
                self._wav_file.setframerate(TARGET_SR)
            except Exception as exc:
                log.error("Could not open WAV file '%s': %s", self._save_path, exc)
                self._wav_file = None
        if self._source in ("system", "both"):
            self._start_system()
        if self._source in ("mic", "both"):
            self._start_mic()
        self._mixer_thread = threading.Thread(target=self._mixer_loop, daemon=True, name="AudioMixer")
        self._mixer_thread.start()
        log.info("AudioCaptureManager started (source=%s)", self._source)

    def stop(self) -> None:
        self._running = False
        self._stop_system()
        self._stop_mic()
        if self._mixer_thread:
            self._mixer_thread.join(timeout=6)
            self._mixer_thread = None
        if self._wav_file:
            try:
                self._wav_file.close()
                log.info("Audio saved to %s", self._save_path)
            except Exception as exc:
                log.error("Error closing WAV file: %s", exc)
            self._wav_file = None
        log.info("AudioCaptureManager stopped")

    # ------------------------------------------------------------------ #
    # System audio — WASAPI loopback via pyaudiowpatch
    # ------------------------------------------------------------------ #

    def _start_system(self) -> None:
        try:
            import pyaudiowpatch as pyaudio

            self._pa_system = pyaudio.PyAudio()
            p = self._pa_system

            if self._sys_idx >= 0:
                device = p.get_device_info_by_index(self._sys_idx)
            else:
                wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_out = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                device = None
                for lb in p.get_loopback_device_info_generator():
                    if default_out["name"] in lb["name"]:
                        device = lb
                        break
                if device is None:
                    log.warning("No loopback device found — skipping system audio")
                    return

            self._sys_sr = int(device["defaultSampleRate"])
            channels = max(1, device["maxInputChannels"])

            # WASAPI loopback format: some devices only work with int16,
            # others only with float32. Try both formats until one produces data.
            self._sys_use_float = False
            for fmt, use_float, fmt_name in [
                (pyaudio.paInt16,   False, "paInt16"),
                (pyaudio.paFloat32, True,  "paFloat32"),
            ]:
                try:
                    self._sys_use_float = use_float

                    def _callback(in_data, frame_count, time_info, status,
                                  _float=use_float, _ch=channels):
                        if _float:
                            raw = np.frombuffer(in_data, dtype=np.float32)
                        else:
                            raw = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                        if _ch > 1:
                            raw = raw.reshape(-1, _ch).mean(axis=1)
                        self._sys_queue.put(raw.copy())
                        return (in_data, pyaudio.paContinue)

                    self._sys_stream = p.open(
                        format=fmt,
                        channels=channels,
                        rate=self._sys_sr,
                        input=True,
                        input_device_index=device["index"],
                        frames_per_buffer=4096,
                        stream_callback=_callback,
                    )
                    log.info("System capture: %s @ %d Hz (%s)",
                             device["name"], self._sys_sr, fmt_name)
                    break
                except Exception as e:
                    log.debug("System capture format %s failed: %s", fmt_name, e)
                    continue
        except Exception as exc:
            log.error("Failed to start system capture: %s", exc)

    def _stop_system(self) -> None:
        try:
            if self._sys_stream:
                self._sys_stream.stop_stream()
                self._sys_stream.close()
                self._sys_stream = None
            if self._pa_system:
                self._pa_system.terminate()
                self._pa_system = None
        except Exception as exc:
            log.warning("Error stopping system capture: %s", exc)

    # ------------------------------------------------------------------ #
    # Microphone — sounddevice
    # ------------------------------------------------------------------ #

    def _start_mic(self) -> None:
        try:
            import sounddevice as sd

            device = self._mic_idx if self._mic_idx >= 0 else None
            dev_info = sd.query_devices(device, "input")
            self._mic_sr = int(dev_info["default_samplerate"])

            def _callback(indata, frames, time_info, status):
                self._mic_queue.put(indata[:, 0].copy().astype(np.float32))

            self._sd_stream = sd.InputStream(
                samplerate=self._mic_sr,
                device=device,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=_callback,
            )
            self._sd_stream.start()
            log.info("Mic capture started @ %d Hz", self._mic_sr)
        except Exception as exc:
            log.error("Failed to start mic capture: %s", exc)

    def _stop_mic(self) -> None:
        try:
            if self._sd_stream:
                self._sd_stream.stop()
                self._sd_stream.close()
                self._sd_stream = None
        except Exception as exc:
            log.warning("Error stopping mic capture: %s", exc)

    # ------------------------------------------------------------------ #
    # Mixer loop — drains queues, resamples, mixes, fires callback
    # ------------------------------------------------------------------ #

    def _drain(self, q: queue.Queue, buf: list, sr: int) -> float:
        """Drain a device queue into the accumulation buffer. Returns peak RMS."""
        peak_rms = 0.0
        while not q.empty():
            try:
                chunk = q.get_nowait()
                if len(chunk):
                    peak_rms = max(peak_rms, float(np.sqrt(np.mean(chunk ** 2))))
                buf.append(resample_to_16k(chunk, sr))
            except queue.Empty:
                break
        return peak_rms

    def _mixer_loop(self) -> None:
        use_sys = self._source in ("system", "both")
        use_mic = self._source in ("mic", "both")

        while self._running:
            time.sleep(0.1)

            if use_sys:
                sys_rms = self._drain(self._sys_queue, self._sys_buf, self._sys_sr)
                self.sys_rms = max(sys_rms, self.sys_rms * 0.75)  # smooth decay
            if use_mic:
                mic_rms = self._drain(self._mic_queue, self._mic_buf, self._mic_sr)
                self.mic_rms = max(mic_rms, self.mic_rms * 0.75)  # smooth decay

            if self._on_levels:
                self._on_levels(self.mic_rms, self.sys_rms)

            sys_total = sum(len(a) for a in self._sys_buf)
            mic_total = sum(len(a) for a in self._mic_buf)

            # Use the LARGEST buffer to decide when to fire a chunk.
            # Previously this only checked sys_total, which meant no chunks
            # were delivered when system audio was silent (nothing playing
            # through speakers → WASAPI loopback produces no data).
            if use_sys and use_mic:
                primary = max(sys_total, mic_total)
            elif use_sys:
                primary = sys_total
            else:
                primary = mic_total

            if primary < self._chunk_samples:
                continue

            sys_audio = self._pop_chunk(self._sys_buf) if use_sys else None
            mic_audio = self._pop_chunk(self._mic_buf) if use_mic else None

            mixed = self._mix(sys_audio, mic_audio)
            if mixed is not None and len(mixed) > 0:
                self._write_wav(mixed)
                try:
                    self._callback(mixed)
                except Exception as exc:
                    log.error("Audio callback error: %s", exc)

        # Flush remaining audio on stop
        if use_sys:
            self._drain(self._sys_queue, self._sys_buf, self._sys_sr)
        if use_mic:
            self._drain(self._mic_queue, self._mic_buf, self._mic_sr)
        sys_audio = np.concatenate(self._sys_buf) if self._sys_buf else None
        mic_audio = np.concatenate(self._mic_buf) if self._mic_buf else None
        self._sys_buf.clear()
        self._mic_buf.clear()
        mixed = self._mix(sys_audio, mic_audio)
        if mixed is not None and len(mixed) > 0:
            self._write_wav(mixed)
            try:
                self._callback(mixed)
            except Exception as exc:
                log.error("Flush callback error: %s", exc)

    def _write_wav(self, audio: np.ndarray) -> None:
        """Write a float32 audio chunk to the WAV file as 16-bit PCM."""
        if self._wav_file is None:
            return
        try:
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            self._wav_file.writeframes(pcm.tobytes())
        except Exception as exc:
            log.error("WAV write error: %s", exc)

    def _pop_chunk(self, buf: list) -> Optional[np.ndarray]:
        """Extract exactly chunk_samples from the buffer, keeping excess."""
        if not buf:
            return None
        full = np.concatenate(buf)
        buf.clear()
        if len(full) > self._chunk_samples:
            buf.append(full[self._chunk_samples:])
            return full[: self._chunk_samples]
        return full

    @staticmethod
    def _mix(
        sys_audio: Optional[np.ndarray],
        mic_audio: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if sys_audio is not None and mic_audio is not None:
            # Pad the shorter source with silence so we don't lose the tail.
            length = max(len(sys_audio), len(mic_audio))
            if len(sys_audio) < length:
                sys_audio = np.pad(sys_audio, (0, length - len(sys_audio)))
            if len(mic_audio) < length:
                mic_audio = np.pad(mic_audio, (0, length - len(mic_audio)))

            # Smart mix: if one source is effectively silent, pass the other
            # through at full volume so VAD can still detect speech.
            sys_energy = float(np.max(np.abs(sys_audio)))
            mic_energy = float(np.max(np.abs(mic_audio)))
            silence_threshold = 0.005

            if sys_energy < silence_threshold:
                return mic_audio               # system silent → mic only
            if mic_energy < silence_threshold:
                return sys_audio               # mic silent → system only
            # Both active: average, but with a gentle 0.7 factor to avoid
            # excessive attenuation (louder than *0.5 but still headroom-safe).
            return np.clip((sys_audio + mic_audio) * 0.7, -1.0, 1.0)
        if sys_audio is not None:
            return sys_audio
        if mic_audio is not None:
            return mic_audio
        return None
