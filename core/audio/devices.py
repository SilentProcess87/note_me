from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class AudioDevice:
    index: int
    name: str
    channels: int
    sample_rate: float
    is_loopback: bool = False


def get_loopback_devices() -> List[AudioDevice]:
    """Return available WASAPI loopback (system audio capture) devices."""
    devices: List[AudioDevice] = []
    try:
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()
        try:
            for info in p.get_loopback_device_info_generator():
                devices.append(
                    AudioDevice(
                        index=info["index"],
                        name=info["name"],
                        channels=max(1, info["maxInputChannels"]),
                        sample_rate=info["defaultSampleRate"],
                        is_loopback=True,
                    )
                )
        finally:
            p.terminate()
    except Exception as exc:
        log.warning("Could not enumerate loopback devices: %s", exc)
    return devices


def get_mic_devices() -> List[AudioDevice]:
    """Return available microphone input devices."""
    devices: List[AudioDevice] = []
    try:
        import sounddevice as sd

        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices.append(
                    AudioDevice(
                        index=idx,
                        name=dev["name"],
                        channels=dev["max_input_channels"],
                        sample_rate=dev["default_samplerate"],
                    )
                )
    except Exception as exc:
        log.warning("Could not enumerate mic devices: %s", exc)
    return devices


def get_default_loopback() -> Optional[AudioDevice]:
    """Return the loopback device corresponding to the default speakers."""
    try:
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_out = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            for info in p.get_loopback_device_info_generator():
                if default_out["name"] in info["name"]:
                    return AudioDevice(
                        index=info["index"],
                        name=info["name"],
                        channels=max(1, info["maxInputChannels"]),
                        sample_rate=info["defaultSampleRate"],
                        is_loopback=True,
                    )
        finally:
            p.terminate()
    except Exception as exc:
        log.warning("Could not get default loopback device: %s", exc)
    return None
