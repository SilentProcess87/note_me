"""
Speaker diarization — identifies individual speakers in an audio file.

Uses pyannote-audio to segment audio by speaker voice. Each speaker gets
a label like "Speaker 1", "Speaker 2", etc. The local user's mic is
labeled "You" based on the energy-based detection from AudioCaptureManager.

NOTE: pyannote-audio models are gated on HuggingFace. Users must:
  1. Accept the terms at https://huggingface.co/pyannote/speaker-diarization-3.1
  2. Accept the terms at https://huggingface.co/pyannote/segmentation-3.0
  3. Create a HuggingFace token at https://huggingface.co/settings/tokens
  4. Enter the token in NoteMe Settings
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class SpeakerTurn:
    """A segment of audio attributed to a specific speaker."""
    start_sec: float
    end_sec: float
    speaker: str  # "SPEAKER_00", "SPEAKER_01", etc.


def diarize(audio_path: str, hf_token: str = "", num_speakers: Optional[int] = None) -> List[SpeakerTurn]:
    """
    Run speaker diarization on an audio file.

    Parameters
    ----------
    audio_path : str
        Path to WAV or OGG audio file.
    hf_token : str
        HuggingFace access token (required for pyannote gated models).
    num_speakers : int, optional
        If known, the exact number of speakers. Improves accuracy.

    Returns
    -------
    List of SpeakerTurn with start/end times and speaker labels.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        log.warning("pyannote.audio not installed — skipping diarization")
        return []

    if not Path(audio_path).is_file():
        log.warning("Audio file not found for diarization: %s", audio_path)
        return []

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token or None,
        )

        # Run on GPU if available
        import torch
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers

        log.info("Running speaker diarization on %s...", audio_path)
        diarization = pipeline(audio_path, **kwargs)

        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(SpeakerTurn(
                start_sec=turn.start,
                end_sec=turn.end,
                speaker=speaker,
            ))

        # Rename speakers to friendly labels (Speaker 1, Speaker 2, ...)
        speaker_map = {}
        counter = 1
        for t in turns:
            if t.speaker not in speaker_map:
                speaker_map[t.speaker] = f"Speaker {counter}"
                counter += 1
            t.speaker = speaker_map[t.speaker]

        log.info("Diarization complete: %d turns, %d speakers", len(turns), len(speaker_map))
        return turns

    except Exception as exc:
        log.error("Diarization failed: %s", exc)
        return []


def assign_speakers_to_segments(
    segments: list,
    turns: List[SpeakerTurn],
) -> None:
    """
    Update Segment objects with speaker labels from diarization results.

    For each transcript segment, finds the diarization turn that overlaps
    the most and assigns that speaker label. Preserves "you" labels from
    the energy-based detection if no diarization turn overlaps.
    """
    if not turns:
        return

    for seg in segments:
        # Find the diarization turn with the most overlap
        best_overlap = 0.0
        best_speaker = ""
        for turn in turns:
            overlap_start = max(seg.start_sec, turn.start_sec)
            overlap_end = min(seg.end_sec, turn.end_sec)
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker

        # Only override if we have a meaningful overlap and the segment
        # isn't already labeled "you" from mic detection
        if best_speaker and best_overlap > 0.5:
            if seg.speaker == "you":
                # Keep "you" but add the speaker number for context
                seg.speaker = f"you ({best_speaker})"
            else:
                seg.speaker = best_speaker
