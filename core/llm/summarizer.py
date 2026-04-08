from __future__ import annotations

import logging
from dataclasses import dataclass

from .client import OllamaClient

log = logging.getLogger(__name__)


_DECISIONS_SEP     = "<<<DECISIONS>>>"
_PARTICIPANTS_SEP  = "<<<PARTICIPANTS>>>"


@dataclass
class SummaryResult:
    summary: str
    action_items: str  # encoded: tasks + <<<DECISIONS>>> + decisions + <<<PARTICIPANTS>>> + participants
    title: str = ""   # short meeting title extracted by LLM (updates DbSession.title)


def _section(text: str, marker: str, *stop_markers: str) -> str:
    """Extract content that starts after ``marker`` and ends at the first ``stop_marker``."""
    idx = text.find(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    end = len(text)
    for sm in stop_markers:
        si = text.find(sm, start)
        if si != -1 and si < end:
            end = si
    return text[start:end].strip()


class Summarizer:
    def __init__(self, client: OllamaClient, model: str):
        self._client = client
        self._model = model

    def summarize(self, transcript: str, language: str = "auto") -> SummaryResult:
        if language == "he":
            lang_note = "The meeting was in Hebrew. Respond in Hebrew."
        elif language == "en":
            lang_note = "The meeting was in English. Respond in English."
        else:
            lang_note = "Detect the language and respond in the same language as the meeting."

        # Truncate very long transcripts to avoid slow inference
        if len(transcript) > 10_000:
            transcript = transcript[-10_000:]

        prompt = f"""You are an expert meeting summarizer. {lang_note}

Read the following meeting transcript and provide:
1. TITLE — a short 3-6 word title for this meeting (e.g. \"Q2 Budget Review\").
2. PARTICIPANTS — a comma-separated list of all participant names mentioned. Write \"Unknown\" if none are identifiable.
3. SUMMARY — 2–3 concise paragraphs covering the main topics.
4. DECISIONS — concrete conclusions reached. Bullet points. \"None\" if empty.
5. TASKS — follow-up actions assigned to people. Bullet points. \"None\" if empty.

Format your response EXACTLY as:

TITLE:
[3-6 word title]

PARTICIPANTS:
[comma-separated names]

SUMMARY:
[summary here]

DECISIONS:
[bullet list or None]

TASKS:
[bullet list or None]

Meeting transcript:
{transcript}"""

        response = self._client.generate(prompt, self._model)

        # Extract each section robustly — sections may appear in any order.
        title        = _section(response, "TITLE:",        "PARTICIPANTS:", "SUMMARY:", "DECISIONS:", "TASKS:")
        participants = _section(response, "PARTICIPANTS:", "SUMMARY:",      "DECISIONS:", "TASKS:")
        summary      = _section(response, "SUMMARY:",      "DECISIONS:",    "TASKS:")
        decisions    = _section(response, "DECISIONS:",    "TASKS:")
        tasks        = _section(response, "TASKS:")

        # Legacy fallback: ACTION ITEMS format
        if not summary and not tasks and "ACTION ITEMS:" in response:
            parts = response.split("ACTION ITEMS:", 1)
            summary = parts[0].replace("SUMMARY:", "").strip()
            tasks = parts[1].strip()
        elif not summary:
            summary = response.strip()

        # Build encoded action_items:  tasks  <<<DECISIONS>>>  decisions  <<<PARTICIPANTS>>>  names
        encoded = tasks
        if decisions and decisions.lower() not in ("none", ""):
            encoded = f"{tasks}\n{_DECISIONS_SEP}\n{decisions}"
        if participants and participants.lower() not in ("unknown", ""):
            encoded = f"{encoded}\n{_PARTICIPANTS_SEP}\n{participants}"

        # Clean up title: remove surrounding quotes if LLM added them
        title = title.strip('"\' ')

        return SummaryResult(summary=summary, action_items=encoded, title=title)
