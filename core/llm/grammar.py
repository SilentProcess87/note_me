from __future__ import annotations

import logging
from dataclasses import dataclass

from .client import OllamaClient

log = logging.getLogger(__name__)


@dataclass
class ImprovedText:
    original: str
    improved: str
    notes: str   # Brief bullet points of what was changed


class GrammarService:
    """
    Takes raw speech transcription and returns a polished, grammar-corrected
    version that the user can copy directly into documents, emails, or notes.
    """

    def __init__(self, client: OllamaClient, model: str):
        self._client = client
        self._model = model

    def improve(self, original_text: str, language: str = "auto") -> ImprovedText:
        if language == "he":
            lang_note = "The text is in Hebrew (עברית). Preserve Hebrew throughout."
        elif language == "en":
            lang_note = "The text is in English."
        else:
            lang_note = "Detect the language automatically (may be Hebrew, English, or mixed)."

        prompt = f"""You are a professional editor and language expert. {lang_note}

The following is a raw transcription of spoken speech. It may contain:
- Grammar and spelling mistakes
- Filler words (um, uh, like, you know, אה, אממ, כאילו)
- Run-on sentences or unclear phrasing
- Missing punctuation

Your task:
1. Rewrite the text with correct grammar, proper punctuation, and clear structure
2. Remove all filler words
3. Keep the EXACT same meaning — do not add or remove information
4. Make it suitable for a professional document, email, or note

Format your response EXACTLY as:

IMPROVED:
[only the improved text here — nothing else]

NOTES:
[max 3 short bullet points describing what was changed]

Original transcription:
{original_text}"""

        response = self._client.generate(prompt, self._model)

        improved = original_text
        notes = ""

        if "NOTES:" in response:
            parts = response.split("NOTES:", 1)
            improved_part = parts[0].replace("IMPROVED:", "").strip()
            notes = parts[1].strip()
            improved = improved_part
        elif "IMPROVED:" in response:
            improved = response.replace("IMPROVED:", "").strip()
        else:
            improved = response.strip()

        return ImprovedText(original=original_text, improved=improved, notes=notes)
