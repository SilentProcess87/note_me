from __future__ import annotations

import logging
import re

from .client import OllamaClient

log = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 8_000


class QAService:
    def __init__(self, client: OllamaClient, model: str):
        self._client = client
        self._model = model

    def answer(self, question: str, transcript: str, language: str = "auto") -> str:
        """Answer a question grounded in the meeting transcript."""
        context = self._select_context(question, transcript)

        if language == "he":
            lang_note = "Answer in Hebrew."
        elif language == "en":
            lang_note = "Answer in English."
        else:
            lang_note = "Answer in the same language as the question."

        prompt = f"""You are a helpful assistant with access to a meeting transcript. {lang_note}

Answer the question using ONLY information from the transcript below.
If the answer is not found in the transcript, say so clearly.

Meeting transcript:
{context}

Question: {question}
Answer:"""

        return self._client.generate(prompt, self._model)

    # ------------------------------------------------------------------ #

    def _select_context(self, question: str, transcript: str) -> str:
        if len(transcript) <= MAX_CONTEXT_CHARS:
            return transcript
        # Simple keyword-relevance ranking — no vector DB required
        keywords = set(re.sub(r"[^\w\s]", "", question.lower()).split())
        lines = transcript.splitlines()
        scored = [(sum(1 for kw in keywords if kw in line.lower()), line) for line in lines]
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = "\n".join(line for _, line in scored[:60])
        return selected[:MAX_CONTEXT_CHARS]
