from __future__ import annotations

import logging
from typing import List

import requests

log = logging.getLogger(__name__)


class OllamaClient:
    """Thin wrapper around the Ollama REST API (http://localhost:11434)."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def generate(self, prompt: str, model: str) -> str:
        """Send a generation request and return the full response text."""
        url = f"{self._base}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            # Keep the model hot in VRAM between calls (avoids 10-15 s reload).
            "keep_alive": "30m",
            "options": {
                "temperature": 0,       # deterministic = marginally faster
                "num_predict": 900,     # cap output tokens; enough for a meeting summary
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            msg = "Ollama is not running. Please start Ollama and try again."
            log.error(msg)
            return f"[Error: {msg}]"
        except requests.exceptions.HTTPError as exc:
            fallback = self._fallback_model_for_error(exc, model)
            if fallback:
                log.warning("Configured Ollama model '%s' unavailable; retrying with '%s'", model, fallback)
                try:
                    resp = requests.post(
                        url,
                        json={**payload, "model": fallback},
                        timeout=self._timeout,
                    )
                    resp.raise_for_status()
                    return resp.json().get("response", "").strip()
                except Exception as retry_exc:
                    log.error("Ollama fallback request failed: %s", retry_exc)
                    return f"[Error: {retry_exc}]"
            log.error("Ollama request failed: %s", exc)
            return f"[Error: {exc}]"
        except Exception as exc:
            log.error("Ollama request failed: %s", exc)
            return f"[Error: {exc}]"

    def _fallback_model_for_error(self, exc: requests.exceptions.HTTPError, requested_model: str) -> str | None:
        response = exc.response
        if response is None or response.status_code != 404:
            return None
        installed = [m for m in self.list_models() if m != requested_model]
        return installed[0] if installed else None

    def list_models(self) -> List[str]:
        """Return names of locally available models."""
        try:
            resp = requests.get(f"{self._base}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            log.warning("Could not list Ollama models: %s", exc)
            return []

    def is_available(self) -> bool:
        try:
            requests.get(f"{self._base}/api/tags", timeout=3)
            return True
        except Exception:
            return False
