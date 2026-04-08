"""
OpenAI-compatible LLM client.

Works with any provider that implements the /v1/chat/completions endpoint:
  - OpenAI (api.openai.com)
  - Groq (api.groq.com)
  - Together AI (api.together.xyz)
  - Mistral AI (api.mistral.ai)
  - Any local server with OpenAI-compat (LM Studio, vLLM, etc.)
"""
from __future__ import annotations

import logging
from typing import List

import requests

log = logging.getLogger(__name__)

# Pre-configured provider base URLs
PROVIDER_URLS = {
    "openai":   "https://api.openai.com/v1",
    "groq":     "https://api.groq.com/openai/v1",
    "together":  "https://api.together.xyz/v1",
    "mistral":  "https://api.mistral.ai/v1",
    "anthropic": "https://api.anthropic.com/v1",
}

# Sensible default models per provider
PROVIDER_DEFAULT_MODELS = {
    "openai":    "gpt-4o-mini",
    "groq":      "llama-3.1-70b-versatile",
    "together":  "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "mistral":   "mistral-small-latest",
    "anthropic": "claude-3-5-haiku-latest",
}


class OpenAIClient:
    """Drop-in replacement for OllamaClient using the OpenAI chat completions API."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 120):
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def generate(self, prompt: str, model: str) -> str:
        """Send a chat completion request and return the response text."""
        url = f"{self._base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 1200,
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            msg = f"Cannot connect to {self._base}. Check your internet connection."
            log.error(msg)
            return f"[Error: {msg}]"
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            body = exc.response.text[:200] if exc.response else ""
            if status == 401:
                msg = "Invalid API key. Please check your key in Settings."
            elif status == 429:
                msg = "Rate limit exceeded. Please wait a moment and try again."
            elif status == 404:
                msg = f"Model '{model}' not found for this provider."
            else:
                msg = f"API error {status}: {body}"
            log.error(msg)
            return f"[Error: {msg}]"
        except Exception as exc:
            log.error("OpenAI-compatible API request failed: %s", exc)
            return f"[Error: {exc}]"

    def list_models(self) -> List[str]:
        """List available models (best-effort; not all providers support this)."""
        url = f"{self._base}/models"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            return [m["id"] for m in models]
        except Exception as exc:
            log.debug("Could not list models: %s", exc)
            return []

    def is_available(self) -> bool:
        """Quick connectivity check."""
        try:
            url = f"{self._base}/models"
            headers = {"Authorization": f"Bearer {self._api_key}"}
            resp = requests.get(url, headers=headers, timeout=5)
            return resp.status_code in (200, 401)  # 401 = key bad but server reachable
        except Exception:
            return False
