"""
LLM provider factory.

Creates the appropriate client (Ollama or OpenAI-compatible) based on config.
All clients expose the same interface: generate(prompt, model) -> str
"""
from __future__ import annotations

import logging
from typing import Union

from .client import OllamaClient
from .openai_client import OpenAIClient, PROVIDER_URLS, PROVIDER_DEFAULT_MODELS

log = logging.getLogger(__name__)

# Provider names the user sees in Settings
PROVIDERS = [
    ("ollama",    "Ollama (Local — fully offline)"),
    ("groq",      "Groq  (free tier, fast)"),
    ("openai",    "OpenAI"),
    ("together",  "Together AI"),
    ("mistral",   "Mistral AI"),
    ("custom",    "Custom OpenAI-compatible API"),
]

LLMClient = Union[OllamaClient, OpenAIClient]


def create_llm_client(
    provider: str,
    api_key: str = "",
    base_url: str = "",
    timeout: int = 120,
) -> LLMClient:
    """
    Build an LLM client for the given provider.

    Parameters
    ----------
    provider : str
        One of: ollama, openai, groq, together, mistral, custom
    api_key : str
        Decrypted API key (required for cloud providers, ignored for ollama).
    base_url : str
        Override URL. For ollama defaults to localhost:11434.
        For cloud providers, defaults to their known URL.
        For 'custom', this must be provided.
    timeout : int
        Request timeout in seconds.
    """
    if provider == "ollama":
        url = base_url or "http://localhost:11434"
        return OllamaClient(url, timeout)

    # All other providers use the OpenAI-compatible client
    if provider in PROVIDER_URLS:
        url = base_url or PROVIDER_URLS[provider]
    elif provider == "custom":
        url = base_url or "http://localhost:8000/v1"
    else:
        url = base_url or PROVIDER_URLS.get("openai", "")

    return OpenAIClient(url, api_key, timeout)


def default_model_for_provider(provider: str) -> str:
    """Return a sensible default model name for the given provider."""
    if provider == "ollama":
        return "llama3.1:8b"
    return PROVIDER_DEFAULT_MODELS.get(provider, "gpt-4o-mini")


def needs_api_key(provider: str) -> bool:
    """Return True if this provider requires an API key."""
    return provider != "ollama"
