from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


def _appdata_config() -> Path:
    """Persistent user config — lives in %APPDATA%/NoteMe/."""
    base = os.environ.get("APPDATA", str(Path.home()))
    return Path(base) / "NoteMe" / "config.yaml"


def _bundled_config() -> Path:
    """Default config shipped with the app (read-only reference)."""
    if getattr(sys, "frozen", False):
        # Running inside a PyInstaller bundle
        return Path(sys._MEIPASS) / "config.yaml"  # type: ignore[attr-defined]
    # Development: config.yaml at project root
    return Path(__file__).resolve().parent.parent / "config.yaml"


class AudioConfig(BaseModel):
    system_device_index: int = -1   # -1 = auto (default WASAPI loopback)
    mic_device_index: int = -1      # -1 = default mic
    default_source: str = "both"    # system | mic | both
    chunk_duration_sec: float = 3.0


class TranscriptionConfig(BaseModel):
    model: str = "large-v3"
    compute_type: str = "int8"
    device: str = "cpu"
    default_language: str = "auto"  # auto | en | he
    vocabulary_hint: str = (
        "Kubernetes, pods, deployment, namespace, cluster, Docker, container, "
        "CI/CD, pipeline, DevOps, microservices, API, endpoint, load balancer, "
        "AWS, Azure, GCP, Terraform, Helm, Istio, ingress, node, replica set, "
        "NGINX, Redis, PostgreSQL, MongoDB, Kafka, RabbitMQ, gRPC, REST, "
        "GitHub, GitLab, Jenkins, Prometheus, Grafana, Elasticsearch, "
        "CPU, GPU, RAM, SSD, VRAM, CUDA, NVMe, "
        "JIRA, Confluence, Slack, Zoom, Sprint, Scrum, Kanban, "
        "frontend, backend, full-stack, TypeScript, React, Python, Node.js"
    )


class LLMConfig(BaseModel):
    provider: str = "ollama"          # ollama | openai | groq | together | mistral | custom
    base_url: str = "http://localhost:11434"   # Ollama URL or custom endpoint
    model: str = "llama3.1:8b"
    api_key_encrypted: str = ""       # DPAPI-encrypted API key (cloud providers)
    timeout_sec: int = 120


class AppConfig(BaseModel):
    storage_path: str = ""
    startup_with_windows: bool = False
    zoom_auto_record: bool = False    # auto-start recording when Zoom meeting begins

    @property
    def resolved_storage_path(self) -> Path:
        if self.storage_path:
            return Path(self.storage_path)
        return Path.home() / "NoteMe"


class Config(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @classmethod
    def load(cls) -> "Config":
        # 1) User's APPDATA config (editable, persists across updates)
        # 2) Bundled default shipped with the app
        for candidate in (_appdata_config(), _bundled_config()):
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    return cls(**data)
                except Exception:
                    continue
        return cls()

    def save(self) -> None:
        """Always save to %APPDATA%/NoteMe/config.yaml."""
        dest = _appdata_config()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                allow_unicode=True,
            )


_instance: Config | None = None


def get_config() -> Config:
    global _instance
    if _instance is None:
        _instance = Config.load()
    return _instance


def reload_config() -> Config:
    global _instance
    _instance = Config.load()
    return _instance
