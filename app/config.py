from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    subsonic_url: str = "http://localhost:4533"
    subsonic_user: str = "admin"
    subsonic_pass: str = ""

    ollama_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma4:cloud"
    ollama_embed_model: str = "nomic-embed-text:v1.5"
    ollama_timeout: float = 600.0

    lastfm_api_key: str = ""
    enable_lastfm: bool = False

    music_dir: str = ""
    data_dir: str = "./data"
    janitor_enabled: bool = False
    acoustid_key: str = ""
    beets_config: str = ""
    beets_bin: str = ""
    beetsdir: str = ""
    ffmpeg_bin: str = ""
    wav_backup_dir: str = ""
    wav_delete_originals: bool = False
    beets_import_quiet: bool = True

    embed_dim: int = 768
    agent_enabled: bool = True
    agent_max_tool_calls: int = 6
    web_search_enabled: bool = False
    enrich_tracks_llm: bool = False
    enrich_tracks_inherit: bool = True
    detect_language: bool = True
    analyze_audio: bool = False
    audio_analysis_seconds: int = 30

    playlist_default_size: int = 30
    playlist_max_size: int = 100
    playlist_rerank_pool: int = 60

    web_host: str = "0.0.0.0"
    web_port: int = 8080

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    @property
    def db_path(self) -> Path:
        return self.data_path / "bardo.db"

    @property
    def runs_path(self) -> Path:
        return self.data_path / "runs"

    @property
    def wav_backup_path(self) -> Path:
        if self.wav_backup_dir:
            return Path(self.wav_backup_dir).expanduser().resolve()
        return self.data_path / "wav_originals"

    def ensure_dirs(self) -> None:
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.runs_path.mkdir(parents=True, exist_ok=True)


OVERRIDABLE: tuple[str, ...] = (
    "subsonic_url",
    "subsonic_user",
    "subsonic_pass",
    "ollama_url",
    "ollama_chat_model",
    "ollama_embed_model",
    "lastfm_api_key",
    "enable_lastfm",
    "janitor_enabled",
    "agent_enabled",
    "web_search_enabled",
    "enrich_tracks_llm",
    "detect_language",
    "analyze_audio",
    "embed_dim",
    "playlist_default_size",
    "playlist_max_size",
    "audio_analysis_seconds",
)

BOOL_FIELDS = {
    "enable_lastfm",
    "janitor_enabled",
    "agent_enabled",
    "web_search_enabled",
    "enrich_tracks_llm",
    "detect_language",
    "analyze_audio",
}
INT_FIELDS = {
    "embed_dim",
    "playlist_default_size",
    "playlist_max_size",
    "audio_analysis_seconds",
}


def coerce_override(field: str, value: Any) -> Any:
    if field in BOOL_FIELDS:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if field in INT_FIELDS:
        return int(value)
    return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


def runtime_settings(conn: Any | None = None) -> Settings:
    base = get_settings()
    if conn is None:
        return base
    from app.db import all_settings

    overrides: dict[str, Any] = {}
    for key, value in all_settings(conn).items():
        if key in OVERRIDABLE and value not in (None, ""):
            overrides[key] = coerce_override(key, value)
    if overrides:
        return base.model_copy(update=overrides)
    return base
