"""Compatibilidad: re-exporta los prompts usados por el agente.

Los prompts viven en `app.enrich.prompts` (compartidos con el enriquecimiento).
Este módulo existe para respetar la estructura del spec (`agent/prompts.py`).
"""

from app.enrich.prompts import (
    AGENT_SYSTEM,
    EXPANSION_SCHEMA,
    EXPANSION_SYSTEM,
    RERANK_SCHEMA,
    RERANK_SYSTEM,
    WEB_SEARCH_SYSTEM,
    expansion_messages,
    rerank_messages,
)

__all__ = [
    "AGENT_SYSTEM",
    "EXPANSION_SCHEMA",
    "EXPANSION_SYSTEM",
    "RERANK_SCHEMA",
    "RERANK_SYSTEM",
    "WEB_SEARCH_SYSTEM",
    "expansion_messages",
    "rerank_messages",
]
