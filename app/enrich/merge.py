"""Fusión de facetas: los datos duros pisan lo que dijo el LLM.

Fuentes duras: tags de Navidrome/archivos (género), Last.fm (crowd tags) y
detección de idioma por letra. El LLM completa el resto y nunca sobreescribe
un hecho.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

from app.enrich.vocab import (
    normalize_countries,
    normalize_languages,
    normalize_energy,
)

log = logging.getLogger("bardo.enrich.merge")

HARD_SOURCE = "hard"


def merge_hard_facets(
    facets: dict[str, Any],
    *,
    genre: str | None = None,
    lastfm_tags: Iterable[str] | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """Aplica hechos duros sobre una ficha ya canonicalizada.

    - genre (Navidrome/archivo): reemplaza `genres` si el tag existe.
    - lastfm_tags: si el LLM no dio géneros, los usa; además se guardan en
      `lastfm_tags` para FTS/facetas.
    - language (letra): reemplaza el idioma detectado por el LLM.
    Devuelve una copia; nunca muta la entrada.
    """
    out = dict(facets or {})

    normalized_genre = _clean(genre)
    if normalized_genre:
        existing = out.get("genres")
        merged = (
            [normalized_genre]
            if not isinstance(existing, list) or not existing
            else _dedupe([normalized_genre, *existing])
        )
        out["genres"] = merged

    tags = _dedupe(
        _clean(tag) for tag in (lastfm_tags or []) if _clean(tag)
    )
    if tags:
        out["lastfm_tags"] = tags
        if not out.get("genres"):
            out["genres"] = tags[:10]

    normalized_language = normalize_languages(language)
    if normalized_language:
        out["language"] = normalized_language[0]
        if len(normalized_language) > 1:
            out["languages"] = normalized_language

    countries = normalize_countries(out.get("country"))
    if countries:
        out["country"] = countries[0]

    energy = normalize_energy(out.get("energy"))
    if energy is not None:
        out["energy"] = energy
    return out


def apply_hard_genre_to_artist(
    facets: dict[str, Any],
    genres_from_library: Iterable[str],
    canon: Any | None = None,
) -> dict[str, Any]:
    """El tag de género de la biblioteca pisa los géneros del LLM.

    Navidrome agrupa `album.genre`; es un hecho observable del archivo.
    Si hay varios, se prioriza el que ya figuraba en la respuesta del LLM
    (para conservar su forma canónica) y se completan los demás.
    """
    hard = _dedupe(
        _clean(g) for g in genres_from_library if _clean(g)
    )
    if not hard:
        return dict(facets or {})
    out = dict(facets or {})
    llm_genres = [
        _clean(g) for g in out.get("genres", []) if _clean(g)
    ] if isinstance(out.get("genres"), list) else []
    preferred = [g for g in hard if g.lower() in {x.lower() for x in llm_genres}]
    rest = [g for g in hard if g not in preferred]
    merged = _dedupe(preferred + rest + llm_genres)[:12]
    if canon is not None:
        merged = canon.canonical_list(merged)
    out["genres"] = merged
    out["genres_source"] = HARD_SOURCE
    return out


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out
