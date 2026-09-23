"""Fusión de facetas: los datos duros pisan lo que dijo el LLM.

Fuentes duras: tags de Navidrome/archivos (género), Last.fm (crowd tags) y
detección de idioma por letra. El LLM completa el resto y nunca sobreescribe
un hecho.

Regla general: nada válido se pierde. Un valor genérico ("Unknown",
"Various Artists") no es información y nunca pisa ni entra.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

from app.enrich.generic import filter_valid, is_generic
from app.enrich.vocab import (
    normalize_countries,
    normalize_languages,
    normalize_energy,
)

log = logging.getLogger("bardo.enrich.merge")

HARD_SOURCE = "hard"

# Facetas donde una fuente dura (audio/letra/tag) no debe ser pisada por el LLM.
HARD_PROTECTED: dict[str, str] = {
    "language_source": "language",
    "energy_source": "energy",
}

LIST_FIELDS: tuple[str, ...] = (
    "genres",
    "subgenres",
    "moods",
    "themes",
    "references",
    "instrumentation",
    "lyrical_themes",
    "for_fans_of",
    "lastfm_tags",
)

SCALAR_FIELDS: tuple[str, ...] = (
    "country",
    "era",
    "vocal_style",
    "concept",
)

SOURCE_FIELDS: tuple[str, ...] = ("language_source", "energy_source")


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
    if normalized_genre and is_generic(normalized_genre):
        normalized_genre = ""
    if normalized_genre:
        existing = out.get("genres")
        merged = (
            [normalized_genre]
            if not isinstance(existing, list) or not existing
            else _dedupe([normalized_genre, *existing])
        )
        out["genres"] = filter_valid(merged)

    tags = filter_valid(lastfm_tags or [])
    if tags:
        out["lastfm_tags"] = tags
        if not filter_valid(out.get("genres") or []):
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
    hard = filter_valid(genres_from_library or [])
    out = dict(facets or {})
    if not hard:
        return out
    raw_llm = out.get("genres")
    llm_genres = (
        filter_valid(raw_llm) if isinstance(raw_llm, list) else []
    )
    preferred = [g for g in hard if g.lower() in {x.lower() for x in llm_genres}]
    rest = [g for g in hard if g not in preferred]
    merged = filter_valid(preferred + rest + llm_genres)[:12]
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


def merge_ficha_facets(
    old: dict[str, Any] | None,
    new: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fusiona facetas sin perder información válida.

    - Listas: se unen (viejas primero) y se descartan genéricos/duplicados.
    - Escalares: el valor nuevo gana sólo si es válido; si no, se conserva el viejo.
    - Fuentes duras: si el viejo tiene `language_source`/`energy_source`,
      el campo protegido no se toca (el LLM no pisa lo medido).
    """
    old = dict(old or {})
    new = dict(new or {})
    result: dict[str, Any] = {}
    protected = _protected_fields(old)

    for field in LIST_FIELDS:
        combined = filter_valid(list(old.get(field) or []) + list(new.get(field) or []))
        if combined:
            result[field] = combined

    for field in SCALAR_FIELDS:
        new_value = new.get(field)
        old_value = old.get(field)
        if is_generic(new_value):
            if not is_generic(old_value):
                result[field] = old_value
        else:
            result[field] = new_value

    # `language` y `energy` con reglas propias (normalización + protección)
    if "language" in protected:
        result["language"] = old["language"]
        if old.get("languages"):
            result["languages"] = old["languages"]
    else:
        languages = normalize_languages(
            new.get("language") or new.get("languages")
        ) or normalize_languages(old.get("language") or old.get("languages"))
        if languages:
            result["language"] = languages[0]
            if len(languages) > 1:
                result["languages"] = languages

    if "energy" in protected:
        result["energy"] = old["energy"]
    else:
        energy = normalize_energy(new.get("energy"))
        if energy is None:
            energy = normalize_energy(old.get("energy"))
        if energy is not None:
            result["energy"] = energy

    countries = normalize_countries(new.get("country") or new.get("countries"))
    if not countries:
        countries = normalize_countries(old.get("country") or old.get("countries"))
    if countries:
        result["country"] = countries[0]
        if len(countries) > 1:
            result["countries"] = countries

    for flag in ("is_ballad", "is_instrumental", "is_concept_album"):
        value = new.get(flag)
        if value is None:
            value = old.get(flag)
        if value is not None:
            result[flag] = value

    for source_field in SOURCE_FIELDS:
        value = new.get(source_field) or old.get(source_field)
        if value:
            result[source_field] = value
    if old.get("genres_source") == HARD_SOURCE or new.get("genres_source") == HARD_SOURCE:
        result["genres_source"] = HARD_SOURCE
    confidence = new.get("language_confidence")
    if confidence is None:
        confidence = old.get("language_confidence")
    if confidence is not None:
        result["language_confidence"] = confidence

    for extra_field in ("inherited_from",):
        value = new.get(extra_field) or old.get(extra_field)
        if value:
            result[extra_field] = value

    return result


def _protected_fields(old: dict[str, Any]) -> set[str]:
    """Campos que el LLM no puede pisar porque hay una medición dura previa."""
    protected: set[str] = set()
    for source_field, target in HARD_PROTECTED.items():
        if old.get(source_field) and not is_generic(old.get(target)):
            protected.add(target)
    return protected


def merge_description(
    old: str | None, new: str | None
) -> str:
    """La descripción nueva sólo gana si aporta algo; nunca borra una válida."""
    cleaned_new = _clean(new)
    if cleaned_new and not is_generic(cleaned_new):
        return cleaned_new
    return _clean(old)


def merge_confidence(old: Any, new: Any) -> float:
    """Si el valor nuevo es 0/vacío pero había uno previo, se conserva."""
    try:
        new_value = float(new or 0)
    except (TypeError, ValueError):
        new_value = 0.0
    try:
        old_value = float(old or 0)
    except (TypeError, ValueError):
        old_value = 0.0
    return new_value if new_value > 0 else old_value
