from __future__ import annotations

import logging
from typing import Any

from app.enrich.artist import Ficha, _hash_inputs, is_stale, save_ficha
from app.enrich.canonicalize import Canonicalizer, canonical_facets
from app.enrich.prompts import TRACK_SCHEMA, track_messages
from app.ollama import OllamaClient

log = logging.getLogger("bardo.enrich.track")


def inherit_track_ficha(
    conn: Any,
    track_row: dict[str, Any],
    album_ficha: dict[str, Any] | None,
    artist_ficha: dict[str, Any] | None,
) -> Ficha | None:
    """Herencia: ficha de canción derivada de álbum + artista (sin LLM).

    Regla de precedencia: canción > álbum > artista. Los campos escalares
    (idioma, país, género, energy) y las listas se completan desde el padre
    más cercano que los tenga.
    """
    if not album_ficha and not artist_ficha:
        return None
    album_facets = dict((album_ficha or {}).get("facets", {}) or {})
    artist_facets = dict((artist_ficha or {}).get("facets", {}) or {})
    title = track_row.get("title") or ""

    def pick_list(field: str) -> list[str]:
        for source in (album_facets, artist_facets):
            values = source.get(field)
            if isinstance(values, list) and values:
                return list(values)
        return []

    def pick_scalar(field: str, default: Any = None) -> Any:
        for source in (album_facets, artist_facets):
            value = source.get(field)
            if value not in (None, "", []):
                return value
        return default

    moods = pick_list("moods")
    themes = pick_list("themes") or pick_list("references") or pick_list("lyrical_themes")
    description_parts = [
        f"{track_row.get('title', '')} de {artist_facets.get('name', '') or ''}".strip()
    ]
    if album_ficha and album_ficha.get("description"):
        description_parts.append(album_ficha["description"])
    description = ". ".join(p for p in description_parts if p)[:400]

    lowered = title.lower()
    is_instrumental = any(
        hint in lowered for hint in ("instrumental", "intro", "outro", "interlude", "prelude")
    )
    is_ballad = any(hint in lowered for hint in ("ballad", "balada", "lament", "elegy"))

    genres = pick_list("genres") or pick_list("subgenres")
    facets: dict[str, Any] = {
        "title": title,
        "moods": moods,
        "themes": themes,
        "energy": pick_scalar("energy", 0.0),
        "is_ballad": is_ballad,
        "is_instrumental": is_instrumental,
        "inherited_from": {
            "album": album_ficha["entity_id"] if album_ficha else "",
            "artist": artist_ficha["entity_id"] if artist_ficha else "",
        },
    }
    optional_scalars = ("language", "country", "vocal_style", "era")
    for field in optional_scalars:
        value = pick_scalar(field)
        if value not in (None, ""):
            facets[field] = value
    for field in ("instrumentation", "for_fans_of", "references"):
        values = pick_list(field)
        if values:
            facets[field] = values
    if genres:
        facets["genres"] = genres
    if album_facets.get("is_concept_album") is not None:
        facets["is_concept_album"] = album_facets["is_concept_album"]

    payload = {
        "track": track_row.get("navidrome_id") or track_row["id"],
        "album_ficha": album_ficha["content_hash"] if album_ficha else "",
        "artist_ficha": artist_ficha["content_hash"] if artist_ficha else "",
    }
    parent_confidences = [
        float(ficha.get("confidence") or 0.0) for ficha in (album_ficha, artist_ficha) if ficha
    ]
    confidence = min(parent_confidences) if parent_confidences else 0.5
    return Ficha(
        entity_type="track",
        entity_id=track_row.get("navidrome_id") or track_row["id"],
        facets=facets,
        description=description,
        confidence=confidence,
        source="inherited",
        content_hash=_hash_inputs(payload),
    )


async def enrich_track_llm(
    conn: Any,
    ollama: OllamaClient,
    canon: Canonicalizer,
    track_row: dict[str, Any],
    album_ficha: dict[str, Any] | None,
    artist_ficha: dict[str, Any] | None,
    *,
    lyrics: str = "",
    force: bool = False,
) -> dict[str, Any] | None:
    navidrome_id = track_row.get("navidrome_id") or track_row["id"]
    album_facets = (album_ficha or {}).get("facets", {})
    artist_name = ""
    if track_row.get("artist_id"):
        artist = conn.execute(
            "SELECT name FROM artists WHERE id = ?", (track_row["artist_id"],)
        ).fetchone()
        artist_name = artist["name"] if artist else ""
    payload = {
        "title": track_row.get("title"),
        "album": track_row.get("album_id"),
        "lyrics": lyrics[:500],
    }
    if not force and not is_stale(conn, "track", navidrome_id, payload):
        return None

    messages = track_messages(
        artist_name,
        track_row.get("album_name", "") or "",
        track_row.get("title", "") or "",
        str(album_facets.get("concept", "")),
        list(album_facets.get("themes", [])) + list(album_facets.get("moods", [])),
        lyrics,
    )
    try:
        raw = await ollama.chat_json(messages, schema=TRACK_SCHEMA)
    except Exception as exc:
        log.warning("track enrich failed for %s: %s", track_row.get("title"), exc)
        return None
    facets = canonical_facets(raw, canon)
    description = str(facets.pop("description", "") or "")
    confidence = float(facets.pop("confidence", 0) or 0)
    ficha = Ficha(
        entity_type="track",
        entity_id=navidrome_id,
        facets=facets,
        description=description,
        confidence=confidence,
        source="llm",
        content_hash=_hash_inputs(payload),
    )
    save_ficha(conn, ficha)
    return {"entity_id": navidrome_id, "title": track_row.get("title")}
