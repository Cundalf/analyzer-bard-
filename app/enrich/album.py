from __future__ import annotations

import logging
from typing import Any

from app.enrich.artist import Ficha, _hash_inputs, get_ficha, is_stale, save_ficha
from app.enrich.canonicalize import Canonicalizer, canonical_facets
from app.enrich.merge import apply_hard_genre_to_artist
from app.enrich.prompts import ALBUM_SCHEMA, album_messages
from app.ollama import OllamaClient

log = logging.getLogger("bardo.enrich.album")


async def enrich_album(
    conn: Any,
    ollama: OllamaClient,
    canon: Canonicalizer,
    album_row: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any] | None:
    navidrome_id = album_row.get("navidrome_id") or album_row["id"]
    from app.enrich.artist import is_generic_entity, mark_needs_janitor

    if is_generic_entity(album_row.get("name")):
        mark_needs_janitor(
            conn, "album", navidrome_id, "álbum genérico", hash_payload="generic"
        )
        return None
    artist_name = ""
    artist_genres: list[str] = []
    artist_ficha = None
    if album_row.get("artist_id"):
        artist = conn.execute(
            "SELECT * FROM artists WHERE id = ?", (album_row["artist_id"],)
        ).fetchone()
        if artist:
            artist_name = artist["name"] or ""
            artist_ficha = get_ficha(
                conn, "artist", artist["navidrome_id"] or artist["id"]
            )
            if artist_ficha:
                artist_genres = artist_ficha.get("facets", {}).get("genres", [])
        if not artist_genres:
            artist_genres = [
                r["genre"]
                for r in conn.execute(
                    "SELECT DISTINCT genre FROM albums WHERE artist_id = ? "
                    "AND COALESCE(genre, '') != ''",
                    (album_row["artist_id"],),
                ).fetchall()
            ]

    track_titles = [
        r["title"]
        for r in conn.execute(
            "SELECT title FROM tracks WHERE album_id = ? ORDER BY id",
            (album_row["id"],),
        ).fetchall()
    ]
    payload = {
        "artist": artist_name,
        "album": album_row["name"],
        "year": album_row["year"],
        "tracks": track_titles,
    }
    if not force and not is_stale(conn, "album", navidrome_id, payload):
        return None

    messages = album_messages(
        artist_name,
        album_row["name"] or "",
        album_row["year"],
        artist_genres,
        track_titles,
    )
    try:
        raw = await ollama.chat_json(messages, schema=ALBUM_SCHEMA)
    except Exception as exc:
        log.warning(
            "album enrich failed for %s - %s: %s",
            artist_name,
            album_row["name"],
            exc,
        )
        return None

    facets = canonical_facets(raw, canon)
    facets = apply_hard_genre_to_artist(
        facets, [album_row["genre"]] if album_row.get("genre") else [], canon
    )
    description = str(facets.pop("description", "") or "")
    confidence = float(facets.pop("confidence", 0) or 0)
    ficha = Ficha(
        entity_type="album",
        entity_id=navidrome_id,
        facets=facets,
        description=description,
        confidence=confidence,
        source="llm",
        content_hash=_hash_inputs(payload),
    )
    save_ficha(conn, ficha)
    return {
        "entity_id": navidrome_id,
        "artist": artist_name,
        "album": album_row["name"],
    }
