from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from app.enrich.canonicalize import Canonicalizer, canonical_facets, content_hash
from app.enrich.merge import apply_hard_genre_to_artist
from app.enrich.prompts import ARTIST_SCHEMA, artist_messages
from app.ollama import OllamaClient

log = logging.getLogger("bardo.enrich.artist")


@dataclass
class Ficha:
    entity_type: str
    entity_id: str
    facets: dict[str, Any]
    description: str
    confidence: float
    source: str
    content_hash: str


def _hash_inputs(payload: Any) -> str:
    return content_hash(payload)


def save_ficha(conn: Any, ficha: Ficha) -> None:
    from app.db import facet_upsert, fts_upsert, utcnow
    from app.enrich.merge import merge_confidence, merge_description, merge_ficha_facets

    previous = get_ficha(conn, ficha.entity_type, ficha.entity_id)
    facets = merge_ficha_facets(previous["facets"] if previous else None, ficha.facets)
    description = merge_description(
        previous.get("description") if previous else None, ficha.description
    )
    confidence = merge_confidence(
        previous.get("confidence") if previous else None, ficha.confidence
    )

    conn.execute(
        """
        INSERT INTO fichas(entity_type, entity_id, facets, description,
                           confidence, source, content_hash, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(entity_type, entity_id) DO UPDATE SET
          facets = excluded.facets, description = excluded.description,
          confidence = excluded.confidence, source = excluded.source,
          content_hash = excluded.content_hash, updated_at = excluded.updated_at
        """,
        (
            ficha.entity_type,
            ficha.entity_id,
            json.dumps(facets, ensure_ascii=False),
            description,
            confidence,
            ficha.source,
            ficha.content_hash,
            utcnow(),
        ),
    )
    from app.enrich.canonicalize import ficha_text

    fts_upsert(
        conn,
        ficha.entity_type,
        ficha.entity_id,
        ficha_text(facets, description),
    )
    facet_upsert(conn, ficha.entity_type, ficha.entity_id, facets)
    conn.commit()


def get_ficha(conn: Any, entity_type: str, entity_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM fichas WHERE entity_type = ? AND entity_id = ?",
        (entity_type, entity_id),
    ).fetchone()
    if not row:
        return None
    data = dict(row)
    try:
        data["facets"] = json.loads(data.get("facets") or "{}")
    except json.JSONDecodeError:
        data["facets"] = {}
    return data


def is_stale(conn: Any, entity_type: str, entity_id: str, payload: Any) -> bool:
    row = conn.execute(
        "SELECT content_hash FROM fichas WHERE entity_type = ? AND entity_id = ?",
        (entity_type, entity_id),
    ).fetchone()
    if not row:
        return True
    return row["content_hash"] != _hash_inputs(payload)


def mark_needs_janitor(
    conn: Any,
    entity_type: str,
    entity_id: str,
    reason: str,
    *,
    hash_payload: Any = None,
) -> None:
    """Marca una entidad genérica como pendiente del Módulo A.

    No gasta LLM: con "[Unknown Artist]" el modelo alucinaría. La ficha
    queda con la marca para que el dashboard la cuente y el Janitor la
    arregle con fingerprinting.
    """
    from app.db import facet_upsert, fts_upsert, utcnow

    facets = {"needs_janitor": True, "needs_janitor_reason": reason}
    conn.execute(
        """
        INSERT INTO fichas(entity_type, entity_id, facets, description,
                           confidence, source, content_hash, updated_at)
        VALUES (?, ?, ?, '', 0.0, 'pending', ?, ?)
        ON CONFLICT(entity_type, entity_id) DO UPDATE SET
          facets = excluded.facets, source = 'pending',
          content_hash = excluded.content_hash, updated_at = excluded.updated_at
        """,
        (
            entity_type,
            entity_id,
            json.dumps(facets, ensure_ascii=False),
            _hash_inputs(hash_payload if hash_payload is not None else reason),
            utcnow(),
        ),
    )
    fts_upsert(conn, entity_type, entity_id, "")
    facet_upsert(conn, entity_type, entity_id, facets)
    conn.commit()


def is_generic_entity(name: str | None) -> bool:
    from app.enrich.generic import is_generic

    return is_generic(name)


async def enrich_artist(
    conn: Any,
    ollama: OllamaClient,
    canon: Canonicalizer,
    artist_row: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any] | None:
    navidrome_id = artist_row.get("navidrome_id") or artist_row["id"]
    if is_generic_entity(artist_row.get("name")):
        mark_needs_janitor(conn, "artist", navidrome_id, "artista genérico", hash_payload="generic")
        return None
    albums = [
        r["name"]
        for r in conn.execute(
            "SELECT name FROM albums WHERE artist_id = ? "
            "AND TRIM(COALESCE(name, '')) != '' ORDER BY year",
            (artist_row["id"],),
        ).fetchall()
    ]
    genres = [
        r["genre"]
        for r in conn.execute(
            "SELECT DISTINCT genre FROM albums WHERE artist_id = ? AND COALESCE(genre, '') != ''",
            (artist_row["id"],),
        ).fetchall()
    ]
    payload = {
        "name": artist_row["name"],
        "albums": sorted(albums),
        "genres": sorted(genres),
    }
    if not force and not is_stale(conn, "artist", navidrome_id, payload):
        return None

    messages = artist_messages(artist_row["name"], genres, albums)
    try:
        raw = await ollama.chat_json(messages, schema=ARTIST_SCHEMA)
    except Exception as exc:
        log.warning("artist enrich failed for %s: %s", artist_row["name"], exc)
        return None

    facets = canonical_facets(raw, canon)
    facets = apply_hard_genre_to_artist(facets, genres, canon)
    description = str(facets.pop("description", "") or "")
    confidence = float(facets.pop("confidence", 0) or 0)
    ficha = Ficha(
        entity_type="artist",
        entity_id=navidrome_id,
        facets=facets,
        description=description,
        confidence=confidence,
        source="llm",
        content_hash=_hash_inputs(payload),
    )
    save_ficha(conn, ficha)
    return {"entity_id": navidrome_id, "name": artist_row["name"]}
