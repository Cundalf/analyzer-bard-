from __future__ import annotations

import logging
from typing import Any, Callable

from app.config import Settings, get_settings
from app.db import start_run, finish_run
from app.enrich import artist as artist_mod
from app.enrich import album as album_mod
from app.enrich import track as track_mod
from app.enrich.canonicalize import Canonicalizer
from app.enrich.lastfm import LastFmClient
from app.ollama import OllamaClient

log = logging.getLogger("bardo.enrich.pipeline")

Progress = Callable[[str, dict[str, Any]], None]


def _noop(stage: str, payload: dict[str, Any]) -> None:
    pass


async def enrich_library(
    conn: Any,
    *,
    settings: Settings | None = None,
    ollama: OllamaClient | None = None,
    lastfm: LastFmClient | None = None,
    artists: bool = True,
    albums: bool = True,
    tracks: bool = True,
    limit: int | None = None,
    force: bool = False,
    progress: Progress | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    progress = progress or _noop
    owns_ollama = ollama is None
    ollama = ollama or OllamaClient(settings)
    lastfm = lastfm or LastFmClient(settings)
    canon = Canonicalizer.from_db(conn)

    run_id = start_run(conn, "enrich", "full")
    stats: dict[str, Any] = {"artists": 0, "albums": 0, "tracks": 0, "skipped": 0}

    try:
        if artists:
            rows = conn.execute(
                "SELECT * FROM artists ORDER BY name"
                + (f" LIMIT {int(limit)}" if limit else "")
            ).fetchall()
            for i, row in enumerate(rows, 1):
                row = dict(row)
                enriched = await artist_mod.enrich_artist(
                    conn, ollama, canon, row, force=force
                )
                if enriched:
                    stats["artists"] += 1
                else:
                    stats["skipped"] += 1
                if settings.enable_lastfm and settings.lastfm_api_key:
                    tags = lastfm.fetch_for(
                        conn, "artist", row["navidrome_id"] or row["id"], row["name"]
                    )
                    if tags:
                        _merge_lastfm_into_ficha(
                            conn, "artist", row["navidrome_id"] or row["id"], tags
                        )
                        progress("lastfm_artist", {"name": row["name"], "tags": tags})
                progress("artist", {"i": i, "total": len(rows), **row})

        if albums:
            rows = conn.execute(
                "SELECT * FROM albums ORDER BY name"
                + (f" LIMIT {int(limit)}" if limit else "")
            ).fetchall()
            for i, row in enumerate(rows, 1):
                row = dict(row)
                enriched = await album_mod.enrich_album(
                    conn, ollama, canon, row, force=force
                )
                if enriched:
                    stats["albums"] += 1
                else:
                    stats["skipped"] += 1
                progress("album", {"i": i, "total": len(rows), **row})

        if tracks:
            rows = conn.execute(
                """
                SELECT t.*, a.name AS album_name, ar.navidrome_id AS artist_nid
                FROM tracks t
                LEFT JOIN albums a ON a.id = t.album_id
                LEFT JOIN artists ar ON ar.id = t.artist_id
                ORDER BY t.album_id, t.id
                """
                + (f" LIMIT {int(limit)}" if limit else "")
            ).fetchall()
            for i, row in enumerate(rows, 1):
                row = dict(row)
                navidrome_id = row.get("navidrome_id") or row["id"]
                album_ficha = (
                    artist_mod.get_ficha(
                        conn,
                        "album",
                        _album_navidrome_id(conn, row.get("album_id")),
                    )
                    if row.get("album_id")
                    else None
                )
                artist_ficha = (
                    artist_mod.get_ficha(
                        conn,
                        "artist",
                        _artist_navidrome_id(conn, row.get("artist_id")),
                    )
                    if row.get("artist_id")
                    else None
                )
                if settings.enrich_tracks_llm:
                    enriched = await track_mod.enrich_track_llm(
                        conn,
                        ollama,
                        canon,
                        row,
                        album_ficha,
                        artist_ficha,
                        force=force,
                    )
                    if enriched:
                        stats["tracks"] += 1
                    else:
                        stats["skipped"] += 1
                elif settings.enrich_tracks_inherit:
                    if force or artist_mod.is_stale(
                        conn, "track", navidrome_id, {"row": navidrome_id}
                    ):
                        ficha = track_mod.inherit_track_ficha(
                            conn, row, album_ficha, artist_ficha
                        )
                        if ficha:
                            artist_mod.save_ficha(conn, ficha)
                            stats["tracks"] += 1
                        else:
                            stats["skipped"] += 1
                    else:
                        stats["skipped"] += 1
                progress("track", {"i": i, "total": len(rows), "title": row.get("title")})

        finish_run(conn, run_id, "ok", stats)
    except Exception as exc:
        log.exception("enrich failed")
        finish_run(conn, run_id, "error", {**stats, "error": str(exc)})
        raise
    finally:
        if owns_ollama:
            await ollama.close()
        lastfm.close()

    return {"run_id": run_id, **stats}


def _merge_lastfm_into_ficha(
    conn: Any, entity_type: str, entity_id: str, tags: list[str]
) -> None:
    """Agrega crowd tags de Last.fm a una ficha existente sin pisar el LLM."""
    from app.enrich.merge import merge_hard_facets

    ficha = artist_mod.get_ficha(conn, entity_type, entity_id)
    if not ficha:
        return
    facets = merge_hard_facets(ficha["facets"], lastfm_tags=tags)
    artist_mod.save_ficha(
        conn,
        artist_mod.Ficha(
            entity_type=entity_type,
            entity_id=entity_id,
            facets=facets,
            description=ficha.get("description") or "",
            confidence=float(ficha.get("confidence") or 0.0),
            source=ficha.get("source") or "llm",
            content_hash=ficha.get("content_hash") or "",
        ),
    )


def _album_navidrome_id(conn: Any, album_pk: str) -> str:
    row = conn.execute(
        "SELECT navidrome_id FROM albums WHERE id = ?", (album_pk,)
    ).fetchone()
    return (row["navidrome_id"] if row else album_pk) or album_pk


def _artist_navidrome_id(conn: Any, artist_pk: str) -> str:
    row = conn.execute(
        "SELECT navidrome_id FROM artists WHERE id = ?", (artist_pk,)
    ).fetchone()
    return (row["navidrome_id"] if row else artist_pk) or artist_pk
