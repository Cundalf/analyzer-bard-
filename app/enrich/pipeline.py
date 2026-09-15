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
from app.enrich import lyrics as lyrics_mod
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
    client: Any | None = None,
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
    stats: dict[str, Any] = {
        "artists": 0,
        "albums": 0,
        "tracks": 0,
        "skipped": 0,
        "languages": 0,
        "audio": 0,
    }

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
                SELECT t.*, a.name AS album_name, ar.navidrome_id AS artist_nid,
                       ar.name AS artist_name
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
                if settings.detect_language and client is not None:
                    detected = _detect_track_language(
                        conn, client, row, progress
                    )
                    if detected:
                        stats["languages"] += 1
                if settings.analyze_audio:
                    analyzed = _analyze_track_audio(
                        conn, row, settings, progress
                    )
                    if analyzed:
                        stats["audio"] += 1
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


def _analyze_track_audio(
    conn: Any,
    track_row: dict[str, Any],
    settings: Settings,
    progress: Progress,
) -> bool:
    """Analiza el audio de una canción (requiere MUSIC_DIR) y pisa la energía."""
    from app.enrich import audio as audio_mod

    music_dir = settings.music_dir
    if not music_dir:
        return False
    track_path = track_row.get("path") or ""
    path = audio_mod.resolve_track_path(track_path, music_dir)
    if path is None:
        return False
    track_id = track_row.get("navidrome_id") or row_id(track_row)
    if not track_id:
        return False
    features = audio_mod.analyze_file(
        path,
        ffmpeg_bin=settings.ffmpeg_bin or None,
        seconds=settings.audio_analysis_seconds,
    )
    if features is None:
        return False
    ficha = artist_mod.get_ficha(conn, "track", track_id)
    if not ficha:
        return False
    facets = audio_mod.apply_audio_features(dict(ficha.get("facets") or {}), features)
    artist_mod.save_ficha(
        conn,
        artist_mod.Ficha(
            entity_type="track",
            entity_id=track_id,
            facets=facets,
            description=ficha.get("description") or "",
            confidence=float(ficha.get("confidence") or 0.0),
            source=ficha.get("source") or "inherited",
            content_hash=ficha.get("content_hash") or "",
        ),
    )
    progress(
        "audio",
        {"track": track_id, "energy": features.energy, "bpm": features.bpm},
    )
    return True


def _detect_track_language(
    conn: Any,
    client: Any,
    track_row: dict[str, Any],
    progress: Progress,
) -> str | None:
    """Detecta el idioma de la letra y lo aplica como dato duro.

    Devuelve el código ISO detectado o None. Nunca rompe el pipeline.
    """
    from app.db import utcnow

    track_id = track_row.get("navidrome_id") or row_id(track_row)
    if not track_id:
        return None
    cached = conn.execute(
        "SELECT lyrics FROM lyrics_cache WHERE track_id = ?", (track_id,)
    ).fetchone()
    if cached is not None:
        lyrics = cached["lyrics"] or ""
    else:
        try:
            lyrics = _fetch_lyrics(client, track_row)
        except Exception as exc:
            log.warning("lyrics fetch failed for %s: %s", track_id, exc)
            return None
        # No cachear letras vacías: el usuario puede agregarlas después.
        if lyrics:
            conn.execute(
                "INSERT INTO lyrics_cache(track_id, lyrics, fetched_at) "
                "VALUES (?, ?, ?) ON CONFLICT(track_id) DO UPDATE SET "
                "lyrics = excluded.lyrics, fetched_at = excluded.fetched_at",
                (track_id, lyrics, utcnow()),
            )
            conn.commit()
    if not lyrics:
        return None
    detected = lyrics_mod.detect_language(lyrics)
    if not detected:
        return None
    code, confidence = detected
    ficha = artist_mod.get_ficha(conn, "track", track_id)
    if not ficha:
        return None
    facets = dict(ficha.get("facets") or {})
    facets["language"] = code
    facets["language_source"] = "lyrics"
    facets["language_confidence"] = confidence
    artist_mod.save_ficha(
        conn,
        artist_mod.Ficha(
            entity_type="track",
            entity_id=track_id,
            facets=facets,
            description=ficha.get("description") or "",
            confidence=float(ficha.get("confidence") or 0.0),
            source=ficha.get("source") or "inherited",
            content_hash=ficha.get("content_hash") or "",
        ),
    )
    progress("language", {"track": track_id, "language": code})
    return code


def _fetch_lyrics(client: Any, track_row: dict[str, Any]) -> str:
    """Letra por ID de canción y, si no hay, por artista/título."""
    track_id = track_row.get("navidrome_id") or row_id(track_row)
    lyrics = ""
    if track_id and hasattr(client, "get_lyrics_by_song_id"):
        lyrics = client.get_lyrics_by_song_id(track_id) or ""
    if lyrics:
        return lyrics
    title = track_row.get("title") or ""
    artist = str(track_row.get("artist_name") or "")
    if title:
        return client.get_lyrics(artist, title) or ""
    return ""


def row_id(track_row: dict[str, Any]) -> str:
    return str(track_row.get("id") or "")


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
