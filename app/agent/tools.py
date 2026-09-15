from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.agent.retrieval import (
    FacetFilters,
    rrf_merge,
    search_by_terms,
    search_fts,
    search_vectors,
)
from app.config import Settings, get_settings
from app.enrich.canonicalize import Canonicalizer
from app.ollama import OllamaClient
from app.subsonic import SubsonicClient

log = logging.getLogger("bardo.agent.tools")


@dataclass
class ToolContext:
    conn: Any
    ollama: OllamaClient
    settings: Settings
    subsonic: SubsonicClient | None = None
    canon: Canonicalizer | None = None
    web_search: Any | None = None
    allow_create: bool = False
    created_playlists: list[dict[str, Any]] = field(default_factory=list)
    validation: list[dict[str, Any]] = field(default_factory=list)


def tool_schemas(settings: Settings | None = None) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_candidates",
                "description": (
                    "Búsqueda híbrida (vector + full-text + tags) sobre la "
                    "biblioteca. Devuelve hasta 60 candidatos reales con "
                    "track_id. Usala primero, siempre."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "pedido o términos de búsqueda",
                        },
                        "limit": {"type": "integer", "default": 50},
                        "year_min": {"type": "integer"},
                        "year_max": {"type": "integer"},
                        "exclude_genres": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "include_genres": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "languages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "códigos ISO 639-1: es, en, pt, ja... "
                                "(o nombres, se normalizan)"
                            ),
                        },
                        "countries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "códigos ISO 3166-1 alpha-2: AR, ES, MX, US..."
                            ),
                        },
                        "decades": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "décadas como 1980, 1990, 2000",
                        },
                        "energy_min": {"type": "number"},
                        "energy_max": {"type": "number"},
                        "instrumental": {"type": "boolean"},
                        "ballad": {"type": "boolean"},
                        "concept_album": {"type": "boolean"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_artists",
                "description": "Lista artistas de la biblioteca que matchean un texto.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_albums",
                "description": "Lista los álbumes de un artista (por nombre).",
                "parameters": {
                    "type": "object",
                    "required": ["artist"],
                    "properties": {"artist": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_tracks",
                "description": "Lista los temas de un álbum (por nombre de álbum).",
                "parameters": {
                    "type": "object",
                    "required": ["album"],
                    "properties": {
                        "album": {"type": "string"},
                        "artist": {"type": "string"},
                    },
                },
            },
        },
    ]
    if settings.web_search_enabled:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Valida UN dato factual sobre música. Usar sólo para "
                        "confirmar una hipótesis fuerte, nunca de forma masiva."
                    ),
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        )
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "create_playlist",
                "description": (
                    "Crea la playlist en Navidrome con track_ids EXACTOS "
                    "devueltos por search_candidates/list_tracks."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["name", "track_ids"],
                    "properties": {
                        "name": {"type": "string"},
                        "track_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "reasoning": {"type": "string"},
                    },
                },
            },
        }
    )
    return tools


def _candidate_payload(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "track_id": c["track_id"],
            "title": c["title"],
            "artist": c["artist"],
            "album": c["album"],
            "moods": c["moods"][:6],
            "themes": c["themes"][:6],
        }
        for c in candidates
    ]


def validate_track_ids(conn: Any, track_ids: list[str]) -> list[str]:
    """Anti-alucinación: sólo IDs que existen en el espejo de Navidrome.

    Deduplica preservando el orden (evita temas repetidos en la playlist).
    """
    if not track_ids:
        return []
    unique = list(dict.fromkeys(track_ids))
    valid: list[str] = []
    for start in range(0, len(unique), 400):
        chunk = unique[start : start + 400]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT navidrome_id FROM tracks WHERE navidrome_id IN ({marks})",
            chunk,
        ).fetchall()
        present = {r["navidrome_id"] for r in rows}
        valid.extend(tid for tid in chunk if tid in present)
    return valid


async def execute_tool(
    name: str, arguments: dict[str, Any], ctx: ToolContext
) -> Any:
    conn = ctx.conn
    if name == "search_candidates":
        query = str(arguments.get("query", ""))
        limit = int(arguments.get("limit") or 50)
        limit = max(1, min(limit, 60))
        filters = FacetFilters.from_dict(arguments)

        terms = [t for t in query.replace(",", " ").split() if len(t) > 2]
        canon = ctx.canon or Canonicalizer.from_db(conn)
        expanded = canon.expand(terms) or terms

        results = [
            search_fts(
                conn,
                query,
                limit=limit,
                filters=filters,
            ),
            search_by_terms(
                conn,
                expanded,
                limit=limit,
                filters=filters,
            ),
        ]
        try:
            embedding = await ctx.ollama.embed_one(query)
            results.append(
                search_vectors(conn, embedding, limit=limit, filters=filters)
            )
        except Exception as exc:
            log.warning("vector search unavailable: %s", exc)

        merged = rrf_merge(results)[:limit]
        if not merged:
            merged = search_by_terms(
                conn, terms, limit=limit, filters=filters
            )
        payload: dict[str, Any] = {
            "count": len(merged),
            "candidates": _candidate_payload(merged),
        }
        if not merged:
            payload["note"] = (
                "No hay resultados con los filtros aplicados. "
                "Probá quitar el filtro más restrictivo (energy, decade o "
                "instrumental) o buscar el género por nombre."
            )
        return payload

    if name == "list_artists":
        query = str(arguments.get("query", "")).strip()
        pattern = f"%{query}%"
        rows = conn.execute(
            "SELECT navidrome_id, name FROM artists WHERE name LIKE ? "
            "ORDER BY name LIMIT 40",
            (pattern,),
        ).fetchall()
        return {"artists": [dict(r) for r in rows]}

    if name == "list_albums":
        artist = str(arguments.get("artist", "")).strip()
        pattern = f"%{artist}%"
        rows = conn.execute(
            """
            SELECT al.navidrome_id, al.name, al.year, ar.name AS artist
            FROM albums al LEFT JOIN artists ar ON ar.id = al.artist_id
            WHERE ar.name LIKE ? OR al.name LIKE ?
            ORDER BY al.year LIMIT 50
            """,
            (pattern, pattern),
        ).fetchall()
        return {"albums": [dict(r) for r in rows]}

    if name == "list_tracks":
        album = str(arguments.get("album", "")).strip()
        artist = str(arguments.get("artist", "")).strip()
        pattern = f"%{album}%"
        clauses = ["al.name LIKE ?"]
        params: list[Any] = [pattern]
        if artist:
            clauses.append("ar.name LIKE ?")
            params.append(f"%{artist}%")
        params.append(80)
        rows = conn.execute(
            f"""
            SELECT t.navidrome_id AS track_id, t.title, t.duration,
                   al.name AS album, ar.name AS artist
            FROM tracks t
            LEFT JOIN albums al ON al.id = t.album_id
            LEFT JOIN artists ar ON ar.id = t.artist_id
            WHERE {' AND '.join(clauses)}
            ORDER BY t.id LIMIT ?
            """,
            params,
        ).fetchall()
        return {"tracks": [dict(r) for r in rows]}

    if name == "web_search":
        if not ctx.web_search:
            return {"error": "web_search deshabilitado"}
        query = str(arguments.get("query", ""))
        return await ctx.web_search(query, ctx)

    if name == "create_playlist":
        track_ids = [str(t) for t in arguments.get("track_ids", [])]
        playlist_name = str(arguments.get("name", "")).strip() or "Bardo"
        valid = validate_track_ids(conn, track_ids)
        rejected = [t for t in track_ids if t not in set(valid)]
        ctx.validation.append(
            {
                "requested": len(track_ids),
                "valid": len(valid),
                "rejected": rejected[:20],
            }
        )
        if not ctx.allow_create:
            return {
                "status": "preview",
                "name": playlist_name,
                "valid_track_ids": valid,
                "rejected_track_ids": rejected,
                "message": "Preview: la playlist no se guardó todavía.",
            }
        if not valid:
            return {"error": "no hay track_ids válidos; no se creó la playlist"}
        if ctx.subsonic is None:
            return {"error": "cliente Subsonic no disponible"}
        playlist = ctx.subsonic.create_playlist(playlist_name, valid)
        payload = {
            "id": playlist.id,
            "name": playlist.name,
            "song_count": playlist.song_count or len(valid),
            "track_ids": valid,
        }
        ctx.created_playlists.append(payload)
        return {"status": "created", **payload}

    return {"error": f"tool desconocida: {name}"}


def tool_result_message(tool_name: str, result: Any) -> str:
    text = json.dumps(result, ensure_ascii=False)
    if len(text) > 24000:
        text = text[:24000] + "...(truncado)"
    return text
