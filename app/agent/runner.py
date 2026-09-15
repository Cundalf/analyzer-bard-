from __future__ import annotations

import logging
from typing import Any, Callable

from app.agent import retrieval
from app.agent.loop import run_agent
from app.agent.retrieval import FacetFilters
from app.agent.tools import ToolContext, validate_track_ids
from app.config import Settings, get_settings
from app.enrich.canonicalize import Canonicalizer
from app.enrich.prompts import EXPANSION_SCHEMA, expansion_messages, rerank_messages
from app.ollama import OllamaClient

log = logging.getLogger("bardo.agent.runner")

Progress = Callable[[str, dict[str, Any]], None]


def _noop(stage: str, payload: dict[str, Any]) -> None:
    pass


async def expand_prompt(
    ollama: OllamaClient,
    canon: Canonicalizer,
    prompt: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    try:
        data = await ollama.chat_json(
            expansion_messages(prompt), schema=EXPANSION_SCHEMA
        )
        if not isinstance(data, dict):
            data = {}
    except Exception as exc:
        log.warning("expansion failed, falling back to plain terms: %s", exc)
        data = {}

    canonical_terms = canon.canonical_list(data.get("canonical_terms") or [])
    expanded = canon.expand(
        list(data.get("expanded_terms") or []) + canonical_terms
    )
    moods = canon.canonical_list(data.get("moods") or [])
    filters_raw = data.get("filters")
    if not isinstance(filters_raw, dict):
        filters_raw = {}
    filters = FacetFilters.from_dict(filters_raw)
    size = _coerce_int(data.get("size"), settings.playlist_default_size)
    size = max(5, min(size, settings.playlist_max_size))
    return {
        "intent": data.get("intent") or "playlist",
        "canonical_terms": canonical_terms,
        "expanded_terms": expanded,
        "moods": moods,
        "reference": data.get("reference") or "",
        "filters": filters,
        "size": size,
        "raw_prompt": prompt,
    }


def _coerce_int(value: Any, default: int | None) -> int | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def recall_candidates(
    conn: Any,
    plan: dict[str, Any],
    *,
    limit: int | None = None,
    embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    settings = get_settings()
    limit = limit or settings.playlist_rerank_pool
    filters = plan.get("filters") or FacetFilters()
    terms = list(dict.fromkeys(plan.get("expanded_terms", [])))
    moods = list(plan.get("moods") or [])
    if moods:
        terms = list(dict.fromkeys(terms + moods))
    if not terms:
        terms = [plan["raw_prompt"]]

    results = [
        retrieval.search_fts(
            conn,
            plan["raw_prompt"],
            limit=limit,
            filters=filters,
        ),
        retrieval.search_by_terms(
            conn,
            terms,
            limit=limit,
            filters=filters,
        ),
    ]
    if embedding:
        results.append(
            retrieval.search_vectors(conn, embedding, limit=limit, filters=filters)
        )
    merged = retrieval.rrf_merge(results)
    if not merged:
        merged = retrieval.search_by_terms(conn, terms, limit=limit, filters=filters)
    return merged


def _candidate_payload(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "track_id": c["track_id"],
            "title": c["title"],
            "artist": c["artist"],
            "album": c["album"],
            "year": c["year"],
            "genre": c["genre"],
            "moods": (c.get("moods") or [])[:5],
            "themes": (c.get("themes") or [])[:5],
        }
        for c in candidates
    ]


async def rerank_candidates(
    conn: Any,
    ollama: OllamaClient,
    prompt: str,
    candidates: list[dict[str, Any]],
    size: int,
    moods: list[str] | None = None,
    reference: str = "",
) -> dict[str, Any]:
    if not candidates:
        return {"playlist_name": "", "track_ids": [], "reasoning": "sin candidatos"}
    payload = _candidate_payload(candidates)
    try:
        raw = await ollama.chat_json(
            rerank_messages(prompt, payload, size, moods, reference)
        )
    except Exception as exc:
        log.warning("rerank failed: %s", exc)
        raw = {}
    requested = [str(t) for t in raw.get("track_ids", [])]
    valid = validate_track_ids(conn, requested)
    if not valid:
        valid = [c["track_id"] for c in candidates[:size]]
        raw["reasoning"] = (
            str(raw.get("reasoning", ""))
            + " [fallback: rerank no devolvió IDs válidos]"
        ).strip()
    return {
        "playlist_name": raw.get("playlist_name") or f"Bardo: {prompt[:40]}",
        "track_ids": valid[:size],
        "reasoning": raw.get("reasoning", ""),
    }


async def generate_playlist(
    conn: Any,
    prompt: str,
    *,
    settings: Settings | None = None,
    ollama: OllamaClient | None = None,
    client: Any | None = None,
    save: bool = False,
    use_agent: bool = True,
    size: int | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Pipeline completo: expansión → recall → rerank/agente → validación → guardar."""
    settings = settings or get_settings()
    progress = progress or _noop
    owns = ollama is None
    ollama = ollama or OllamaClient(settings)
    canon = Canonicalizer.from_db(conn)
    close_client = False

    try:
        if client is None:
            from app.subsonic import SubsonicClient

            client = SubsonicClient(settings)
            close_client = True

        plan = await expand_prompt(ollama, canon, prompt, settings)
        progress("plan", plan)
        size = size or plan["size"]

        embedding: list[float] | None = None
        try:
            embedding = await ollama.embed_one(prompt)
        except Exception as exc:
            log.warning("embedding unavailable (Fase 3 sin vectores?): %s", exc)

        candidates = recall_candidates(
            conn, plan, limit=settings.playlist_rerank_pool, embedding=embedding
        )
        progress("recall", {"count": len(candidates)})

        if use_agent and settings.agent_enabled:
            ctx = ToolContext(
                conn=conn,
                ollama=ollama,
                settings=settings,
                subsonic=client,
                canon=canon,
                allow_create=save,
            )
            agent_result = await run_agent(
                conn,
                prompt,
                ctx=ctx,
                settings=settings,
                ollama=ollama,
                progress=progress,
            )
            created = agent_result["created_playlists"]
            validation = agent_result["validation"]
            result = {
                "playlist_name": (
                    created[0]["name"]
                    if created
                    else _name_from_trace(agent_result) or f"Bardo: {prompt[:40]}"
                ),
                "track_ids": (
                    created[0]["track_ids"]
                    if created
                    else _ids_from_trace(agent_result, conn)
                ),
                "reasoning": agent_result.get("text", ""),
                "tool_calls": agent_result["tool_calls"],
                "trace": agent_result["trace"],
                "created_playlists": created,
                "validation": validation,
                "saved": bool(created),
                "mode": "agent",
            }
            if save and not created and result["track_ids"]:
                result.update(
                    _save_direct(client, result["playlist_name"], result["track_ids"])
                )
            return result

        reranked = await rerank_candidates(
            conn, ollama, prompt, candidates, size,
            moods=plan.get("moods"), reference=plan.get("reference", ""),
        )
        progress("rerank", reranked)
        result = {
            "playlist_name": reranked["playlist_name"],
            "track_ids": reranked["track_ids"],
            "reasoning": reranked["reasoning"],
            "tool_calls": 0,
            "trace": [],
            "created_playlists": [],
            "validation": [],
            "saved": False,
            "mode": "rerank",
        }
        if save and result["track_ids"]:
            result.update(
                _save_direct(client, result["playlist_name"], result["track_ids"])
            )
        return result
    finally:
        if owns:
            await ollama.close()
        if close_client and client is not None:
            client.close()


def _save_direct(client: Any, name: str, track_ids: list[str]) -> dict[str, Any]:
    playlist = client.create_playlist(name, track_ids)
    return {
        "saved": True,
        "created_playlists": [
            {
                "id": playlist.id,
                "name": playlist.name,
                "song_count": playlist.song_count or len(track_ids),
                "track_ids": track_ids,
            }
        ],
    }


def _name_from_trace(agent_result: dict[str, Any]) -> str:
    for entry in reversed(agent_result.get("trace", [])):
        if entry.get("tool") == "create_playlist":
            name = (entry.get("arguments") or {}).get("name")
            if name:
                return str(name)
    return ""


def _ids_from_trace(agent_result: dict[str, Any], conn: Any) -> list[str]:
    for entry in reversed(agent_result.get("trace", [])):
        if entry.get("tool") == "create_playlist":
            ids = (entry.get("arguments") or {}).get("track_ids") or []
            return validate_track_ids(conn, [str(i) for i in ids])
    return []
