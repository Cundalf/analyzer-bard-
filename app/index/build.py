from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from app.config import Settings, get_settings
from app.enrich.canonicalize import ficha_text
from app.ollama import OllamaClient

log = logging.getLogger("bardo.index.build")

Progress = Callable[[str, dict[str, Any]], None]


def _noop(stage: str, payload: dict[str, Any]) -> None:
    pass


async def build_index(
    conn: Any,
    *,
    settings: Settings | None = None,
    ollama: OllamaClient | None = None,
    force: bool = False,
    limit: int | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    from app.db import vec_upsert

    settings = settings or get_settings()
    progress = progress or _noop
    owns = ollama is None
    ollama = ollama or OllamaClient(settings)

    rows = conn.execute(
        "SELECT entity_type, entity_id, facets, description, source "
        "FROM fichas ORDER BY entity_type, entity_id" + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()

    stats = {"total": len(rows), "embedded": 0, "skipped": 0, "errors": 0}
    try:
        for i, row in enumerate(rows, 1):
            row = dict(row)
            import json

            try:
                facets = json.loads(row["facets"] or "{}")
            except json.JSONDecodeError:
                facets = {}
            text = ficha_text(facets, row["description"] or "")
            if not text.strip():
                stats["skipped"] += 1
                continue
            try:
                vector = await ollama.embed_one(text)
                if len(vector) != settings.embed_dim:
                    log.warning(
                        "embedding dim mismatch: got %s expected %s",
                        len(vector),
                        settings.embed_dim,
                    )
                vec_upsert(conn, row["entity_type"], row["entity_id"], vector)
                stats["embedded"] += 1
            except Exception as exc:
                stats["errors"] += 1
                log.warning(
                    "embed failed for %s/%s: %s",
                    row["entity_type"],
                    row["entity_id"],
                    exc,
                )
            if i % 25 == 0:
                conn.commit()
            progress(
                "embed",
                {
                    "i": i,
                    "total": len(rows),
                    "entity": f"{row['entity_type']}/{row['entity_id']}",
                },
            )
        conn.commit()
    finally:
        if owns:
            await ollama.close()
    return stats
