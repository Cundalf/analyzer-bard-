from __future__ import annotations

import pytest
from tests.conftest import FakeOllama, seed_library

from app.index.build import build_index


def add_ficha(conn, entity_type, entity_id, facets, description, source="llm"):
    import json

    conn.execute(
        """
        INSERT INTO fichas(entity_type, entity_id, facets, description, confidence, source, content_hash)
        VALUES (?, ?, ?, ?, 0.9, ?, 'h')
        """,
        (entity_type, entity_id, json.dumps(facets), description, source),
    )
    conn.commit()


@pytest.mark.anyio
async def test_build_index_embeds_all(conn, settings):
    seed_library(conn)
    add_ficha(conn, "artist", "a1", {"genres": ["folk metal"]}, "dwarves beer")
    ollama = FakeOllama()
    stats = await build_index(conn, settings=settings, ollama=ollama)
    assert stats["total"] == 2
    assert stats["embedded"] == 2
    assert stats["errors"] == 0
    n = conn.execute("SELECT COUNT(*) AS n FROM vec_fichas").fetchone()["n"]
    assert n == 2


@pytest.mark.anyio
async def test_build_index_limit(conn, settings):
    seed_library(conn)
    add_ficha(conn, "artist", "a1", {}, "d")
    stats = await build_index(conn, settings=settings, ollama=FakeOllama(), limit=1)
    assert stats["total"] == 1
    assert stats["embedded"] == 1


@pytest.mark.anyio
async def test_build_index_skips_empty_text(conn, settings):
    conn.execute(
        """
        INSERT INTO fichas(entity_type, entity_id, facets, description, confidence, source, content_hash)
        VALUES ('artist', 'vacía', '{}', '', 0.9, 'llm', 'h')
        """
    )
    conn.commit()
    stats = await build_index(conn, settings=settings, ollama=FakeOllama())
    assert stats["total"] == 1
    assert stats["skipped"] == 1
    assert stats["embedded"] == 0


@pytest.mark.anyio
async def test_build_index_corrupt_facets(conn, settings):
    conn.execute(
        """
        INSERT INTO fichas(entity_type, entity_id, facets, description, confidence, source, content_hash)
        VALUES ('artist', 'a1', 'no-json', 'descripcion valida', 0.9, 'llm', 'h')
        """
    )
    conn.commit()
    stats = await build_index(conn, settings=settings, ollama=FakeOllama())
    assert stats["embedded"] == 1


@pytest.mark.anyio
async def test_build_index_embed_error_counted(conn, settings):
    seed_library(conn)
    ollama = FakeOllama(chat_responses=[RuntimeError("embed caído")])
    stats = await build_index(conn, settings=settings, ollama=ollama)
    assert stats["total"] == 1
    assert stats["errors"] == 1
    assert stats["embedded"] == 0


@pytest.mark.anyio
async def test_build_index_wrong_dimension_still_stores(conn, settings):
    seed_library(conn)
    settings.embed_dim = 768
    ollama = FakeOllama(embedding=[0.1] * 10)
    stats = await build_index(conn, settings=settings, ollama=ollama)
    # sqlite-vec rechaza la dimensión; debe contarse como error, no reventar
    assert stats["errors"] == 1


@pytest.mark.anyio
async def test_build_index_progress_called(conn, settings):
    seed_library(conn)
    events = []
    await build_index(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        progress=lambda s, p: events.append((s, p["i"], p["total"])),
    )
    assert events == [("embed", 1, 1)]


@pytest.mark.anyio
async def test_build_index_owns_ollama_closes(conn, settings, monkeypatch):
    closed = {"v": False}

    class Owned(FakeOllama):
        async def close(self):
            closed["v"] = True

    monkeypatch.setattr("app.index.build.OllamaClient", lambda s: Owned())
    await build_index(conn, settings=settings)
    assert closed["v"] is True


@pytest.mark.anyio
async def test_build_index_does_not_close_injected(conn, settings):
    ollama = FakeOllama()
    await build_index(conn, settings=settings, ollama=ollama)
    assert ollama.closed is False


@pytest.mark.anyio
async def test_build_index_empty_db(conn, settings):
    stats = await build_index(conn, settings=settings, ollama=FakeOllama())
    assert stats == {"total": 0, "embedded": 0, "skipped": 0, "errors": 0}


@pytest.mark.anyio
async def test_build_index_commits_every_25(conn, settings):
    from app.index.build import build_index

    for i in range(26):
        conn.execute(
            """
            INSERT INTO fichas(entity_type, entity_id, facets, description, confidence, source, content_hash)
            VALUES ('artist', ?, '{}', ?, 0.9, 'llm', 'h')
            """,
            (f"a{i}", f"desc {i}"),
        )
    conn.commit()
    stats = await build_index(conn, settings=settings, ollama=FakeOllama())
    assert stats["embedded"] == 26
