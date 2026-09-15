from __future__ import annotations

import pytest

from app.enrich.artist import get_ficha
from app.enrich.pipeline import enrich_library
from tests.conftest import FakeOllama, seed_library

ART = {"name": "X", "genres": ["Folk Metal"], "moods": ["Epico"], "description": "d", "confidence": 0.9}
ALB = {
    "artist": "X",
    "album": "Y",
    "year": 2019,
    "is_concept_album": False,
    "concept": "",
    "themes": ["Cerveza"],
    "moods": ["Fiesta"],
    "references": [],
    "energy": 0.8,
    "description": "album d",
    "confidence": 0.85,
}


class FakeLastFm:
    def __init__(self, tags=None):
        self.tags = tags or []
        self.calls = []

    def fetch_for(self, conn, entity_type, key, artist, track=""):
        self.calls.append((entity_type, key, artist, track))
        return self.tags

    def close(self):
        pass


def make_ollama() -> FakeOllama:
    return FakeOllama(chat_responses=[ART, ART, ALB, ALB, ART, ART, ALB, ALB])


@pytest.mark.anyio
async def test_enrich_library_all_entities_inherit(conn, settings):
    seed_library(conn)
    ollama = make_ollama()
    lastfm = FakeLastFm()
    result = await enrich_library(
        conn, settings=settings, ollama=ollama, lastfm=lastfm, force=True
    )
    assert result["artists"] == 2
    assert result["albums"] == 2
    assert result["tracks"] == 4
    track_ficha = get_ficha(conn, "track", "t1")
    assert track_ficha["source"] == "inherited"
    assert "fiesta" in (track_ficha["facets"]["moods"] or [])


@pytest.mark.anyio
async def test_enrich_library_skips_when_not_forced(conn, settings):
    seed_library(conn)
    ollama = make_ollama()
    first = await enrich_library(conn, settings=settings, ollama=ollama, lastfm=FakeLastFm())
    assert first["artists"] == 2
    second = await enrich_library(conn, settings=settings, ollama=ollama, lastfm=FakeLastFm())
    assert second["artists"] == 0
    assert second["skipped"] >= 4


@pytest.mark.anyio
async def test_enrich_library_selection_flags(conn, settings):
    seed_library(conn)
    ollama = make_ollama()
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=ollama,
        lastfm=FakeLastFm(),
        artists=False,
        albums=False,
        tracks=True,
        force=True,
    )
    assert result["artists"] == 0
    assert result["albums"] == 0
    # t1 y t2 heredan de la ficha de álbum al1 preexistente; t3 y t4 no tienen.
    assert result["tracks"] == 2
    assert result["skipped"] == 2
    assert get_ficha(conn, "track", "t1")["source"] == "inherited"
    assert get_ficha(conn, "track", "t3") is None


@pytest.mark.anyio
async def test_enrich_library_limit(conn, settings):
    seed_library(conn)
    ollama = make_ollama()
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=ollama,
        lastfm=FakeLastFm(),
        limit=1,
        force=True,
    )
    assert result["artists"] == 1
    assert result["albums"] == 1
    assert result["tracks"] == 1


@pytest.mark.anyio
async def test_enrich_library_tracks_llm(conn, settings):
    seed_library(conn)
    settings.enrich_tracks_llm = True
    ollama = FakeOllama(
        chat_responses=[
            ART,
            ART,
            ALB,
            ALB,
            {"title": "T", "moods": ["Fiesta"], "themes": ["Cerveza"], "description": "d", "confidence": 0.9},
            {"title": "T", "moods": ["Fiesta"], "themes": ["Cerveza"], "description": "d", "confidence": 0.9},
            {"title": "T", "moods": ["Fiesta"], "themes": ["Cerveza"], "description": "d", "confidence": 0.9},
            {"title": "T", "moods": ["Fiesta"], "themes": ["Cerveza"], "description": "d", "confidence": 0.9},
        ]
    )
    result = await enrich_library(
        conn, settings=settings, ollama=ollama, lastfm=FakeLastFm(), force=True
    )
    assert result["tracks"] == 4
    assert get_ficha(conn, "track", "t1")["source"] == "llm"


@pytest.mark.anyio
async def test_enrich_library_tracks_inherit_disabled(conn, settings):
    seed_library(conn)
    settings.enrich_tracks_inherit = False
    settings.enrich_tracks_llm = False
    ollama = make_ollama()
    result = await enrich_library(
        conn, settings=settings, ollama=ollama, lastfm=FakeLastFm(), force=True
    )
    assert result["tracks"] == 0
    assert get_ficha(conn, "track", "t1") is None


@pytest.mark.anyio
async def test_enrich_library_lastfm_triggered(conn, settings):
    seed_library(conn)
    settings.enable_lastfm = True
    settings.lastfm_api_key = "key"
    ollama = make_ollama()
    lastfm = FakeLastFm(tags=["folk metal"])
    events = []
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=ollama,
        lastfm=lastfm,
        artists=True,
        albums=False,
        tracks=False,
        force=True,
        progress=lambda s, p: events.append(s),
    )
    assert lastfm.calls
    assert "lastfm_artist" in events
    assert result["artists"] == 2


@pytest.mark.anyio
async def test_enrich_library_lastfm_disabled_no_calls(conn, settings):
    seed_library(conn)
    settings.enable_lastfm = False
    lastfm = FakeLastFm()
    await enrich_library(
        conn,
        settings=settings,
        ollama=make_ollama(),
        lastfm=lastfm,
        albums=False,
        tracks=False,
        force=True,
    )
    assert lastfm.calls == []


@pytest.mark.anyio
async def test_enrich_library_creates_run_and_finishes_ok(conn, settings):
    seed_library(conn)
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=make_ollama(),
        lastfm=FakeLastFm(),
        artists=True,
        albums=False,
        tracks=False,
        force=True,
    )
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (result["run_id"],)).fetchone()
    assert run["module"] == "enrich"
    assert run["status"] == "ok"


@pytest.mark.anyio
async def test_enrich_library_error_marks_run(conn, settings, monkeypatch):
    seed_library(conn)

    async def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("app.enrich.pipeline.artist_mod.enrich_artist", boom)
    with pytest.raises(RuntimeError):
        await enrich_library(
            conn,
            settings=settings,
            ollama=make_ollama(),
            lastfm=FakeLastFm(),
            artists=True,
            albums=False,
            tracks=False,
            force=True,
        )
    run = conn.execute(
        "SELECT * FROM runs WHERE module='enrich' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert run["status"] == "error"
    assert "kaboom" in run["stats"]


@pytest.mark.anyio
async def test_enrich_library_closes_owned_ollama(conn, settings, monkeypatch):
    seed_library(conn)
    closed = {"v": False}

    class OwnedOllama(FakeOllama):
        async def close(self):
            closed["v"] = True

    monkeypatch.setattr("app.enrich.pipeline.OllamaClient", lambda s: OwnedOllama())
    await enrich_library(
        conn,
        settings=settings,
        lastfm=FakeLastFm(),
        artists=False,
        albums=False,
        tracks=False,
    )
    assert closed["v"] is True


def test_album_navidrome_id_helpers(conn):
    from app.enrich.pipeline import _album_navidrome_id, _artist_navidrome_id

    seed_library(conn)
    assert _album_navidrome_id(conn, "album:al1") == "al1"
    assert _artist_navidrome_id(conn, "artist:a1") == "a1"
    assert _album_navidrome_id(conn, "album:ghost") == "album:ghost"
    assert _artist_navidrome_id(conn, "artist:ghost") == "artist:ghost"


@pytest.mark.anyio
async def test_enrich_library_track_without_album_or_artist(conn, settings):
    conn.execute(
        "INSERT INTO tracks(id, navidrome_id, title) VALUES ('track:solo', 'solo', 'Solo')"
    )
    conn.commit()
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=FakeLastFm(),
        artists=False,
        albums=False,
        tracks=True,
        force=True,
    )
    assert result["tracks"] == 0
    assert result["skipped"] == 1
