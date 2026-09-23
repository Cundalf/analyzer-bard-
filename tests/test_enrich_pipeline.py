from __future__ import annotations

import pytest
from tests.conftest import FakeOllama, seed_library

from app.enrich.artist import get_ficha
from app.enrich.pipeline import enrich_library

ART = {
    "name": "X",
    "genres": ["Folk Metal"],
    "moods": ["Epico"],
    "description": "d",
    "confidence": 0.9,
}
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
    result = await enrich_library(conn, settings=settings, ollama=ollama, lastfm=lastfm, force=True)
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
            {
                "title": "T",
                "moods": ["Fiesta"],
                "themes": ["Cerveza"],
                "description": "d",
                "confidence": 0.9,
            },
            {
                "title": "T",
                "moods": ["Fiesta"],
                "themes": ["Cerveza"],
                "description": "d",
                "confidence": 0.9,
            },
            {
                "title": "T",
                "moods": ["Fiesta"],
                "themes": ["Cerveza"],
                "description": "d",
                "confidence": 0.9,
            },
            {
                "title": "T",
                "moods": ["Fiesta"],
                "themes": ["Cerveza"],
                "description": "d",
                "confidence": 0.9,
            },
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


@pytest.mark.anyio
async def test_pipeline_inherit_skip_when_not_stale(conn, settings):
    seed_library(conn)
    from app.enrich.artist import Ficha, save_ficha

    save_ficha(
        conn,
        Ficha(
            "track",
            "t1",
            {"moods": []},
            "d",
            0.9,
            "inherited",
            __import__("app.enrich.artist", fromlist=["_hash_inputs"])._hash_inputs({"row": "t1"}),
        ),
    )
    from tests.conftest import FakeOllama as FO

    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FO(),
        lastfm=type("L", (), {"fetch_for": lambda *a, **k: [], "close": lambda self: None})(),
        artists=False,
        albums=False,
        tracks=True,
        force=False,
    )
    assert result["skipped"] >= 1


@pytest.mark.anyio
async def test_pipeline_track_without_fichas_skipped(conn, settings):
    conn.execute(
        "INSERT INTO tracks(id, navidrome_id, title) VALUES ('track:solo', 'solo', 'Solo')"
    )
    conn.commit()
    from tests.conftest import FakeOllama as FO

    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FO(),
        lastfm=type("L", (), {"fetch_for": lambda *a, **k: [], "close": lambda self: None})(),
        artists=False,
        albums=False,
        tracks=True,
        force=True,
    )
    assert result["skipped"] == 1


@pytest.mark.anyio
async def test_pipeline_track_llm_failure_counts_skipped(conn, settings):
    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    settings.enrich_tracks_llm = True
    ollama = FakeOllama(
        chat_responses=[
            {"name": "X", "genres": [], "moods": [], "description": "d", "confidence": 0.9},
            {"name": "Y", "genres": [], "moods": [], "description": "d", "confidence": 0.9},
            {
                "artist": "X",
                "album": "Y",
                "themes": [],
                "moods": [],
                "description": "d",
                "confidence": 0.9,
            },
            {
                "artist": "X",
                "album": "Z",
                "themes": [],
                "moods": [],
                "description": "d",
                "confidence": 0.9,
            },
            RuntimeError("track falla"),
            RuntimeError("track falla"),
            RuntimeError("track falla"),
            RuntimeError("track falla"),
        ]
    )
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=ollama,
        lastfm=type("L", (), {"fetch_for": lambda *a, **k: [], "close": lambda self: None})(),
        force=True,
    )
    assert result["skipped"] >= 4


@pytest.mark.anyio
async def test_pipeline_lastfm_no_tags_no_event(conn, settings):
    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    settings.enable_lastfm = True
    settings.lastfm_api_key = "k"
    events = []

    class EmptyLastFm:
        def fetch_for(self, *args, **kwargs):
            return []

        def close(self):
            pass

    await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(
            chat_responses=[
                {"name": "X", "genres": [], "moods": [], "description": "d", "confidence": 1},
                {"name": "Y", "genres": [], "moods": [], "description": "d", "confidence": 1},
            ]
        ),
        lastfm=EmptyLastFm(),
        artists=True,
        albums=False,
        tracks=False,
        force=True,
        progress=lambda s, p: events.append(s),
    )
    assert "lastfm_artist" not in events
    assert "artist" in events


@pytest.mark.anyio
async def test_pipeline_detection_no_signal(conn, settings, monkeypatch):
    from tests.conftest import FakeOllama

    from app.enrich.artist import Ficha, save_ficha
    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    save_ficha(conn, Ficha("track", "t1", {"moods": []}, "d", 0.9, "inherited", "h"))

    class Client:
        def get_lyrics_by_song_id(self, track_id):
            return "texto suficientemente largo pero sin idioma detectable xxxxxx"

    monkeypatch.setattr("app.enrich.lyrics.detect_language", lambda text: None)
    settings.detect_language = True
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=type("L", (), {"fetch_for": lambda *a, **k: [], "close": lambda s: None})(),
        client=Client(),
        artists=False,
        albums=False,
        tracks=True,
        force=True,
    )
    assert result["languages"] == 0


def test_merge_lastfm_missing_ficha_is_noop(conn):
    from app.enrich.pipeline import _merge_lastfm_into_ficha

    _merge_lastfm_into_ficha(conn, "artist", "no-existe", ["tag"])
    assert conn.execute("SELECT COUNT(*) AS n FROM fichas").fetchone()["n"] == 0


def test_merge_lastfm_into_existing_ficha(conn):
    from app.enrich.artist import Ficha, get_ficha, save_ficha
    from app.enrich.pipeline import _merge_lastfm_into_ficha

    save_ficha(conn, Ficha("artist", "a1", {"genres": []}, "d", 0.9, "llm", "h"))
    _merge_lastfm_into_ficha(conn, "artist", "a1", ["folk metal", "drinking song"])
    ficha = get_ficha(conn, "artist", "a1")
    assert ficha["facets"]["lastfm_tags"] == ["folk metal", "drinking song"]
    assert ficha["facets"]["genres"] == ["folk metal", "drinking song"]
    # no debe perder descripción/confianza
    assert ficha["description"] == "d"
    assert ficha["confidence"] == 0.9


@pytest.mark.anyio
async def test_pipeline_inherits_language_and_country(conn, settings):
    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    conn.execute("UPDATE albums SET genre = ''")
    conn.commit()
    # Orden real del pipeline: artistas y álbumes ORDER BY name.
    # Artistas: Blind Guardian (a2), Wind Rose (a1).
    # Álbumes: Nightfall in Middle-Earth (al2), Wintersaga (al1).
    ollama = FakeOllama(
        chat_responses=[
            {
                "name": "Blind Guardian",
                "genres": [],
                "moods": [],
                "description": "d",
                "confidence": 0.8,
            },
            {
                "name": "Wind Rose",
                "genres": ["folk metal"],
                "moods": ["epico"],
                "language": "Italiano",
                "country": "Italia",
                "energy": "alta",
                "description": "d",
                "confidence": 0.9,
            },
            {
                "artist": "Blind Guardian",
                "album": "Nightfall",
                "themes": [],
                "moods": [],
                "description": "d",
                "confidence": 0.8,
            },
            {
                "artist": "Wind Rose",
                "album": "Wintersaga",
                "themes": ["enanos"],
                "moods": ["fiesta"],
                "language": "en",
                "decades": ["2010s"],
                "description": "d",
                "confidence": 0.9,
            },
        ]
    )
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=ollama,
        lastfm=type("L", (), {"fetch_for": lambda *a, **k: [], "close": lambda self: None})(),
        force=True,
    )
    assert result["tracks"] == 4
    from app.enrich.artist import get_ficha

    track = get_ficha(conn, "track", "t1")
    assert track["facets"]["language"] == "en"
    assert track["facets"]["country"] == "IT"
    assert track["facets"]["energy"] == 0.75
    other = get_ficha(conn, "track", "t3")
    assert other is not None
    assert other["facets"].get("country") in (None, "DE")
