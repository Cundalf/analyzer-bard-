from __future__ import annotations

import pytest

from app.enrich.album import enrich_album
from app.enrich.artist import Ficha, enrich_artist, get_ficha, is_stale, save_ficha
from app.enrich.canonicalize import Canonicalizer, content_hash
from app.enrich.track import enrich_track_llm, inherit_track_ficha
from tests.conftest import FakeOllama, seed_library

ARTIST_FACETS = {
    "name": "Wind Rose",
    "genres": ["Folk Metal"],
    "subgenres": ["Celtic"],
    "country": "Italy",
    "era": "2010s",
    "language": "en",
    "instrumentation": ["bagpipes"],
    "vocal_style": "harsh",
    "lyrical_themes": ["dwarves"],
    "moods": ["Epic", "fiesta"],
    "references": [],
    "for_fans_of": ["Alestorm"],
    "energy": 0.9,
    "description": "Banda italiana de folk metal festivo.",
    "confidence": 0.95,
}


# ------------------------------------------------------------ save/get/stale

def test_save_and_get_ficha(conn):
    ficha = Ficha(
        entity_type="artist",
        entity_id="a1",
        facets={"moods": ["epico"]},
        description="desc",
        confidence=0.9,
        source="llm",
        content_hash="abc",
    )
    save_ficha(conn, ficha)
    got = get_ficha(conn, "artist", "a1")
    assert got["facets"] == {"moods": ["epico"]}
    assert got["description"] == "desc"
    assert got["confidence"] == 0.9


def test_save_ficha_upserts(conn):
    for desc in ("uno", "dos"):
        save_ficha(
            conn,
            Ficha("artist", "a1", {"moods": []}, desc, 0.5, "llm", "h"),
        )
    assert get_ficha(conn, "artist", "a1")["description"] == "dos"
    assert conn.execute("SELECT COUNT(*) AS n FROM fichas").fetchone()["n"] == 1


def test_save_ficha_updates_fts(conn):
    from app.db import fts_search

    save_ficha(
        conn,
        Ficha("artist", "a1", {"genres": ["folk metal"]}, "dwarves beer", 0.5, "llm", "h"),
    )
    assert fts_search(conn, "dwarves")
    save_ficha(
        conn,
        Ficha("artist", "a1", {"genres": ["pop"]}, "synth dreams", 0.5, "llm", "h2"),
    )
    assert not fts_search(conn, "dwarves")
    assert fts_search(conn, "synth")


def test_get_ficha_missing(conn):
    assert get_ficha(conn, "artist", "nope") is None


def test_get_ficha_corrupt_facets(conn):
    conn.execute(
        "INSERT INTO fichas(entity_type, entity_id, facets, description, confidence, source, content_hash) "
        "VALUES ('artist', 'a1', 'no-json', 'd', 0.5, 'llm', 'h')"
    )
    assert get_ficha(conn, "artist", "a1")["facets"] == {}


def test_is_stale_missing_and_equal(conn):
    payload = {"name": "X"}
    assert is_stale(conn, "artist", "a1", payload) is True
    save_ficha(
        conn,
        Ficha("artist", "a1", {}, "d", 1.0, "llm", content_hash(payload)),
    )
    assert is_stale(conn, "artist", "a1", payload) is False
    assert is_stale(conn, "artist", "a1", {"name": "Y"}) is True


# ------------------------------------------------------------ enrich_artist

@pytest.mark.anyio
async def test_enrich_artist_happy(conn, settings):
    seed_library(conn)
    ollama = FakeOllama(chat_responses=[ARTIST_FACETS])
    canon = Canonicalizer.from_db(conn)
    row = dict(conn.execute("SELECT * FROM artists WHERE navidrome_id='a1'").fetchone())
    result = await enrich_artist(conn, ollama, canon, row, force=True)
    assert result["entity_id"] == "a1"
    ficha = get_ficha(conn, "artist", "a1")
    assert ficha["facets"]["genres"] == ["folk metal"]
    assert ficha["facets"]["moods"] == ["epico", "fiesta"]
    assert ficha["description"] == "Banda italiana de folk metal festivo."
    assert ficha["confidence"] == 0.95
    assert "description" not in ficha["facets"]
    assert "confidence" not in ficha["facets"]


@pytest.mark.anyio
async def test_enrich_artist_skips_fresh(conn, settings):
    seed_library(conn)
    ollama = FakeOllama(chat_responses=[ARTIST_FACETS])
    canon = Canonicalizer.from_db(conn)
    row = dict(conn.execute("SELECT * FROM artists WHERE navidrome_id='a1'").fetchone())
    assert await enrich_artist(conn, ollama, canon, row) is not None
    assert await enrich_artist(conn, ollama, canon, row) is None
    assert len(ollama.calls) == 1


@pytest.mark.anyio
async def test_enrich_artist_llm_failure_returns_none(conn, settings):
    seed_library(conn)
    ollama = FakeOllama(chat_responses=[RuntimeError("llm caído")])
    canon = Canonicalizer.from_db(conn)
    row = dict(conn.execute("SELECT * FROM artists WHERE navidrome_id='a1'").fetchone())
    assert await enrich_artist(conn, ollama, canon, row, force=True) is None
    assert get_ficha(conn, "artist", "a1") is None


@pytest.mark.anyio
async def test_enrich_artist_missing_optional_fields(conn, settings):
    seed_library(conn)
    minimal = {"name": "X", "genres": [], "moods": [], "description": "", "confidence": 0}
    ollama = FakeOllama(chat_responses=[minimal])
    canon = Canonicalizer.from_db(conn)
    row = dict(conn.execute("SELECT * FROM artists WHERE navidrome_id='a1'").fetchone())
    await enrich_artist(conn, ollama, canon, row, force=True)
    ficha = get_ficha(conn, "artist", "a1")
    assert ficha["confidence"] == 0.0
    assert ficha["description"] == ""


@pytest.mark.anyio
async def test_enrich_artist_confidence_string(conn, settings):
    seed_library(conn)
    payload = dict(ARTIST_FACETS, confidence="0.5", description=None)
    ollama = FakeOllama(chat_responses=[payload])
    canon = Canonicalizer.from_db(conn)
    row = dict(conn.execute("SELECT * FROM artists WHERE navidrome_id='a1'").fetchone())
    await enrich_artist(conn, ollama, canon, row, force=True)
    ficha = get_ficha(conn, "artist", "a1")
    assert ficha["confidence"] == 0.5
    assert ficha["description"] == ""


@pytest.mark.anyio
async def test_enrich_artist_row_without_navidrome_id(conn, settings):
    seed_library(conn)
    ollama = FakeOllama(chat_responses=[ARTIST_FACETS])
    canon = Canonicalizer.from_db(conn)
    row = {"id": "artist:local", "name": "Local", "navidrome_id": None}
    result = await enrich_artist(conn, ollama, canon, row, force=True)
    assert result["entity_id"] == "artist:local"


# ------------------------------------------------------------ enrich_album

ALBUM_FACETS = {
    "artist": "Wind Rose",
    "album": "Wintersaga",
    "year": 2019,
    "is_concept_album": True,
    "concept": "Enanos y cerveza",
    "themes": ["Battle"],
    "moods": ["Epic"],
    "references": ["Tolkien"],
    "energy": 0.9,
    "description": "Álbum conceptual de enanos.",
    "confidence": 0.9,
}


@pytest.mark.anyio
async def test_enrich_album_happy(conn, settings):
    seed_library(conn)
    save_ficha(
        conn,
        Ficha("artist", "a1", {"genres": ["folk metal"]}, "d", 0.9, "llm", "h"),
    )
    ollama = FakeOllama(chat_responses=[ALBUM_FACETS])
    canon = Canonicalizer.from_db(conn)
    row = dict(conn.execute("SELECT * FROM albums WHERE navidrome_id='al1'").fetchone())
    result = await enrich_album(conn, ollama, canon, row, force=True)
    assert result["entity_id"] == "al1"
    ficha = get_ficha(conn, "album", "al1")
    assert ficha["facets"]["concept"] == "Enanos y cerveza"
    assert ficha["facets"]["moods"] == ["epico"]
    assert ficha["facets"]["references"] == ["señor de los anillos"]


@pytest.mark.anyio
async def test_enrich_album_without_artist_row(conn, settings):
    seed_library(conn)
    ollama = FakeOllama(chat_responses=[ALBUM_FACETS])
    canon = Canonicalizer.from_db(conn)
    row = dict(conn.execute("SELECT * FROM albums WHERE navidrome_id='al1'").fetchone())
    row["artist_id"] = None
    result = await enrich_album(conn, ollama, canon, row, force=True)
    assert result is not None


@pytest.mark.anyio
async def test_enrich_album_llm_failure(conn, settings):
    seed_library(conn)
    ollama = FakeOllama(chat_responses=[RuntimeError("boom")])
    canon = Canonicalizer.from_db(conn)
    row = dict(conn.execute("SELECT * FROM albums WHERE navidrome_id='al1'").fetchone())
    assert await enrich_album(conn, ollama, canon, row, force=True) is None


@pytest.mark.anyio
async def test_enrich_album_skips_fresh(conn, settings):
    seed_library(conn)
    ollama = FakeOllama(chat_responses=[ALBUM_FACETS, ALBUM_FACETS])
    canon = Canonicalizer.from_db(conn)
    row = dict(conn.execute("SELECT * FROM albums WHERE navidrome_id='al1'").fetchone())
    assert await enrich_album(conn, ollama, canon, row) is not None
    assert await enrich_album(conn, ollama, canon, row) is None


@pytest.mark.anyio
async def test_enrich_album_prompt_includes_tracks(conn, settings):
    seed_library(conn)
    ollama = FakeOllama(chat_responses=[ALBUM_FACETS])
    canon = Canonicalizer.from_db(conn)
    row = dict(conn.execute("SELECT * FROM albums WHERE navidrome_id='al1'").fetchone())
    await enrich_album(conn, ollama, canon, row, force=True)
    user_msg = ollama.calls[0]["messages"][1]["content"]
    assert "Drunken Dwarves" in user_msg
    assert "Wintersaga" in user_msg


# ------------------------------------------------------------ inherit_track

def test_inherit_track_from_album_and_artist(conn):
    album_ficha = {
        "entity_id": "al1",
        "facets": {"moods": ["fiesta"], "themes": ["enanos"], "energy": 0.8},
        "description": "Álbum festivo.",
        "confidence": 0.9,
        "content_hash": "ha",
    }
    artist_ficha = {
        "entity_id": "a1",
        "facets": {"moods": ["epico"], "references": ["tolkien"], "energy": 0.5},
        "description": "Artista.",
        "confidence": 0.8,
        "content_hash": "hb",
    }
    track = {"id": "track:t1", "navidrome_id": "t1", "title": "Drunken Dwarves"}
    ficha = inherit_track_ficha(conn, track, album_ficha, artist_ficha)
    assert ficha.entity_type == "track"
    assert ficha.entity_id == "t1"
    assert ficha.facets["moods"] == ["fiesta"]
    assert ficha.facets["themes"] == ["enanos"]
    assert ficha.facets["energy"] == 0.8
    assert "Álbum festivo." in ficha.description
    assert ficha.source == "inherited"
    assert ficha.confidence == 0.8


def test_inherit_track_only_artist(conn):
    artist_ficha = {
        "entity_id": "a1",
        "facets": {"moods": ["epico"], "references": ["tolkien"]},
        "description": "",
        "confidence": 0.7,
        "content_hash": "hb",
    }
    track = {"id": "track:t1", "navidrome_id": "t1", "title": "Song"}
    ficha = inherit_track_ficha(conn, track, None, artist_ficha)
    assert ficha.facets["moods"] == ["epico"]
    assert ficha.facets["themes"] == ["tolkien"]
    assert ficha.confidence == 0.7


def test_inherit_track_no_fichas():
    assert inherit_track_ficha(None, {"title": "x"}, None, None) is None


def test_inherit_track_detects_ballad_and_instrumental():
    album_ficha = {"entity_id": "al1", "facets": {}, "description": "", "confidence": 0.5, "content_hash": "h"}
    for title, expected_instrumental in [
        ("Instrumental Intro", True),
        ("Prelude", True),
        ("Interlude", True),
        ("Outro", True),
        ("Normal Song", False),
    ]:
        ficha = inherit_track_ficha(
            None, {"navidrome_id": "t", "title": title}, album_ficha, None
        )
        assert ficha.facets["is_instrumental"] is expected_instrumental
    for title in ("The Ballad", "Balada Triste", "Lament", "Elegy"):
        ficha = inherit_track_ficha(
            None, {"navidrome_id": "t", "title": title}, album_ficha, None
        )
        assert ficha.facets["is_ballad"] is True


def test_inherit_track_without_navidrome_id():
    album_ficha = {"entity_id": "al1", "facets": {}, "description": "", "confidence": 0.5, "content_hash": "h"}
    ficha = inherit_track_ficha(
        None, {"id": "track:x", "navidrome_id": None, "title": "T"}, album_ficha, None
    )
    assert ficha.entity_id == "track:x"


def test_inherit_track_description_truncated():
    long_desc = "x" * 1000
    album_ficha = {
        "entity_id": "al1",
        "facets": {},
        "description": long_desc,
        "confidence": 0.5,
        "content_hash": "h",
    }
    ficha = inherit_track_ficha(
        None, {"navidrome_id": "t", "title": "T"}, album_ficha, None
    )
    assert len(ficha.description) <= 400


# ------------------------------------------------------------ enrich_track_llm

TRACK_FACETS = {
    "title": "Drunken Dwarves",
    "moods": ["Fiesta"],
    "themes": ["Cerveza"],
    "energy": 0.9,
    "is_ballad": False,
    "is_instrumental": False,
    "description": "Himno para beber.",
    "confidence": 0.85,
}


@pytest.mark.anyio
async def test_enrich_track_llm_happy(conn, settings):
    seed_library(conn)
    ollama = FakeOllama(chat_responses=[TRACK_FACETS])
    canon = Canonicalizer.from_db(conn)
    row = dict(
        conn.execute(
            """
            SELECT t.*, a.name AS album_name FROM tracks t
            LEFT JOIN albums a ON a.id = t.album_id
            WHERE t.navidrome_id = 't1'
            """
        ).fetchone()
    )
    result = await enrich_track_llm(conn, ollama, canon, row, None, None, force=True)
    assert result["entity_id"] == "t1"
    ficha = get_ficha(conn, "track", "t1")
    assert ficha["facets"]["moods"] == ["fiesta"]


@pytest.mark.anyio
async def test_enrich_track_llm_skips_fresh(conn, settings):
    seed_library(conn)
    ollama = FakeOllama(chat_responses=[TRACK_FACETS, TRACK_FACETS])
    canon = Canonicalizer.from_db(conn)
    row = {"id": "track:t1", "navidrome_id": "t1", "title": "T", "album_id": None, "artist_id": None}
    assert await enrich_track_llm(conn, ollama, canon, row, None, None) is not None
    assert await enrich_track_llm(conn, ollama, canon, row, None, None) is None


@pytest.mark.anyio
async def test_enrich_track_llm_failure(conn, settings):
    seed_library(conn)
    ollama = FakeOllama(chat_responses=[RuntimeError("boom")])
    canon = Canonicalizer.from_db(conn)
    row = {"id": "track:t1", "navidrome_id": "t1", "title": "T", "album_id": None, "artist_id": None}
    assert await enrich_track_llm(conn, ollama, canon, row, None, None, force=True) is None


@pytest.mark.anyio
async def test_enrich_track_llm_uses_artist_and_album_context(conn, settings):
    seed_library(conn)
    ollama = FakeOllama(chat_responses=[TRACK_FACETS])
    canon = Canonicalizer.from_db(conn)
    row = dict(
        conn.execute(
            """
            SELECT t.*, a.name AS album_name FROM tracks t
            LEFT JOIN albums a ON a.id = t.album_id
            WHERE t.navidrome_id = 't1'
            """
        ).fetchone()
    )
    album_ficha = {
        "entity_id": "al1",
        "facets": {"concept": "Enanos", "themes": ["cerveza"], "moods": ["fiesta"]},
    }
    await enrich_track_llm(conn, ollama, canon, row, album_ficha, None, force=True)
    user_msg = ollama.calls[0]["messages"][1]["content"]
    assert "Wind Rose" in user_msg
    assert "Wintersaga" in user_msg
    assert "Enanos" in user_msg
