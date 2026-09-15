from __future__ import annotations

import pytest

from tests.conftest import FakeOllama, seed_library


def test_hydrate_artist_lyrical_themes_short_circuit(conn):
    """lyrical_themes presente: no evalúa references (ramas 93->95)."""
    from app.agent import retrieval
    from app.enrich.artist import Ficha, save_ficha

    seed_library(conn)
    conn.execute("DELETE FROM fichas WHERE entity_type='album'")
    save_ficha(
        conn,
        Ficha(
            "artist",
            "a1",
            {"moods": ["m"], "lyrical_themes": ["letras"], "references": ["ref"]},
            "",
            0.9,
            "llm",
            "h",
        ),
    )
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["themes"] == ["letras"]


def test_hydrate_artist_themes_empty_uses_references(conn):
    """Sin lyrical_themes, cae a references (rama 95->99)."""
    from app.agent import retrieval
    from app.enrich.artist import Ficha, save_ficha

    seed_library(conn)
    conn.execute("DELETE FROM fichas WHERE entity_type='album'")
    save_ficha(
        conn,
        Ficha(
            "artist",
            "a1",
            {"moods": ["m"], "lyrical_themes": [], "references": ["ref"]},
            "",
            0.9,
            "llm",
            "h",
        ),
    )
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["themes"] == ["ref"]


def test_hydrate_track_only_moods_completes_themes(conn):
    """Ficha de track con moods pero sin themes: rama 93->95."""
    from app.agent import retrieval
    from app.enrich.artist import Ficha, save_ficha

    seed_library(conn)
    save_ficha(
        conn,
        Ficha("track", "t1", {"moods": ["solo moods"]}, "d", 0.9, "llm", "h"),
    )
    save_ficha(
        conn,
        Ficha(
            "artist",
            "a1",
            {"moods": ["del artista"], "lyrical_themes": ["letras"]},
            "",
            0.9,
            "llm",
            "h2",
        ),
    )
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["moods"] == ["solo moods"]
    assert target["themes"] == ["letras"]


def test_hydrate_track_only_themes_completes_moods(conn):
    """Ficha de track con themes pero sin moods: rama 95->99."""
    from app.agent import retrieval
    from app.enrich.artist import Ficha, save_ficha

    seed_library(conn)
    save_ficha(
        conn,
        Ficha("track", "t1", {"themes": ["solo themes"]}, "d", 0.9, "llm", "h"),
    )
    save_ficha(
        conn,
        Ficha("artist", "a1", {"moods": ["del artista"]}, "", 0.9, "llm", "h2"),
    )
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["moods"] == ["del artista"]
    assert target["themes"] == ["solo themes"]


def test_hydrate_themes_present_skips_artist_fallback(conn):
    """El álbum aporta moods y themes: no consulta al artista (rama 93->99)."""
    from app.agent import retrieval

    seed_library(conn)
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["moods"] == ["fiesta", "epico"]
    assert target["themes"] == ["cerveza", "enanos"]


def test_name_from_trace_iterates_after_empty_name():
    from app.agent.runner import _name_from_trace

    result = {
        "trace": [
            {"tool": "otra", "arguments": {"name": "ignorada"}},
            {"tool": "create_playlist", "arguments": {"name": ""}},
        ]
    }
    assert _name_from_trace(result) == ""


def test_ids_from_trace_iterates_after_non_match(conn):
    from app.agent.runner import _ids_from_trace

    seed_library(conn)
    result = {
        "trace": [
            {"tool": "create_playlist", "arguments": {"track_ids": ["t1"]}},
            {"tool": "list_artists", "arguments": {}},
        ]
    }
    assert _ids_from_trace(result, conn) == ["t1"]


@pytest.mark.anyio
async def test_enrich_album_artist_id_without_row(conn, settings):
    from app.enrich.album import enrich_album
    from app.enrich.canonicalize import Canonicalizer

    seed_library(conn)
    conn.execute("DELETE FROM artists WHERE id = 'artist:a1'")
    conn.commit()
    ollama = FakeOllama(
        chat_responses=[
            {
                "artist": "X",
                "album": "Y",
                "themes": [],
                "moods": [],
                "description": "d",
                "confidence": 0.9,
            }
        ]
    )
    row = dict(conn.execute("SELECT * FROM albums WHERE navidrome_id='al1'").fetchone())
    result = await enrich_album(
        conn, ollama, Canonicalizer.from_db(conn), row, force=True
    )
    assert result is not None
    prompt = ollama.calls[0]["messages"][1]["content"]
    assert "Artista: (desconocido)" in prompt
    # sin ficha de artista, usa los géneros de los álbumes
    assert "Géneros del artista: folk metal" in prompt


def test_playlists_generate_client_constructor_fails(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import db as db_mod
    from app import main as main_mod
    from app.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    db_mod._conn = None

    class BrokenSubsonic:
        def __init__(self, settings):
            raise RuntimeError("no se pudo crear el cliente")

    monkeypatch.setattr("app.subsonic.SubsonicClient", BrokenSubsonic)
    with TestClient(main_mod.app) as web_client:
        response = web_client.post("/playlists/generate", data={"prompt": "x"})
    assert response.status_code == 200
    assert "no se pudo crear el cliente" in response.text
    if db_mod._conn is not None:
        db_mod._conn.close()
        db_mod._conn = None
    get_settings.cache_clear()
