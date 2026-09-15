from __future__ import annotations

import pytest

from app.config import Settings
from tests.conftest import FakeOllama, seed_library


# ------------------------------------------------------------ retrieval branches

def test_hydrate_artist_only_themes_missing(conn):
    """El álbum aporta moods, pero no themes: el artista completa themes."""
    from app.enrich.artist import Ficha, save_ficha
    from app.agent import retrieval

    seed_library(conn)
    conn.execute("DELETE FROM fichas WHERE entity_type='album'")
    save_ficha(
        conn,
        Ficha(
            "artist",
            "a1",
            {"moods": ["del artista"], "lyrical_themes": ["temas del artista"]},
            "",
            0.9,
            "llm",
            "h",
        ),
    )
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["moods"] == ["del artista"]
    assert target["themes"] == ["temas del artista"]


def test_hydrate_artist_moods_present_themes_from_references(conn):
    from app.enrich.artist import Ficha, save_ficha
    from app.agent import retrieval

    seed_library(conn)
    conn.execute("DELETE FROM fichas WHERE entity_type='album'")
    save_ficha(
        conn,
        Ficha("artist", "a1", {"moods": ["x"], "references": ["ref"]}, "", 0.9, "llm", "h"),
    )
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["moods"] == ["x"]
    assert target["themes"] == ["ref"]


# ------------------------------------------------------------ runner branches

def test_name_from_trace_skips_empty_name():
    from app.agent.runner import _name_from_trace

    result = {
        "trace": [
            {"tool": "create_playlist", "arguments": {"name": ""}},
            {"tool": "create_playlist", "arguments": {"name": "Segunda"}},
        ]
    }
    assert _name_from_trace(result) == "Segunda"


def test_ids_from_trace_ignores_non_create_tools_and_empty(conn):
    from app.agent.runner import _ids_from_trace

    assert _ids_from_trace({"trace": [{"tool": "otra", "arguments": {}}]}, conn) == []
    assert (
        _ids_from_trace(
            {"trace": [{"tool": "create_playlist", "arguments": {"track_ids": []}}]},
            conn,
        )
        == []
    )


# ------------------------------------------------------------ db branches

def test_ensure_vec_table_recreates_when_table_missing(tmp_path, monkeypatch):
    from app import db as db_mod
    from app.config import get_settings

    monkeypatch.setenv("EMBED_DIM", "768")
    get_settings.cache_clear()
    conn = db_mod.connect(tmp_path / "vec.db")
    db_mod.init_db(conn)
    # borra la tabla pero deja el meta: debe recrearla
    conn.execute("DROP TABLE vec_fichas")
    conn.commit()
    db_mod._ensure_vec_table(conn, 768)
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_fichas'"
    ).fetchone()
    assert exists is not None
    conn.close()
    get_settings.cache_clear()


# ------------------------------------------------------------ album enrich branches

@pytest.mark.anyio
async def test_enrich_album_artist_exists_without_ficha(conn, settings):
    from app.enrich.album import enrich_album
    from app.enrich.canonicalize import Canonicalizer

    seed_library(conn)
    ollama = FakeOllama(
        chat_responses=[
            {
                "artist": "W",
                "album": "A",
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
    # sin ficha de artista, usa los géneros de los álbumes del artista
    prompt = ollama.calls[0]["messages"][1]["content"]
    assert "folk metal" in prompt


# ------------------------------------------------------------ pipeline branches

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


# ------------------------------------------------------------ janitor runner branches

def test_run_janitor_no_ingest_without_run_id(tmp_path, monkeypatch):
    from app.janitor.runner import run_janitor

    music = tmp_path / "music"
    music.mkdir()
    beetsdir = tmp_path / "config"
    beetsdir.mkdir()
    s = Settings(
        data_dir=str(tmp_path / "data"),
        music_dir=str(music),
        beetsdir=str(beetsdir),
        janitor_enabled=True,
    )
    s.ensure_dirs()
    from app.janitor.report import TagDiff, write_jsonl

    def fake_beets(*args, **kwargs):
        logpath = kwargs["logpath"]
        logpath.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(
            type("R", (), {"items": [TagDiff(file="/a", old_tags={}, new_tags={})]})(),
            logpath,
        )
        return {"cmd": ["beet"], "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("app.janitor.runner.run_beets", fake_beets)
    result = run_janitor(
        settings=s, conn=None, pretend=False, skip_wav=True, do_rescan=False
    )
    assert len(result.diffs) == 1
    assert result.run_id is None


# ------------------------------------------------------------ wav2flac branches

def test_convert_all_explicit_backup_and_delete(tmp_path):
    from app.janitor.wav2flac import convert_all, scan_wavs

    music = tmp_path / "music"
    music.mkdir()
    (music / "a.wav").write_bytes(b"RIFF")
    script = tmp_path / "ff"
    script.write_text(
        "#!/bin/sh\nout=''\nfor a in \"$@\"; do out=\"$a\"; done\nprintf 'fLaC' > \"$out\"\n"
    )
    script.chmod(0o755)
    scan = scan_wavs(music)
    settings = Settings(data_dir=str(tmp_path / "data"), ffmpeg_bin=str(script))
    backup = tmp_path / "explicit-backup"
    convert_all(
        scan,
        settings=settings,
        delete_original=False,
        backup_dir=backup,
        ffmpeg_bin=str(script),
    )
    assert scan.findings[0].backed_up is True
    assert (backup / "a.wav").exists()


# ------------------------------------------------------------ main branches

def test_playlists_generate_client_close_on_success(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import db as db_mod
    from app import main as main_mod
    from app.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    db_mod._conn = None

    closed = {"v": False}

    class ClosingSubsonic:
        def __init__(self, settings):
            pass

        def close(self):
            closed["v"] = True

    async def fake_generate(conn, prompt, **kwargs):
        return {
            "playlist_name": "P",
            "track_ids": [],
            "reasoning": "",
            "tool_calls": 0,
            "trace": [],
            "created_playlists": [],
            "validation": [],
            "saved": False,
            "mode": "rerank",
        }

    monkeypatch.setattr("app.subsonic.SubsonicClient", ClosingSubsonic)
    monkeypatch.setattr(main_mod, "generate_playlist", fake_generate)
    with TestClient(main_mod.app) as web_client:
        response = web_client.post("/playlists/generate", data={"prompt": "x"})
    assert response.status_code == 200
    assert closed["v"] is True
    if db_mod._conn is not None:
        db_mod._conn.close()
        db_mod._conn = None
    get_settings.cache_clear()
