from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from app.config import Settings
from tests.conftest import FakeOllama, FakeSubsonic, seed_library

PLUGIN = Path(__file__).resolve().parents[1] / "app" / "janitor" / "bardo.py"


# ------------------------------------------------------------ retrieval 71-72

def test_hydrate_invalid_json_facets(conn):
    seed_library(conn)
    conn.execute("UPDATE fichas SET facets = '{roto' WHERE entity_type='album' AND entity_id='al1'")
    conn.commit()
    from app.agent import retrieval

    results = retrieval.search_by_terms(conn, ["dwarves"])
    assert results[0]["moods"] == []


# ------------------------------------------------------------ runner 260

@pytest.mark.anyio
async def test_generate_playlist_save_without_created(conn, settings, monkeypatch):
    from app.agent.runner import generate_playlist

    seed_library(conn)
    fake_agent_result = {
        "text": "ok",
        "tool_calls": 1,
        "trace": [
            {
                "tool": "create_playlist",
                "arguments": {"name": "P", "track_ids": ["t1"]},
            }
        ],
        "created_playlists": [],
        "validation": [],
    }

    async def fake_agent(*args, **kwargs):
        return fake_agent_result

    monkeypatch.setattr("app.agent.runner.run_agent", fake_agent)
    subsonic = FakeSubsonic()
    result = await generate_playlist(
        conn,
        "dwarves",
        settings=settings,
        ollama=FakeOllama(chat_responses=[{"canonical_terms": [], "expanded_terms": ["dwarves"], "moods": []}, {}]),
        client=subsonic,
        save=True,
        use_agent=True,
    )
    assert result["saved"] is True
    assert subsonic.created == [("P", ["t1"])]


# ------------------------------------------------------------ cli 219-224

def test_main_verbose_reraises(monkeypatch, capsys):
    from app import cli

    def boom(args):
        raise RuntimeError("detalle")

    monkeypatch.setattr(cli, "cmd_health", boom)
    with pytest.raises(RuntimeError):
        cli.main(["-v", "health"])


# ------------------------------------------------------------ pipeline 131

@pytest.mark.anyio
async def test_pipeline_track_llm_failure_counts_skipped(conn, settings):
    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    settings.enrich_tracks_llm = True
    ollama = FakeOllama(
        chat_responses=[
            {"name": "X", "genres": [], "moods": [], "description": "d", "confidence": 0.9},
            {"name": "Y", "genres": [], "moods": [], "description": "d", "confidence": 0.9},
            {"artist": "X", "album": "Y", "themes": [], "moods": [], "description": "d", "confidence": 0.9},
            {"artist": "X", "album": "Z", "themes": [], "moods": [], "description": "d", "confidence": 0.9},
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


# ------------------------------------------------------------ index 76

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


# ------------------------------------------------------------ bardo plugin 37, 61, 121, 138

@pytest.fixture()
def plugin(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("bardo_plugin3", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bardo_plugin3"] = module
    spec.loader.exec_module(module)
    monkeypatch.setenv("BARDO_BEETS_LOG", str(tmp_path / "beets.jsonl"))
    return module.BardoLogPlugin()


def test_plugin_log_path_default(plugin, monkeypatch):
    from beets import config

    from app.janitor import bardo as mod

    monkeypatch.delenv("BARDO_BEETS_LOG", raising=False)
    config["bardo"]["logpath"] = "/tmp/desde-config.jsonl"
    assert mod._log_path() == Path("/tmp/desde-config.jsonl")


def test_plugin_log_path_final_fallback(plugin, monkeypatch):
    from beets import config

    from app.janitor import bardo as mod

    monkeypatch.delenv("BARDO_BEETS_LOG", raising=False)
    config["bardo"]["logpath"] = ""
    assert mod._log_path() == Path("/data/runs/beets.jsonl")


def test_plugin_distance_not_number_not_object():
    from app.janitor import bardo as mod

    class Weird:
        distance = "texto"

    assert mod._distance_of(Weird()) is None


def test_plugin_on_write_unknown_path(plugin):
    plugin.on_write(item={}, path=b"/nuevo.flac", tags={})


def test_plugin_emit_twice_no_duplicate(plugin, tmp_path):
    plugin._before["/a"] = {"x": 1}
    plugin._emit("/a", {"x": 2})
    plugin._emit("/a", {"x": 3})
    lines = (tmp_path / "beets.jsonl").read_text().splitlines()
    assert len(lines) == 1


# ------------------------------------------------------------ janitor runner gaps

def test_bool_opt_helper(tmp_path):
    from app.janitor.runner import _bool_opt

    config = {"a": {"b": True}}
    assert _bool_opt(config, "a.b", False) is True
    assert _bool_opt(config, "a.c", "def") == "def"
    assert _bool_opt(config, "x.y.z", 1) == 1
    assert _bool_opt({}, "a", 2) == 2


def test_run_janitor_converts_wav_with_settings_bin(tmp_path, monkeypatch):
    from app.db import connect, init_db
    from app.janitor.runner import run_janitor

    music = tmp_path / "music"
    music.mkdir()
    (music / "a.wav").write_bytes(b"RIFF")
    beetsdir = tmp_path / "config"
    beetsdir.mkdir()
    script = tmp_path / "myffmpeg"
    script.write_text("#!/bin/sh\nout=''\nfor a in \"$@\"; do out=\"$a\"; done\nprintf 'fLaC' > \"$out\"\n")
    script.chmod(0o755)
    s = Settings(
        data_dir=str(tmp_path / "data"),
        music_dir=str(music),
        beetsdir=str(beetsdir),
        janitor_enabled=True,
        ffmpeg_bin=str(script),
    )
    s.ensure_dirs()
    monkeypatch.setattr("app.janitor.runner.run_beets", lambda *a, **k: {
        "cmd": ["beet"], "returncode": 0, "stdout": "", "stderr": "",
    })
    conn = connect(tmp_path / "data" / "bardo.db")
    init_db(conn)
    result = run_janitor(settings=s, conn=conn, do_rescan=False)
    assert result.wav["converted"] == 1
    conn.close()


def test_run_janitor_rescan_progress(tmp_path, monkeypatch):
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
    monkeypatch.setattr("app.janitor.runner.run_beets", lambda *a, **k: {
        "cmd": ["beet"], "returncode": 0, "stdout": "", "stderr": "",
    })
    events = []
    run_janitor(
        settings=s,
        client=FakeSubsonic(),
        do_rescan=True,
        progress=lambda stage, payload: events.append(stage),
    )
    assert "rescan" in events


# ------------------------------------------------------------ wav2flac gaps

def test_scan_stat_error_uses_zero(tmp_path, monkeypatch):
    from app.janitor import wav2flac

    (tmp_path / "a.wav").write_bytes(b"RIFF")
    real_stat = Path.stat

    def broken_stat(self, *args, **kwargs):
        if self.name == "a.wav":
            raise OSError("sin permisos")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr("pathlib.Path.stat", broken_stat)
    result = wav2flac.scan_wavs(tmp_path)
    assert result.findings[0].size == 0


def test_find_ffmpeg_binary_name_in_path(tmp_path, monkeypatch):
    from app.janitor import wav2flac

    fake = tmp_path / "customtool"
    fake.write_text("#!/bin/sh")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert wav2flac.find_ffmpeg("customtool") == str(fake)


# ------------------------------------------------------------ main gaps

def test_dashboard_ollama_exception(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import db as db_mod
    from app import main as main_mod
    from app.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    db_mod._conn = None

    async def boom(self):
        raise RuntimeError("ollama caído")

    monkeypatch.setattr("app.ollama.OllamaClient.ping", boom)
    with TestClient(main_mod.app) as web_client:
        response = web_client.get("/")
    assert response.status_code == 200
    db_mod._conn = None
    get_settings.cache_clear()


def test_api_health_subsonic_exception(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import db as db_mod
    from app import main as main_mod
    from app.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    db_mod._conn = None

    monkeypatch.setattr(
        "app.subsonic.SubsonicClient",
        lambda s: (_ for _ in ()).throw(RuntimeError("x")),
    )
    with TestClient(main_mod.app) as web_client:
        body = web_client.get("/api/health").json()
    assert body["subsonic_ok"] is False
    db_mod._conn = None
    get_settings.cache_clear()


# ------------------------------------------------------------ ollama gaps

def test_extract_json_trailing_comma_fence(settings):
    from app.ollama import extract_json

    assert extract_json('```json\n{"a": 1,}\n```') == {"a": 1}


def test_extract_json_only_whitespace_fence(settings):
    from app.ollama import extract_json

    with pytest.raises(ValueError):
        extract_json("```json\n   \n```")


def test_extract_json_escaped_quotes(settings):
    from app.ollama import extract_json

    text = '{"texto": "dice \\"hola\\" y {llaves}"}'
    assert extract_json(text)["texto"] == 'dice "hola" y {llaves}'


# ------------------------------------------------------------ subsonic gaps

def test_sync_track_without_album_id(settings, conn):
    import httpx

    from app.subsonic import SubsonicClient, sync_library

    def handler(request):
        endpoint = request.url.path.rsplit("/", 1)[-1]
        if endpoint == "getArtists.view":
            return httpx.Response(200, json={"subsonic-response": {"status": "ok", "artists": {"index": []}}})
        if endpoint == "getAlbumList2.view":
            offset = int(request.url.params.get("offset", 0))
            payload = [{"id": "al1", "name": "A", "artistId": "a1"}] if offset == 0 else []
            return httpx.Response(200, json={"subsonic-response": {"status": "ok", "albumList2": {"album": payload}}})
        return httpx.Response(
            200,
            json={
                "subsonic-response": {
                    "status": "ok",
                    "album": {"id": "al1", "song": [{"id": "t1", "title": "S", "artistId": "a1"}]},
                }
            },
        )

    client = SubsonicClient(settings)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    sync_library(conn, client)
    track = conn.execute("SELECT album_id FROM tracks WHERE navidrome_id='t1'").fetchone()
    assert track["album_id"] is None


def test_sync_track_artist_empty_string(settings, conn):
    import httpx

    from app.subsonic import SubsonicClient, sync_library

    def handler(request):
        endpoint = request.url.path.rsplit("/", 1)[-1]
        if endpoint == "getArtists.view":
            return httpx.Response(200, json={"subsonic-response": {"status": "ok", "artists": {"index": []}}})
        if endpoint == "getAlbumList2.view":
            offset = int(request.url.params.get("offset", 0))
            payload = [{"id": "al1", "name": "A", "artistId": "a1"}] if offset == 0 else []
            return httpx.Response(200, json={"subsonic-response": {"status": "ok", "albumList2": {"album": payload}}})
        return httpx.Response(
            200,
            json={
                "subsonic-response": {
                    "status": "ok",
                    "album": {"id": "al1", "song": [{"id": "t1", "albumId": "al1", "artistId": ""}]},
                }
            },
        )

    client = SubsonicClient(settings)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    sync_library(conn, client)
    track = conn.execute("SELECT artist_id FROM tracks WHERE navidrome_id='t1'").fetchone()
    # sin artista propio, hereda el artista del álbum
    assert track["artist_id"] == "artist:a1"
