from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.conftest import seed_library

PLUGIN = Path(__file__).resolve().parents[1] / "app" / "janitor" / "bardo.py"


# retrieval 71-72: facets JSON inválido en ficha de track

def test_hydrate_track_ficha_corrupt_json(conn):
    from app.agent import retrieval

    seed_library(conn)
    conn.execute(
        "INSERT INTO fichas(entity_type, entity_id, facets, description, confidence, source, content_hash) "
        "VALUES ('track', 't1', '{roto', 'd', 0.9, 'llm', 'h')"
    )
    conn.commit()
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    # al fallar el JSON, cae a la ficha del álbum (fiesta, epico)
    assert target["moods"] == ["fiesta", "epico"]
    assert target["description"] == "d"


# cli 224: __main__ guard (runpy ejecuta el módulo como __main__)

def test_cli_main_guard(monkeypatch):
    import runpy

    monkeypatch.setattr(sys, "argv", ["bardo", "health"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("app.cli", run_name="__main__")
    assert exc.value.code == 0


# bardo 61: distance None -> raw is None

def test_plugin_distance_of_none_attr():
    from app.janitor import bardo as mod

    class NoDistance:
        pass

    assert mod._distance_of(NoDistance()) is None


# bardo 121: on_write con path conocido emite

def test_plugin_on_write_emits(plugin_factory, tmp_path):
    plugin = plugin_factory()
    plugin._before["/music/a.flac"] = {"artist": "?"}
    plugin.on_write(
        item={"artist": "Nuevo", "path": b"/music/a.flac"},
        path=b"/music/a.flac",
        tags={},
    )
    import json

    records = [
        json.loads(line)
        for line in (tmp_path / "beets.jsonl").read_text().splitlines()
    ]
    assert records[0]["new_tags"]["artist"] == "Nuevo"


# main 335: _check_subsonic ok

def test_check_subsonic_ok(monkeypatch):
    from app.config import get_settings
    from app.main import _check_subsonic

    class OkClient:
        def __init__(self, settings):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def ping(self):
            return {"status": "ok"}

    monkeypatch.setattr("app.subsonic.SubsonicClient", OkClient)
    assert _check_subsonic(get_settings()) is True


# ollama 41-53: escaneo de llaves con escapes y strings

def test_extract_json_escapes_and_nested_braces():
    from app.ollama import extract_json

    text = '{"a": "con \\\\ backslash", "b": {"c": [1, 2]}}'
    assert extract_json(text)["b"]["c"] == [1, 2]


def test_extract_json_string_with_unbalanced_brace():
    from app.ollama import extract_json

    text = 'pre {"x": "texto con } suelto"} post'
    assert extract_json(text) == {"x": "texto con } suelto"}


def test_extract_json_string_with_open_brace():
    from app.ollama import extract_json

    assert extract_json('{"x": "a { b"}') == {"x": "a { b"}


def test_extract_json_escaped_backslash_before_quote():
    from app.ollama import extract_json

    # texto previo fuerza el escáner; el string cierra tras un backslash escapado
    assert extract_json('nota: {"x": "c:\\\\"}') == {"x": "c:\\"}


def test_extract_json_string_with_escape_at_end():
    from app.ollama import extract_json

    text = 'nota {"path": "C:\\\\ruta", "n": 1} fin'
    assert extract_json(text)["n"] == 1


def test_extract_json_escaped_quote_inside_string():
    from app.ollama import extract_json

    text = 'bla {"x": "dice \\"hola\\"", "y": 2} bla'
    result = extract_json(text)
    assert result["y"] == 2
    assert "hola" in result["x"]


def test_extract_json_broken_opener_skips_to_next():
    from app.ollama import extract_json

    # el primer '{' nunca cierra válido; el segundo es el objeto correcto
    text = "{roto {\"ok\": 1}"
    assert extract_json(text) == {"ok": 1}


# ollama 82-84: error con detalle y sin detalle

def test_extract_json_invalid_detail():
    from app.ollama import extract_json

    with pytest.raises(ValueError) as exc:
        extract_json('{"a": 1 2 3}')
    assert "invalid JSON" in str(exc.value)


def test_extract_json_mismatched_brackets():
    from app.ollama import extract_json

    with pytest.raises(ValueError):
        extract_json('{"a": [1, 2}')


# subsonic 496: artista existente para track

def test_sync_track_reuses_existing_artist(settings, conn):
    import httpx

    from app.subsonic import SubsonicClient, sync_library

    def handler(request):
        endpoint = request.url.path.rsplit("/", 1)[-1]
        if endpoint == "getArtists.view":
            return httpx.Response(
                200,
                json={
                    "subsonic-response": {
                        "status": "ok",
                        "artists": {
                            "index": [{"artist": [{"id": "a2", "name": "Guest"}]}]
                        },
                    }
                },
            )
        if endpoint == "getAlbumList2.view":
            offset = int(request.url.params.get("offset", 0))
            payload = [{"id": "al1", "name": "A", "artistId": "a1"}] if offset == 0 else []
            return httpx.Response(
                200,
                json={"subsonic-response": {"status": "ok", "albumList2": {"album": payload}}},
            )
        return httpx.Response(
            200,
            json={
                "subsonic-response": {
                    "status": "ok",
                    "album": {
                        "id": "al1",
                        "song": [
                            {
                                "id": "t1",
                                "albumId": "al1",
                                # artista invitado ya existente en el índice
                                "artistId": "a2",
                            }
                        ],
                    },
                }
            },
        )

    client = SubsonicClient(settings)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    sync_library(conn, client)
    track = conn.execute("SELECT artist_id FROM tracks WHERE navidrome_id='t1'").fetchone()
    assert track["artist_id"] == "artist:a2"
    assert conn.execute("SELECT COUNT(*) AS n FROM artists").fetchone()["n"] == 2


@pytest.fixture()
def plugin_factory(tmp_path, monkeypatch):
    def factory():
        spec = importlib.util.spec_from_file_location("bardo_plugin4", PLUGIN)
        module = importlib.util.module_from_spec(spec)
        sys.modules["bardo_plugin4"] = module
        spec.loader.exec_module(module)
        monkeypatch.setenv("BARDO_BEETS_LOG", str(tmp_path / "beets.jsonl"))
        return module.BardoLogPlugin()

    return factory
