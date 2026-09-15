from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from app.agent import retrieval
from app.agent.runner import generate_playlist
from app.enrich.pipeline import enrich_library
from tests.conftest import FakeOllama, FakeSubsonic, seed_library

PLUGIN = Path(__file__).resolve().parents[1] / "app" / "janitor" / "bardo.py"


# ------------------------------------------------------------ retrieval extras

def test_search_by_terms_all_filters(conn):
    seed_library(conn)
    results = retrieval.search_by_terms(
        conn,
        ["metal"],
        year_min=1990,
        year_max=2020,
        exclude_genres=["pop", "rock"],
    )
    assert all(r["track_id"] in {"t1", "t2", "t3", "t4"} for r in results)


def test_search_fts_all_filters(conn):
    from app.db import fts_upsert

    seed_library(conn)
    fts_upsert(conn, "track", "t1", "dwarves beer")
    fts_upsert(conn, "album", "al2", "nightfall tolkien")
    fts_upsert(conn, "artist", "a2", "blind guardian epico")
    conn.commit()
    results = retrieval.search_fts(
        conn, "dwarves tolkien epico", year_min=1990, year_max=2020,
        exclude_genres=["pop"],
    )
    assert results


def test_search_vectors_all_filters(conn):
    from app.db import vec_upsert

    seed_library(conn)
    vec_upsert(conn, "track", "t1", [1.0] * 768)
    vec_upsert(conn, "album", "al2", [0.9] * 768)
    vec_upsert(conn, "artist", "a2", [0.8] * 768)
    results = retrieval.search_vectors(
        conn, [1.0] * 768, year_min=1990, year_max=2020, exclude_genres=["pop"]
    )
    assert results


def test_search_vectors_year_filter_excludes(conn):
    from app.db import vec_upsert

    seed_library(conn)
    vec_upsert(conn, "album", "al2", [1.0] * 768)
    assert retrieval.search_vectors(conn, [1.0] * 768, year_min=2020) == []


def test_hydrate_non_json_facets_falls_back_to_album(conn):
    seed_library(conn)
    retrieval.search_by_terms(conn, ["dwarves"])
    assert conn.execute("SELECT COUNT(*) AS n FROM fichas").fetchone()["n"] == 1


# ------------------------------------------------------------ agent runner extras

@pytest.mark.anyio
async def test_expand_prompt_non_dict_response(settings):
    from app.agent.runner import expand_prompt
    from app.enrich.canonicalize import Canonicalizer

    plan = await expand_prompt(
        FakeOllama(chat_responses=[["lista", "inesperada"]]),
        Canonicalizer(),
        "x",
        settings,
    )
    assert plan["intent"] == "playlist"
    assert plan["expanded_terms"] == []


@pytest.mark.anyio
async def test_generate_playlist_save_direct_after_agent_failure(conn, settings):
    seed_library(conn)
    responses = [
        {"canonical_terms": [], "expanded_terms": ["dwarves"], "moods": []},
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "create_playlist",
                            "arguments": {"name": "P", "track_ids": ["t1", "t2"]},
                        }
                    }
                ],
            }
        },
    ]
    subsonic = FakeSubsonic()

    class NoCreateTool(FakeOllama):
        pass

    result = await generate_playlist(
        conn,
        "dwarves",
        settings=settings,
        ollama=NoCreateTool(chat_responses=responses),
        client=subsonic,
        save=True,
        use_agent=True,
    )
    assert result["saved"] is True
    assert subsonic.created


@pytest.mark.anyio
async def test_generate_playlist_owned_client_and_ollama_closed(conn, settings, monkeypatch):
    closed = {"client": False, "ollama": False}

    class OwnedSubsonic(FakeSubsonic):
        def close(self):
            closed["client"] = True

    class OwnedOllama(FakeOllama):
        async def close(self):
            closed["ollama"] = True

    monkeypatch.setattr("app.subsonic.SubsonicClient", lambda s: OwnedSubsonic())
    monkeypatch.setattr("app.agent.runner.OllamaClient", lambda s: OwnedOllama(
        chat_responses=[
            {"canonical_terms": [], "expanded_terms": [], "moods": []},
            {"playlist_name": "P", "track_ids": [], "reasoning": ""},
        ]
    ))
    seed_library(conn)
    await generate_playlist(
        conn, "x", settings=settings, client=None, use_agent=False
    )
    assert closed == {"client": True, "ollama": True}


# ------------------------------------------------------------ pipeline extras

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
            __import__("app.enrich.artist", fromlist=["_hash_inputs"])._hash_inputs(
                {"row": "t1"}
            ),
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


# ------------------------------------------------------------ plugin beets extras

@pytest.fixture()
def plugin(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("bardo_plugin2", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bardo_plugin2"] = module
    spec.loader.exec_module(module)
    monkeypatch.setenv("BARDO_BEETS_LOG", str(tmp_path / "beets.jsonl"))
    return module.BardoLogPlugin()


def test_plugin_log_path_env_first(plugin, tmp_path, monkeypatch):
    from app.janitor import bardo as mod

    monkeypatch.setenv("BARDO_BEETS_LOG", str(tmp_path / "override.jsonl"))
    assert mod._log_path() == tmp_path / "override.jsonl"


def test_plugin_clean_bytes():
    from app.janitor import bardo as mod

    assert mod._clean(b"/music/a.flac") == "/music/a.flac"
    assert mod._clean("texto") == "texto"
    assert mod._clean(7) == 7


def test_plugin_distance_of_none_and_plain():
    from app.janitor import bardo as mod

    assert mod._distance_of(None) is None

    class Plain:
        distance = 0.4

    assert mod._distance_of(Plain()) == pytest.approx(0.4)


@pytest.mark.anyio
async def test_plugin_empty_queue_noop(plugin, tmp_path):
    class Task:
        items: list = []
        choice_flag = None
        skip = False
        album = ""
        match = None
        candidates: list = []

    task = Task()
    plugin.on_task_created(session=None, task=task)
    plugin.on_task_choice(session=None, task=task)
    plugin.on_write(item={}, path=b"/x", tags={})
    assert plugin.commands() == []


def test_plugin_skip_status_recorded(plugin, tmp_path):
    import json

    class Item(dict):
        path = b"/music/skip.flac"

    item = Item(artist="?")

    class Task:
        items = [item]
        choice_flag = type("F", (), {"name": "SKIP"})()
        skip = True
        album = ""
        match = None
        candidates: list = []

    task = Task()
    plugin.on_task_created(session=None, task=task)
    plugin.on_task_choice(session=None, task=task)
    log = tmp_path / "beets.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["status"] == "SKIP"
    assert records[0]["file"] == "/music/skip.flac"


def test_plugin_album_fallback_from_item(plugin):
    class Item(dict):
        path = b"/m/a.flac"

    item = Item(album="DesdeItem")
    task = type(
        "T",
        (),
        {
            "items": [item],
            "choice_flag": type("F", (), {"name": "ASIS"})(),
            "skip": False,
            "album": "",
            "match": None,
            "candidates": [],
        },
    )()
    plugin.on_task_created(session=None, task=task)
    plugin.on_task_choice(session=None, task=task)
    assert plugin._albums["/m/a.flac"] == "DesdeItem"


def test_plugin_emit_ignores_path_without_before(plugin, tmp_path):
    plugin._emit("/desconocido", {"artist": "x"})
    assert not (tmp_path / "beets.jsonl").exists()
