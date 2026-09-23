from __future__ import annotations

import pytest
from tests.conftest import FakeOllama, FakeSubsonic, seed_library

from app.agent.loop import run_agent
from app.agent.retrieval import FacetFilters
from app.agent.runner import (
    _attach_track_details,
    _candidate_payload,
    _fetch_candidates,
    _ids_from_trace,
    _name_from_trace,
    _save_direct,
    expand_prompt,
    generate_playlist,
    recall_candidates,
    rerank_candidates,
)
from app.agent.tools import ToolContext
from app.enrich.canonicalize import Canonicalizer


def ctx_for(conn, settings, ollama, **kwargs) -> ToolContext:
    return ToolContext(
        conn=conn,
        ollama=ollama,
        settings=settings,
        canon=Canonicalizer.from_db(conn),
        **kwargs,
    )


def tool_call(name, arguments):
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
        }
    }


# ------------------------------------------------------------ expand_prompt


@pytest.mark.anyio
async def test_expand_prompt_full(settings):
    payload = {
        "intent": "playlist",
        "canonical_terms": ["taberna"],
        "expanded_terms": ["Folk Metal", "party"],
        "moods": ["Epico"],
        "reference": "tolkien",
        "filters": {"year_min": 1990, "year_max": 2020, "exclude_genres": ["pop"]},
        "size": 25,
    }
    ollama = FakeOllama(chat_responses=[payload])
    canon = Canonicalizer({"party": "fiesta"})
    plan = await expand_prompt(ollama, canon, "fiesta de taberna")
    assert plan["intent"] == "playlist"
    assert plan["canonical_terms"] == ["taberna"]
    assert "fiesta" in plan["expanded_terms"]
    assert plan["moods"] == ["epico"]
    assert plan["reference"] == "tolkien"
    assert plan["filters"].year_max == 2020
    assert plan["filters"].exclude_genres == ["pop"]
    assert plan["size"] == 25
    assert plan["raw_prompt"] == "fiesta de taberna"


@pytest.mark.anyio
async def test_expand_prompt_llm_failure_falls_back(settings):
    ollama = FakeOllama(chat_responses=[RuntimeError("boom")])
    plan = await expand_prompt(ollama, Canonicalizer(), "taberna")
    assert plan["intent"] == "playlist"
    assert plan["canonical_terms"] == []
    assert plan["expanded_terms"] == []
    assert plan["size"] == settings.playlist_default_size
    assert plan["filters"].year_min is None
    assert plan["filters"].year_max is None
    assert plan["filters"].exclude_genres == []


@pytest.mark.anyio
async def test_expand_prompt_clamps_size(settings):
    settings.playlist_max_size = 50
    plan = await expand_prompt(
        FakeOllama(chat_responses=[{"size": 9999}]), Canonicalizer(), "x", settings
    )
    assert plan["size"] == 50
    plan2 = await expand_prompt(
        FakeOllama(chat_responses=[{"size": 0}]), Canonicalizer(), "x", settings
    )
    assert plan2["size"] == 5


@pytest.mark.anyio
async def test_expand_prompt_garbage_filters(settings):
    payload = {"filters": "no-dict", "size": "abc"}
    plan = await expand_prompt(FakeOllama(chat_responses=[payload]), Canonicalizer(), "x")
    assert plan["filters"].year_min is None
    assert plan["size"] == settings.playlist_default_size


@pytest.mark.anyio
async def test_expand_prompt_none_terms(settings):
    payload = {"canonical_terms": None, "expanded_terms": None, "moods": None}
    plan = await expand_prompt(FakeOllama(chat_responses=[payload]), Canonicalizer(), "x")
    assert plan["canonical_terms"] == []
    assert plan["expanded_terms"] == []


# ------------------------------------------------------------ recall


def test_recall_candidates_uses_prompt_and_terms(conn):
    seed_library(conn)
    plan = {
        "raw_prompt": "enanos",
        "expanded_terms": ["dwarves", "cerveza"],
        "filters": {},
    }
    results = recall_candidates(conn, plan, limit=10)
    assert {r["track_id"] for r in results} == {"t1", "t2"}


def test_recall_candidates_with_embedding(conn):
    from app.db import vec_upsert

    seed_library(conn)
    vec_upsert(conn, "track", "t3", [1.0] * 768)
    plan = {"raw_prompt": "x", "expanded_terms": [], "moods": [], "filters": FacetFilters()}
    results = recall_candidates(conn, plan, limit=10, embedding=[1.0] * 768)
    assert results


def test_recall_candidates_empty_plan(conn):
    seed_library(conn)
    plan = {"raw_prompt": "zzz", "expanded_terms": [], "moods": [], "filters": FacetFilters()}
    assert recall_candidates(conn, plan, limit=10) == []


def test_recall_candidates_filters_applied(conn):
    seed_library(conn)
    plan = {
        "raw_prompt": "nightfall",
        "expanded_terms": ["nightfall"],
        "moods": [],
        "filters": FacetFilters(year_min=2000, year_max=2010),
    }
    assert recall_candidates(conn, plan, limit=10) == []


def test_candidate_payload_truncates_lists():
    candidates = [
        {
            "track_id": "t",
            "title": "T",
            "artist": "A",
            "album": "B",
            "year": None,
            "genre": None,
            "moods": [f"m{i}" for i in range(10)],
            "themes": [f"t{i}" for i in range(10)],
        }
    ]
    payload = _candidate_payload(candidates)
    assert len(payload[0]["moods"]) == 5
    assert len(payload[0]["themes"]) == 5


# ------------------------------------------------------------ rerank


@pytest.mark.anyio
async def test_rerank_no_candidates(conn, settings):
    result = await rerank_candidates(conn, FakeOllama(), "x", [], 10)
    assert result == {"playlist_name": "", "track_ids": [], "reasoning": "sin candidatos"}


@pytest.mark.anyio
async def test_rerank_validates_and_limits(conn, settings):
    seed_library(conn)
    raw = {"playlist_name": "N", "track_ids": ["t1", "fake", "t2"], "reasoning": "r"}
    result = await rerank_candidates(conn, FakeOllama(chat_responses=[raw]), "x", _cands(), 1)
    assert result["track_ids"] == ["t1"]
    assert result["playlist_name"] == "N"


@pytest.mark.anyio
async def test_rerank_fallback_all_invalid(conn, settings):
    seed_library(conn)
    raw = {"playlist_name": "N", "track_ids": ["fake1", "fake2"], "reasoning": "r"}
    result = await rerank_candidates(conn, FakeOllama(chat_responses=[raw]), "prompt", _cands(), 2)
    assert result["track_ids"] == ["t1", "t2"]
    assert "fallback" in result["reasoning"]


@pytest.mark.anyio
async def test_rerank_llm_failure_uses_candidates(conn, settings):
    seed_library(conn)
    result = await rerank_candidates(
        conn, FakeOllama(chat_responses=[RuntimeError("x")]), "prompt", _cands(), 5
    )
    assert result["track_ids"] == ["t1", "t2"]
    assert result["playlist_name"].startswith("Bardo:")


@pytest.mark.anyio
async def test_rerank_empty_playlist_name_default(conn, settings):
    seed_library(conn)
    raw = {"playlist_name": "", "track_ids": ["t1"], "reasoning": ""}
    result = await rerank_candidates(
        conn, FakeOllama(chat_responses=[raw]), "mi prompt", _cands(), 5
    )
    assert result["playlist_name"] == "Bardo: mi prompt"


def _cands():
    return [
        {
            "track_id": "t1",
            "title": "A",
            "artist": "X",
            "album": "Al",
            "year": 2000,
            "genre": "g",
            "moods": [],
            "themes": [],
        },
        {
            "track_id": "t2",
            "title": "B",
            "artist": "X",
            "album": "Al",
            "year": 2001,
            "genre": "g",
            "moods": [],
            "themes": [],
        },
    ]


# ------------------------------------------------------------ generate_playlist


@pytest.mark.anyio
async def test_generate_playlist_agent_path(conn, settings):
    seed_library(conn)
    responses = [
        {"canonical_terms": [], "expanded_terms": ["dwarves"], "moods": []},
        tool_call("search_candidates", {"query": "dwarves"}),
        tool_call("create_playlist", {"name": "Taberna", "track_ids": ["t1", "t2"]}),
        {"content": "Listo"},
    ]
    ollama = FakeOllama(chat_responses=responses)
    result = await generate_playlist(
        conn,
        "dwarves",
        settings=settings,
        ollama=ollama,
        client=FakeSubsonic(),
        save=False,
        use_agent=True,
    )
    assert result["mode"] == "agent"
    assert result["playlist_name"] == "Taberna"
    assert result["track_ids"] == ["t1", "t2"]
    assert result["saved"] is False


@pytest.mark.anyio
async def test_generate_playlist_agent_save(conn, settings):
    seed_library(conn)
    responses = [
        {"canonical_terms": [], "expanded_terms": ["dwarves"], "moods": []},
        tool_call("create_playlist", {"name": "Taberna", "track_ids": ["t1"]}),
        {"content": "ok"},
    ]
    subsonic = FakeSubsonic()
    result = await generate_playlist(
        conn,
        "dwarves",
        settings=settings,
        ollama=FakeOllama(chat_responses=responses),
        client=subsonic,
        save=True,
        use_agent=True,
    )
    assert result["saved"] is True
    assert subsonic.created == [("Taberna", ["t1"])]


@pytest.mark.anyio
async def test_generate_playlist_agent_no_create_saves_direct(conn, settings):
    seed_library(conn)
    responses = [
        {"canonical_terms": [], "expanded_terms": ["dwarves"], "moods": []},
        tool_call("search_candidates", {"query": "dwarves"}),
        tool_call("create_playlist", {"name": "Taberna", "track_ids": ["t1"]}),
        {"content": "no la guardé"},
    ]
    subsonic = FakeSubsonic()
    result = await generate_playlist(
        conn,
        "dwarves",
        settings=settings,
        ollama=FakeOllama(chat_responses=responses),
        client=subsonic,
        save=True,
        use_agent=True,
    )
    # El tool no tenía permiso de crear (allow_create sigue el flag save=True en el ctx)
    assert result["track_ids"] == ["t1"]
    assert result["saved"] is True
    assert subsonic.created == [("Taberna", ["t1"])]


@pytest.mark.anyio
async def test_generate_playlist_agent_text_only(conn, settings):
    seed_library(conn)
    responses = [
        {"canonical_terms": [], "expanded_terms": [], "moods": []},
        {"content": "No encontré nada."},
    ]
    result = await generate_playlist(
        conn,
        "zzz",
        settings=settings,
        ollama=FakeOllama(chat_responses=responses),
        client=FakeSubsonic(),
        use_agent=True,
    )
    assert result["track_ids"] == []
    assert "No encontré nada" in result["reasoning"]


@pytest.mark.anyio
async def test_generate_playlist_rerank_path(conn, settings):
    seed_library(conn)
    responses = [
        {"canonical_terms": [], "expanded_terms": ["dwarves"], "moods": []},
        {"playlist_name": "Enanos", "track_ids": ["t1"], "reasoning": "r"},
    ]
    result = await generate_playlist(
        conn,
        "dwarves",
        settings=settings,
        ollama=FakeOllama(chat_responses=responses),
        client=FakeSubsonic(),
        use_agent=False,
    )
    assert result["mode"] == "rerank"
    assert result["playlist_name"] == "Enanos"
    assert result["track_ids"] == ["t1"]


@pytest.mark.anyio
async def test_generate_playlist_rerank_save(conn, settings):
    seed_library(conn)
    responses = [
        {"canonical_terms": [], "expanded_terms": ["dwarves"], "moods": []},
        {"playlist_name": "Enanos", "track_ids": ["t1"], "reasoning": "r"},
    ]
    subsonic = FakeSubsonic()
    result = await generate_playlist(
        conn,
        "dwarves",
        settings=settings,
        ollama=FakeOllama(chat_responses=responses),
        client=subsonic,
        save=True,
        use_agent=False,
    )
    assert result["saved"] is True
    assert subsonic.created == [("Enanos", ["t1"])]


@pytest.mark.anyio
async def test_generate_playlist_agent_disabled(conn, settings):
    settings.agent_enabled = False
    seed_library(conn)
    responses = [
        {"canonical_terms": [], "expanded_terms": ["dwarves"], "moods": []},
        {"playlist_name": "N", "track_ids": ["t1"], "reasoning": ""},
    ]
    result = await generate_playlist(
        conn,
        "dwarves",
        settings=settings,
        ollama=FakeOllama(chat_responses=responses),
        client=FakeSubsonic(),
        use_agent=True,
    )
    assert result["mode"] == "rerank"


@pytest.mark.anyio
async def test_generate_playlist_creates_own_client(conn, settings, monkeypatch):
    created = {"n": 0}

    class OwnedSubsonic(FakeSubsonic):
        def __init__(self, settings=None):
            super().__init__()
            created["n"] += 1

    monkeypatch.setattr("app.subsonic.SubsonicClient", OwnedSubsonic)
    seed_library(conn)
    responses = [
        {"canonical_terms": [], "expanded_terms": ["dwarves"], "moods": []},
        {"playlist_name": "N", "track_ids": ["t1"], "reasoning": ""},
    ]
    result = await generate_playlist(
        conn,
        "dwarves",
        settings=settings,
        ollama=FakeOllama(chat_responses=responses),
        client=None,
        use_agent=False,
    )
    assert created["n"] == 1
    assert result["track_ids"] == ["t1"]


@pytest.mark.anyio
async def test_generate_playlist_embedding_failure_tolerated(conn, settings):
    seed_library(conn)

    class NoEmbed(FakeOllama):
        async def embed_one(self, text, **kwargs):
            raise RuntimeError("sin vectores")

    responses = [
        {"canonical_terms": [], "expanded_terms": ["dwarves"], "moods": []},
        {"playlist_name": "N", "track_ids": ["t1"], "reasoning": ""},
    ]
    ollama = NoEmbed(chat_responses=responses)
    result = await generate_playlist(
        conn,
        "dwarves",
        settings=settings,
        ollama=ollama,
        client=FakeSubsonic(),
        use_agent=False,
    )
    assert result["track_ids"] == ["t1"]


@pytest.mark.anyio
async def test_generate_playlist_size_override(conn, settings):
    seed_library(conn)
    responses = [
        {"canonical_terms": [], "expanded_terms": ["dwarves"], "moods": []},
        {"playlist_name": "N", "track_ids": ["t1", "t2"], "reasoning": ""},
    ]
    result = await generate_playlist(
        conn,
        "dwarves",
        settings=settings,
        ollama=FakeOllama(chat_responses=responses),
        client=FakeSubsonic(),
        use_agent=False,
        size=1,
    )
    assert len(result["track_ids"]) == 1


# ------------------------------------------------------------ helpers


def test_name_from_trace():
    result = {
        "trace": [
            {"tool": "search_candidates", "arguments": {}},
            {"tool": "create_playlist", "arguments": {"name": "Taberna"}},
        ]
    }
    assert _name_from_trace(result) == "Taberna"
    assert _name_from_trace({"trace": []}) == ""
    assert _name_from_trace({}) == ""


def test_ids_from_trace_validates(conn):
    seed_library(conn)
    result = {"trace": [{"tool": "create_playlist", "arguments": {"track_ids": ["t1", "fake"]}}]}
    assert _ids_from_trace(result, conn) == ["t1"]
    assert _ids_from_trace({"trace": []}, conn) == []


def test_save_direct():
    subsonic = FakeSubsonic()
    result = _save_direct(subsonic, "P", ["t1", "t2"])
    assert result["saved"] is True
    assert result["created_playlists"][0]["song_count"] == 2


# ------------------------------------------------------------ loop


@pytest.mark.anyio
async def test_run_agent_string_arguments(conn, settings):
    seed_library(conn)
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "list_artists",
                            "arguments": '{"query": "wind"}',
                        }
                    }
                ],
            }
        },
        {"message": {"role": "assistant", "content": "ok"}},
    ]
    result = await run_agent(
        conn,
        "x",
        ctx=ctx_for(conn, settings, FakeOllama()),
        settings=settings,
        ollama=FakeOllama(chat_responses=responses),
    )
    assert result["tool_calls"] == 1


@pytest.mark.anyio
async def test_run_agent_invalid_string_arguments(conn, settings):
    seed_library(conn)
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "list_artists", "arguments": "no-json"}}],
            }
        },
        {"message": {"role": "assistant", "content": "ok"}},
    ]
    result = await run_agent(
        conn,
        "x",
        ctx=ctx_for(conn, settings, FakeOllama()),
        settings=settings,
        ollama=FakeOllama(chat_responses=responses),
    )
    assert result["tool_calls"] == 1
    assert result["trace"][0]["arguments"] == {}


@pytest.mark.anyio
async def test_run_agent_missing_message_key(conn, settings):
    seed_library(conn)
    result = await run_agent(
        conn,
        "x",
        ctx=ctx_for(conn, settings, FakeOllama()),
        settings=settings,
        ollama=FakeOllama(chat_responses=[{}]),
    )
    assert result["text"] == ""
    assert result["tool_calls"] == 0


@pytest.mark.anyio
async def test_run_agent_multiple_calls_in_one_response(conn, settings):
    seed_library(conn)
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "list_artists", "arguments": {"query": "a"}}},
                    {"function": {"name": "list_albums", "arguments": {"artist": "a"}}},
                ],
            }
        },
        {"message": {"role": "assistant", "content": "listo"}},
    ]
    result = await run_agent(
        conn,
        "x",
        ctx=ctx_for(conn, settings, FakeOllama()),
        settings=settings,
        ollama=FakeOllama(chat_responses=responses),
    )
    assert result["tool_calls"] == 2
    assert [t["tool"] for t in result["trace"]] == ["list_artists", "list_albums"]


@pytest.mark.anyio
async def test_run_agent_max_calls_cut_mid_response(conn, settings):
    seed_library(conn)
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "list_artists", "arguments": {}}},
                    {"function": {"name": "list_artists", "arguments": {}}},
                    {"function": {"name": "list_artists", "arguments": {}}},
                ],
            }
        }
    ]
    result = await run_agent(
        conn,
        "x",
        ctx=ctx_for(conn, settings, FakeOllama()),
        settings=settings,
        ollama=FakeOllama(chat_responses=responses),
        max_tool_calls=2,
    )
    assert result["tool_calls"] == 2


@pytest.mark.anyio
async def test_run_agent_creates_ctx_when_none(conn, settings):
    seed_library(conn)
    result = await run_agent(
        conn,
        "x",
        settings=settings,
        ollama=FakeOllama(chat_responses=[{"message": {"content": "sin tools"}}]),
    )
    assert result["tool_calls"] == 0


@pytest.mark.anyio
async def test_run_agent_owns_ollama_closes(conn, settings, monkeypatch):
    closed = {"v": False}

    class Owned(FakeOllama):
        async def close(self):
            closed["v"] = True

    monkeypatch.setattr("app.agent.loop.OllamaClient", lambda s: Owned())
    await run_agent(conn, "x", settings=settings)
    assert closed["v"] is True


@pytest.mark.anyio
async def test_run_agent_progress_events(conn, settings):
    seed_library(conn)
    events = []
    responses = [
        {"message": {"tool_calls": [{"function": {"name": "list_artists", "arguments": {}}}]}},
        {"message": {"content": "ok"}},
    ]
    await run_agent(
        conn,
        "x",
        ctx=ctx_for(conn, settings, FakeOllama()),
        settings=settings,
        ollama=FakeOllama(chat_responses=responses),
        progress=lambda s, p: events.append((s, p.get("name"))),
    )
    assert events == [("tool_call", "list_artists")]


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
    monkeypatch.setattr(
        "app.agent.runner.OllamaClient",
        lambda s: OwnedOllama(
            chat_responses=[
                {"canonical_terms": [], "expanded_terms": [], "moods": []},
                {"playlist_name": "P", "track_ids": [], "reasoning": ""},
            ]
        ),
    )
    seed_library(conn)
    await generate_playlist(conn, "x", settings=settings, client=None, use_agent=False)
    assert closed == {"client": True, "ollama": True}


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
        ollama=FakeOllama(
            chat_responses=[{"canonical_terms": [], "expanded_terms": ["dwarves"], "moods": []}, {}]
        ),
        client=subsonic,
        save=True,
        use_agent=True,
    )
    assert result["saved"] is True
    assert subsonic.created == [("P", ["t1"])]


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


def test_attach_track_details_from_candidates(conn):
    seed_library(conn)
    candidates = [
        {
            "track_id": "t1",
            "title": "Drunken Dwarves",
            "artist": "Wind Rose",
            "album": "Wintersaga",
            "year": 2019,
            "genre": "folk metal",
            "moods": ["fiesta", "a", "b", "c", "d"],
            "themes": ["cerveza"],
        }
    ]
    result = {"track_ids": ["t1"]}
    _attach_track_details(conn, result, candidates)
    assert result["tracks"][0]["title"] == "Drunken Dwarves"
    assert len(result["tracks"][0]["moods"]) == 4


def test_attach_track_details_empty(conn):
    seed_library(conn)
    result = {"track_ids": []}
    _attach_track_details(conn, result, [])
    assert result["tracks"] == []


def test_attach_track_details_fetches_missing(conn):
    seed_library(conn)
    result = {"track_ids": ["t3"]}
    _attach_track_details(conn, result, [])
    assert result["tracks"][0]["title"] == "Nightfall"
    assert result["tracks"][0]["artist"] == "Blind Guardian"


def test_attach_track_details_unknown_id(conn):
    seed_library(conn)
    result = {"track_ids": ["desconocido"]}
    _attach_track_details(conn, result, [])
    assert result["tracks"][0]["track_id"] == "desconocido"
    assert result["tracks"][0]["title"] == ""


def test_fetch_candidates_multiple_chunks(conn):
    seed_library(conn)
    ids = [f"fake{i}" for i in range(301)] + ["t1"]
    found = _fetch_candidates(conn, ids)
    assert "t1" in found
    assert len(found) == 1


def test_recall_candidates_moods_added_to_terms(conn):
    seed_library(conn)
    plan = {
        "raw_prompt": "sin match en texto",
        "expanded_terms": [],
        "moods": ["fiesta"],
        "filters": FacetFilters(),
    }
    results = recall_candidates(conn, plan, limit=10)
    assert results
