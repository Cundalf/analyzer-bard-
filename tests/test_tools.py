from __future__ import annotations

import json

import pytest

from app.agent.tools import (
    ToolContext,
    execute_tool,
    tool_result_message,
    tool_schemas,
    validate_track_ids,
)
from app.enrich.canonicalize import Canonicalizer
from tests.conftest import FakeOllama, FakeSubsonic, seed_library


def make_ctx(settings, conn, **kwargs) -> ToolContext:
    return ToolContext(
        conn=conn,
        ollama=kwargs.pop("ollama", FakeOllama()),
        settings=settings,
        canon=Canonicalizer.from_db(conn),
        **kwargs,
    )


# ------------------------------------------------------------ schemas

def test_tool_schemas_without_web_search(settings):
    settings.web_search_enabled = False
    names = [t["function"]["name"] for t in tool_schemas(settings)]
    assert names == [
        "search_candidates",
        "list_artists",
        "list_albums",
        "list_tracks",
        "create_playlist",
    ]


def test_tool_schemas_with_web_search(settings):
    settings.web_search_enabled = True
    names = [t["function"]["name"] for t in tool_schemas(settings)]
    assert "web_search" in names
    assert names[-1] == "create_playlist"


# ------------------------------------------------------------ validate

def test_validate_track_ids_empty(conn):
    assert validate_track_ids(conn, []) == []


def test_validate_track_ids_mixed(conn):
    seed_library(conn)
    assert validate_track_ids(conn, ["t1", "fake", "t3"]) == ["t1", "t3"]


def test_validate_track_ids_dedupes_preserving_order(conn):
    seed_library(conn)
    assert validate_track_ids(conn, ["t3", "t1", "t3"]) == ["t3", "t1"]


def test_validate_track_ids_chunking(conn):
    seed_library(conn)
    ids = ["fake"] * 401 + ["t1"]
    assert validate_track_ids(conn, ids) == ["t1"]


# ------------------------------------------------------------ search_candidates

@pytest.mark.anyio
async def test_search_candidates_returns_payload(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn)
    result = await execute_tool(
        "search_candidates", {"query": "dwarves cerveza", "limit": 10}, ctx
    )
    assert result["count"] >= 1
    candidate = result["candidates"][0]
    assert set(candidate) == {"track_id", "title", "artist", "album", "moods", "themes"}


@pytest.mark.anyio
async def test_search_candidates_limit_clamped(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn)
    low = await execute_tool("search_candidates", {"query": "metal", "limit": 0}, ctx)
    high = await execute_tool("search_candidates", {"query": "metal", "limit": 9999}, ctx)
    assert low["count"] >= 1
    assert high["count"] >= 1


@pytest.mark.anyio
async def test_search_candidates_default_query(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn)
    result = await execute_tool("search_candidates", {}, ctx)
    assert "count" in result


@pytest.mark.anyio
async def test_search_candidates_short_terms_ignored(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn)
    result = await execute_tool("search_candidates", {"query": "ab cd"}, ctx)
    assert result["count"] == 0


@pytest.mark.anyio
async def test_search_candidates_vector_failure_is_tolerated(settings, conn):
    seed_library(conn)

    class BrokenEmbed(FakeOllama):
        async def embed_one(self, text, **kwargs):
            raise RuntimeError("embed caído")

    ctx = make_ctx(settings, conn, ollama=BrokenEmbed())
    result = await execute_tool("search_candidates", {"query": "dwarves"}, ctx)
    assert result["count"] >= 1


@pytest.mark.anyio
async def test_search_candidates_year_filters(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn)
    result = await execute_tool(
        "search_candidates",
        {"query": "nightfall", "year_min": 2000, "year_max": 2010},
        ctx,
    )
    assert result["count"] == 0


@pytest.mark.anyio
async def test_search_candidates_exclude_genres(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn)
    result = await execute_tool(
        "search_candidates",
        {"query": "metal", "exclude_genres": ["power metal"]},
        ctx,
    )
    for c in result["candidates"]:
        assert c["album"] != "Nightfall in Middle-Earth"


# ------------------------------------------------------------ list tools

@pytest.mark.anyio
async def test_list_artists(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn)
    result = await execute_tool("list_artists", {"query": "wind"}, ctx)
    assert result["artists"] == [{"navidrome_id": "a1", "name": "Wind Rose"}]


@pytest.mark.anyio
async def test_list_artists_no_query_returns_all(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn)
    result = await execute_tool("list_artists", {}, ctx)
    assert len(result["artists"]) == 2


@pytest.mark.anyio
async def test_list_albums_by_artist(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn)
    result = await execute_tool("list_albums", {"artist": "Wind Rose"}, ctx)
    assert result["albums"][0]["navidrome_id"] == "al1"
    assert result["albums"][0]["artist"] == "Wind Rose"


@pytest.mark.anyio
async def test_list_albums_by_name(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn)
    result = await execute_tool("list_albums", {"artist": "Wintersaga"}, ctx)
    assert result["albums"][0]["navidrome_id"] == "al1"


@pytest.mark.anyio
async def test_list_tracks_all_and_filtered(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn)
    all_tracks = await execute_tool("list_tracks", {"album": "Wintersaga"}, ctx)
    assert {t["track_id"] for t in all_tracks["tracks"]} == {"t1", "t2"}
    filtered = await execute_tool(
        "list_tracks", {"album": "Wintersaga", "artist": "Blind"}, ctx
    )
    assert filtered["tracks"] == []


@pytest.mark.anyio
async def test_unknown_tool(settings, conn):
    seed_library(conn)
    result = await execute_tool("no_existe", {}, make_ctx(settings, conn))
    assert result["error"].startswith("tool desconocida")


# ------------------------------------------------------------ web_search

@pytest.mark.anyio
async def test_web_search_disabled(settings, conn):
    result = await execute_tool("web_search", {"query": "x"}, make_ctx(settings, conn))
    assert "error" in result


@pytest.mark.anyio
async def test_web_search_enabled(settings, conn):
    async def fake_search(query, ctx):
        return {"answer": True, "query": query}

    ctx = make_ctx(settings, conn, web_search=fake_search)
    result = await execute_tool("web_search", {"query": "tolkien?"}, ctx)
    assert result == {"answer": True, "query": "tolkien?"}


# ------------------------------------------------------------ create_playlist

@pytest.mark.anyio
async def test_create_playlist_preview(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn, allow_create=False)
    result = await execute_tool(
        "create_playlist", {"name": "P", "track_ids": ["t1", "fake", "t1"]}, ctx
    )
    assert result["status"] == "preview"
    assert result["valid_track_ids"] == ["t1"]
    assert result["rejected_track_ids"] == ["fake"]
    assert ctx.validation[0]["requested"] == 3


@pytest.mark.anyio
async def test_create_playlist_no_name_uses_default(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn)
    result = await execute_tool("create_playlist", {"track_ids": ["t1"]}, ctx)
    assert result["name"] == "Bardo"


@pytest.mark.anyio
async def test_create_playlist_whitespace_name(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn)
    result = await execute_tool(
        "create_playlist", {"name": "   ", "track_ids": ["t1"]}, ctx
    )
    assert result["name"] == "Bardo"


@pytest.mark.anyio
async def test_create_playlist_created(settings, conn):
    seed_library(conn)
    subsonic = FakeSubsonic()
    ctx = make_ctx(settings, conn, allow_create=True, subsonic=subsonic)
    result = await execute_tool(
        "create_playlist", {"name": "Taberna", "track_ids": ["t1", "t2"]}, ctx
    )
    assert result["status"] == "created"
    assert result["id"] == "pl-test"
    assert subsonic.created == [("Taberna", ["t1", "t2"])]
    assert ctx.created_playlists[0]["track_ids"] == ["t1", "t2"]


@pytest.mark.anyio
async def test_create_playlist_all_invalid(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn, allow_create=True, subsonic=FakeSubsonic())
    result = await execute_tool(
        "create_playlist", {"name": "P", "track_ids": ["fake"]}, ctx
    )
    assert "error" in result
    assert ctx.created_playlists == []


@pytest.mark.anyio
async def test_create_playlist_no_subsonic_client(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn, allow_create=True, subsonic=None)
    result = await execute_tool(
        "create_playlist", {"name": "P", "track_ids": ["t1"]}, ctx
    )
    assert "error" in result


@pytest.mark.anyio
async def test_create_playlist_empty_ids(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn, allow_create=True, subsonic=FakeSubsonic())
    result = await execute_tool("create_playlist", {"name": "P", "track_ids": []}, ctx)
    assert "error" in result


@pytest.mark.anyio
async def test_create_playlist_song_count_fallback(settings, conn):
    seed_library(conn)

    class NoCount(FakeSubsonic):
        def create_playlist(self, name, track_ids):
            self.created.append((name, list(track_ids)))

            class P:
                pass

            playlist = P()
            playlist.id = "pl"
            playlist.name = name
            playlist.song_count = 0
            return playlist

    ctx = make_ctx(settings, conn, allow_create=True, subsonic=NoCount())
    result = await execute_tool(
        "create_playlist", {"name": "P", "track_ids": ["t1", "t2"]}, ctx
    )
    assert result["song_count"] == 2


@pytest.mark.anyio
async def test_create_playlist_numeric_ids_coerced(settings, conn):
    seed_library(conn)
    ctx = make_ctx(settings, conn)
    result = await execute_tool(
        "create_playlist", {"name": "P", "track_ids": [1, None]}, ctx
    )
    assert result["valid_track_ids"] == []


# ------------------------------------------------------------ result message

def test_tool_result_message_short():
    text = tool_result_message("t", {"a": 1})
    assert json.loads(text) == {"a": 1}


def test_tool_result_message_truncates():
    text = tool_result_message("t", {"a": "x" * 30000})
    assert len(text) == 24000 + len("...(truncado)")
    assert text.endswith("...(truncado)")


def test_tool_result_message_unicode():
    text = tool_result_message("t", {"x": "ñandú"})
    assert "ñandú" in text
