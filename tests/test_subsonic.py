from __future__ import annotations

import hashlib

import httpx
import pytest

from app.subsonic import (
    API_VERSION,
    Album,
    Artist,
    Playlist,
    SubsonicClient,
    SubsonicError,
    Track,
    sync_library,
)

BASE = "http://navidrome.test:4533"


def envelope(body: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={"subsonic-response": {"status": "ok", "version": "1.16.1", **body}},
    )


def failed(code=10, message="bad") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "subsonic-response": {
                "status": "failed",
                "error": {"code": code, "message": message},
            }
        },
    )


def mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.fixture()
def client(settings):
    return SubsonicClient(settings)


# ---------------------------------------------------------------- modelos


def test_track_from_json_full():
    t = Track.from_json(
        {
            "id": "t1",
            "title": "Song",
            "album": "Alb",
            "albumId": "al1",
            "artist": "Art",
            "artistId": "a1",
            "track": 3,
            "year": 1999,
            "genre": "Rock",
            "duration": 240,
            "path": "a/b/c.flac",
            "suffix": "flac",
            "musicBrainzId": "mb1",
            "coverArt": "cov1",
        }
    )
    assert (t.id, t.title, t.album_id, t.track, t.year, t.suffix) == (
        "t1",
        "Song",
        "al1",
        3,
        1999,
        "flac",
    )


def test_track_from_json_garbage_types():
    t = Track.from_json({"id": "t", "track": "x", "year": None, "duration": "abc"})
    assert t.track is None and t.year is None and t.duration is None
    assert t.genre == "" and t.path == ""


def test_album_from_json_name_and_title_fallback():
    a = Album.from_json({"id": "al", "title": "FromTitle"})
    assert a.name == "FromTitle"
    b = Album.from_json({"id": "al", "name": "", "title": "T2"})
    assert b.name == "T2"
    c = Album.from_json({"id": "al", "name": "Both", "title": "T2"})
    assert c.name == "Both"


def test_album_from_json_songs_and_defaults():
    a = Album.from_json({"id": "al", "name": "N", "songCount": "3", "song": [{"id": "t1"}]})
    assert a.song_count == 3
    assert a.duration == 0
    assert a.songs[0].id == "t1"


def test_artist_from_json_defaults():
    a = Artist.from_json({"id": "a"})
    assert a.name == "" and a.album_count == 0


def test_playlist_from_json_entries():
    p = Playlist.from_json(
        {"id": "pl", "name": "P", "songCount": 2, "entry": [{"id": "t1"}, {"id": "t2"}]}
    )
    assert p.song_count == 2
    assert [t.id for t in p.entries] == ["t1", "t2"]


# ---------------------------------------------------------------- auth/url


def test_auth_params_token_is_md5_pass_salt(client):
    params = client._auth_params()
    expected = hashlib.md5(("secret" + params["s"]).encode()).hexdigest()
    assert params["t"] == expected
    assert params["u"] == "admin"
    assert params["v"] == API_VERSION
    assert params["c"] == "bardo"
    assert params["f"] == "json"
    assert len(params["s"]) == 16


def test_auth_salt_is_random(client):
    assert client._auth_params()["s"] != client._auth_params()["s"]


def test_url_for_includes_params_and_skips_none(client):
    url = client.url_for("stream.view", id="t1", maxBitRate=None)
    assert "id=t1" in url
    assert "maxBitRate" not in url
    assert "u=admin" in url


def test_stream_url_with_and_without_bitrate(client):
    assert "maxBitRate=128" in client.stream_url("t1", 128)
    assert "maxBitRate" not in client.stream_url("t1")


def test_cover_url(client):
    url = client.cover_url("cov", size=500)
    assert "id=cov" in url and "size=500" in url


def test_context_manager_closes(settings):
    with SubsonicClient(settings) as c:
        assert c.base_url == BASE
    assert c._client.is_closed


# ---------------------------------------------------------------- request


def test_ping_ok_and_auth(client):
    def handler(request):
        assert request.url.params["u"] == "admin"
        return envelope({})

    client._client = httpx.Client(transport=mock_transport(handler))
    assert client.ping()["status"] == "ok"


def test_failed_response_raises_with_code(client):
    client._client = httpx.Client(transport=mock_transport(lambda r: failed(40, "nope")))
    with pytest.raises(SubsonicError) as exc:
        client.ping()
    assert exc.value.code == 40
    assert "nope" in str(exc.value)


def test_failed_without_code(client):
    def handler(request):
        return httpx.Response(200, json={"subsonic-response": {"status": "failed"}})

    client._client = httpx.Client(transport=mock_transport(handler))
    with pytest.raises(SubsonicError) as exc:
        client.ping()
    assert exc.value.code is None


def test_missing_subsonic_response_key(client):
    client._client = httpx.Client(transport=mock_transport(lambda r: httpx.Response(200, json={})))
    assert client.ping() == {}


def test_retries_then_success(client):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("boom", request=request)
        return envelope({})

    client._client = httpx.Client(transport=mock_transport(handler))
    client.ping()
    assert calls["n"] == 3


def test_retries_exhausted_raises(client):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ConnectError("boom", request=request)

    client._client = httpx.Client(transport=mock_transport(handler))
    with pytest.raises(SubsonicError) as exc:
        client.ping()
    assert "connection failed" in str(exc.value)
    assert calls["n"] == 3


def test_invalid_json_retries(client):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, text="<html>not json</html>")
        return envelope({})

    client._client = httpx.Client(transport=mock_transport(handler))
    client.ping()
    assert calls["n"] == 2


def test_http_500_retries(client):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="server error")
        return envelope({})

    client._client = httpx.Client(transport=mock_transport(handler))
    client.ping()
    assert calls["n"] == 2


def test_subsonic_error_does_not_retry(client):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return failed(70, "not found")

    client._client = httpx.Client(transport=mock_transport(handler))
    with pytest.raises(SubsonicError):
        client.ping()
    assert calls["n"] == 1


# ---------------------------------------------------------------- lectura


def test_get_artists_multiple_indexes_and_empty(client):
    def handler(request):
        return envelope(
            {
                "artists": {
                    "index": [
                        {"artist": [{"id": "a1", "name": "A"}]},
                        {"artist": [{"id": "a2", "name": "B"}, {"id": "a3", "name": "C"}]},
                        {},
                    ]
                }
            }
        )

    client._client = httpx.Client(transport=mock_transport(handler))
    assert [a.id for a in client.get_artists()] == ["a1", "a2", "a3"]


def test_get_artists_missing_key(client):
    client._client = httpx.Client(transport=mock_transport(lambda r: envelope({})))
    assert client.get_artists() == []


def test_get_artist_returns_albums(client):
    def handler(request):
        assert request.url.params["id"] == "a1"
        return envelope(
            {
                "artist": {
                    "id": "a1",
                    "name": "Wind",
                    "album": [{"id": "al1", "name": "Wintersaga"}],
                }
            }
        )

    client._client = httpx.Client(transport=mock_transport(handler))
    artist, albums = client.get_artist("a1")
    assert artist.name == "Wind"
    assert albums[0].id == "al1"


def test_get_album_empty(client):
    client._client = httpx.Client(transport=mock_transport(lambda r: envelope({})))
    assert client.get_album("nope").id == ""


def test_get_album_list2(client):
    def handler(request):
        assert request.url.params["type"] == "recent"
        assert request.url.params["size"] == "10"
        return envelope({"albumList2": {"album": [{"id": "al1"}]}})

    client._client = httpx.Client(transport=mock_transport(handler))
    albums = client.get_album_list2(list_type="recent", size=10, offset=0)
    assert albums[0].id == "al1"


def test_iter_all_albums_paginates_and_stops(client):
    pages = {
        0: [{"id": f"al{i}"} for i in range(3)],
        3: [{"id": "al3"}],
        4: [],
    }

    def handler(request):
        offset = int(request.url.params.get("offset", 0))
        return envelope({"albumList2": {"album": pages.get(offset, [])}})

    client._client = httpx.Client(transport=mock_transport(handler))
    albums = list(client.iter_all_albums(page_size=3))
    assert [a.id for a in albums] == ["al0", "al1", "al2", "al3"]


def test_iter_all_albums_full_last_page_exact(client):
    def handler(request):
        offset = int(request.url.params.get("offset", 0))
        if offset == 0:
            return envelope({"albumList2": {"album": [{"id": "a"}, {"id": "b"}]}})
        return envelope({"albumList2": {"album": []}})

    client._client = httpx.Client(transport=mock_transport(handler))
    albums = list(client.iter_all_albums(page_size=2))
    assert len(albums) == 2


def test_search3_full_and_empty(client):
    def handler(request):
        assert request.url.params["query"] == "tolkien"
        return envelope(
            {
                "searchResult3": {
                    "artist": [{"id": "a1", "name": "BG"}],
                    "album": [{"id": "al1", "name": "Nightfall"}],
                    "song": [{"id": "t1", "title": "Nightfall"}],
                }
            }
        )

    client._client = httpx.Client(transport=mock_transport(handler))
    result = client.search3("tolkien")
    assert result["artists"][0].name == "BG"
    assert result["albums"][0].id == "al1"
    assert result["songs"][0].id == "t1"

    client._client = httpx.Client(transport=mock_transport(lambda r: envelope({})))
    empty = client.search3("x")
    assert empty == {"artists": [], "albums": [], "songs": []}


def test_search3_offsets_passed(client):
    captured = {}

    def handler(request):
        captured.update(request.url.params)
        return envelope({})

    client._client = httpx.Client(transport=mock_transport(handler))
    client.search3("q", artist_offset=5, album_offset=6, song_offset=7)
    assert captured["artistOffset"] == "5"
    assert captured["albumOffset"] == "6"
    assert captured["songOffset"] == "7"


def test_get_playlists_and_playlist(client):
    def handler(request):
        if request.url.path.endswith("getPlaylists.view"):
            return envelope({"playlists": {"playlist": [{"id": "pl1", "name": "P"}]}})
        return envelope({"playlist": {"id": "pl1", "name": "P", "entry": [{"id": "t1"}]}})

    client._client = httpx.Client(transport=mock_transport(handler))
    assert client.get_playlists()[0].id == "pl1"
    assert client.get_playlist("pl1").entries[0].id == "t1"


# ---------------------------------------------------------------- escritura


def test_create_playlist_repeated_song_ids(client):
    captured = []

    def handler(request):
        captured.extend(request.url.params.get_list("songId"))
        return envelope({"playlist": {"id": "pl1", "name": "P", "songCount": 3}})

    client._client = httpx.Client(transport=mock_transport(handler))
    playlist = client.create_playlist("P", ["t1", "t2", "t3"])
    assert captured == ["t1", "t2", "t3"]
    assert playlist.id == "pl1"


def test_create_playlist_empty_ids(client):
    captured = {}

    def handler(request):
        captured["name"] = request.url.params.get("name")
        captured["ids"] = request.url.params.get_list("songId")
        return envelope({"playlist": {"id": "pl1", "name": "P"}})

    client._client = httpx.Client(transport=mock_transport(handler))
    client.create_playlist("Vacía", [])
    assert captured == {"name": "Vacía", "ids": []}


def test_create_playlist_empty_response(client):
    client._client = httpx.Client(transport=mock_transport(lambda r: envelope({})))
    playlist = client.create_playlist("P", ["t1"])
    assert playlist.id == ""
    assert playlist.name == ""


def test_update_playlist_all_params(client):
    captured = {}

    def handler(request):
        captured["id"] = request.url.params.get("playlistId")
        captured["name"] = request.url.params.get("name")
        captured["add"] = request.url.params.get_list("songIdToAdd")
        captured["remove"] = request.url.params.get_list("songIndexToRemove")
        return envelope({"playlist": {"id": "pl1"}})

    client._client = httpx.Client(transport=mock_transport(handler))
    client.update_playlist(
        "pl1", name="Nuevo", song_ids_to_add=["t1", "t2"], song_indexes_to_remove=[0, 3]
    )
    assert captured == {
        "id": "pl1",
        "name": "Nuevo",
        "add": ["t1", "t2"],
        "remove": ["0", "3"],
    }


def test_update_playlist_name_none_omitted(client):
    captured = {}

    def handler(request):
        captured.update(request.url.params)
        return envelope({"playlist": {"id": "pl1"}})

    client._client = httpx.Client(transport=mock_transport(handler))
    client.update_playlist("pl1", song_ids_to_add=["t1"])
    assert "name" not in captured


def test_delete_star_unstar(client):
    seen = []

    def handler(request):
        seen.append(request.url.path.rsplit("/", 1)[-1])
        assert request.url.params["id"] in {"pl1", "t1"}
        return envelope({})

    client._client = httpx.Client(transport=mock_transport(handler))
    client.delete_playlist("pl1")
    client.star("t1")
    client.unstar("t1")
    assert seen == ["deletePlaylist.view", "star.view", "unstar.view"]


def test_start_scan_full_flag(client):
    captured = []

    def handler(request):
        captured.append(request.url.params.get("fullScan"))
        return envelope({"scanStatus": {"scanning": True}})

    client._client = httpx.Client(transport=mock_transport(handler))
    assert client.start_scan(full=True)["scanning"] is True
    assert client.start_scan()["scanning"] is True
    assert captured[0] == "true"
    assert captured[1] in (None, "")


def test_get_scan_status_missing_key(client):
    client._client = httpx.Client(transport=mock_transport(lambda r: envelope({})))
    assert client.get_scan_status() == {}


# ---------------------------------------------------------------- sync


def make_sync_handler(pages, artist_index=None, album_details=None):
    artist_index = (
        artist_index
        if artist_index is not None
        else [{"artist": [{"id": "a1", "name": "Wind Rose", "albumCount": 1}]}]
    )
    album_details = album_details or {}

    def handler(request):
        endpoint = request.url.path.rsplit("/", 1)[-1]
        if endpoint == "getArtists.view":
            return envelope({"artists": {"index": artist_index}})
        if endpoint == "getAlbumList2.view":
            offset = int(request.url.params.get("offset", 0))
            return envelope({"albumList2": {"album": pages.get(offset, [])}})
        if endpoint == "getAlbum.view":
            payload = album_details.get(request.url.params["id"], {"id": request.url.params["id"]})
            return envelope({"album": payload})
        return envelope({})

    return handler


def sync_client(settings, handler) -> SubsonicClient:
    c = SubsonicClient(settings)
    c._client = httpx.Client(transport=mock_transport(handler))
    return c


def test_sync_library_full(settings, conn):
    handler = make_sync_handler(
        {
            0: [
                {
                    "id": "al1",
                    "name": "Wintersaga",
                    "artist": "Wind Rose",
                    "artistId": "a1",
                    "year": 2019,
                    "genre": "folk metal",
                    "songCount": 1,
                }
            ]
        },
        album_details={
            "al1": {
                "id": "al1",
                "name": "Wintersaga",
                "song": [
                    {
                        "id": "t1",
                        "title": "Drunken Dwarves",
                        "albumId": "al1",
                        "artistId": "a1",
                        "duration": 240,
                    },
                    {"id": "t2", "title": "Mine Mine Mine!", "albumId": "al1", "artistId": "a1"},
                ],
            }
        },
    )
    client = sync_client(settings, handler)
    stats = sync_library(conn, client)
    assert stats == {"artists": 1, "albums": 1, "tracks": 2}
    track = conn.execute("SELECT * FROM tracks WHERE navidrome_id = 't1'").fetchone()
    assert track["title"] == "Drunken Dwarves"
    assert track["album_id"] == "album:al1"
    assert track["artist_id"] == "artist:a1"
    assert track["duration"] == 240
    row = conn.execute("SELECT path FROM tracks WHERE navidrome_id='t2'").fetchone()
    assert row["path"] is None or row["path"] == ""


def test_sync_library_incremental_skips_existing_tracks(settings, conn):
    calls = {"album_fetch": 0}

    def handler(request):
        endpoint = request.url.path.rsplit("/", 1)[-1]
        if endpoint == "getArtists.view":
            return envelope({"artists": {"index": []}})
        if endpoint == "getAlbumList2.view":
            offset = int(request.url.params.get("offset", 0))
            if offset == 0:
                return envelope(
                    {"albumList2": {"album": [{"id": "al1", "name": "A", "artistId": "a1"}]}}
                )
            return envelope({"albumList2": {"album": []}})
        if endpoint == "getAlbum.view":
            calls["album_fetch"] += 1
            return envelope({"album": {"id": "al1", "song": [{"id": "t1", "albumId": "al1"}]}})
        return envelope({})

    client = sync_client(settings, handler)
    sync_library(conn, client)
    assert calls["album_fetch"] == 1
    sync_library(conn, client)
    assert calls["album_fetch"] == 1


def test_sync_library_without_tracks(settings, conn):
    handler = make_sync_handler(
        {0: [{"id": "al1", "name": "A", "artistId": "a1"}]},
        artist_index=[],
    )
    client = sync_client(settings, handler)
    stats = sync_library(conn, client, with_tracks=False)
    assert stats["tracks"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM tracks").fetchone()["n"] == 0


def test_sync_library_max_albums(settings, conn):
    handler = make_sync_handler(
        {0: [{"id": "al1", "name": "A"}, {"id": "al2", "name": "B"}, {"id": "al3", "name": "C"}]},
        artist_index=[],
    )
    client = sync_client(settings, handler)
    stats = sync_library(conn, client, with_tracks=False, max_albums=2)
    assert stats["albums"] == 2
    assert conn.execute("SELECT COUNT(*) AS n FROM albums").fetchone()["n"] == 2


def test_sync_library_creates_placeholder_artist_for_album(settings, conn):
    handler = make_sync_handler(
        {0: [{"id": "al1", "name": "A", "artist": "Orphan", "artistId": "aX"}]},
        artist_index=[],
    )
    client = sync_client(settings, handler)
    sync_library(conn, client, with_tracks=False)
    artist = conn.execute("SELECT * FROM artists WHERE navidrome_id = 'aX'").fetchone()
    assert artist is not None
    assert artist["name"] == "Orphan"
    album = conn.execute("SELECT * FROM albums WHERE navidrome_id='al1'").fetchone()
    assert album["artist_id"] == "artist:aX"


def test_sync_library_album_without_artist_id(settings, conn):
    handler = make_sync_handler({0: [{"id": "al1", "name": "A"}]}, artist_index=[])
    client = sync_client(settings, handler)
    stats = sync_library(conn, client, with_tracks=False)
    assert stats["albums"] == 1
    album = conn.execute("SELECT * FROM albums WHERE navidrome_id='al1'").fetchone()
    assert album["artist_id"] is None


def test_sync_library_track_with_different_artist(settings, conn):
    handler = make_sync_handler(
        {0: [{"id": "al1", "name": "A", "artistId": "a1"}]},
        artist_index=[{"artist": [{"id": "a1", "name": "Main"}]}],
        album_details={
            "al1": {
                "id": "al1",
                "song": [
                    {"id": "t1", "title": "Feat", "albumId": "al1", "artistId": "a2"},
                    {"id": "t2", "title": "Main", "albumId": "al1", "artistId": "a1"},
                ],
            }
        },
    )
    client = sync_client(settings, handler)
    sync_library(conn, client)
    feat = conn.execute("SELECT artist_id FROM tracks WHERE navidrome_id='t1'").fetchone()
    assert feat["artist_id"] == "artist:a2"
    assert (
        conn.execute("SELECT COUNT(*) AS n FROM artists WHERE navidrome_id='a2'").fetchone()["n"]
        == 1
    )
    main = conn.execute("SELECT artist_id FROM tracks WHERE navidrome_id='t2'").fetchone()
    assert main["artist_id"] == "artist:a1"


def test_sync_library_track_unknown_album(settings, conn):
    handler = make_sync_handler(
        {0: [{"id": "al1", "name": "A", "artistId": "a1"}]},
        artist_index=[{"artist": [{"id": "a1", "name": "Main"}]}],
        album_details={
            "al1": {
                "id": "al1",
                "song": [{"id": "t1", "albumId": "other", "album": "Ghost", "artistId": "a1"}],
            }
        },
    )
    client = sync_client(settings, handler)
    sync_library(conn, client)
    ghost = conn.execute("SELECT * FROM albums WHERE navidrome_id='other'").fetchone()
    assert ghost is not None
    assert ghost["name"] == "Ghost"
    track = conn.execute("SELECT album_id FROM tracks WHERE navidrome_id='t1'").fetchone()
    assert track["album_id"] == "album:other"


def test_sync_library_resync_updates_metadata(settings, conn):
    state = {"name": "Old", "year": 1999}

    def handler(request):
        endpoint = request.url.path.rsplit("/", 1)[-1]
        if endpoint == "getArtists.view":
            return envelope({"artists": {"index": []}})
        if endpoint == "getAlbumList2.view":
            offset = int(request.url.params.get("offset", 0))
            if offset == 0:
                return envelope(
                    {
                        "albumList2": {
                            "album": [
                                {
                                    "id": "al1",
                                    "name": state["name"],
                                    "artistId": "a1",
                                    "year": state["year"],
                                }
                            ]
                        }
                    }
                )
            return envelope({"albumList2": {"album": []}})
        return envelope({})

    client = sync_client(settings, handler)
    sync_library(conn, client, with_tracks=False)
    state["name"] = "New"
    state["year"] = 2020
    sync_library(conn, client, with_tracks=False)
    album = conn.execute("SELECT * FROM albums WHERE navidrome_id='al1'").fetchone()
    assert album["name"] == "New"
    assert album["year"] == 2020


def test_sync_library_artist_rename(settings, conn):
    state = {"name": "Old", "album": []}

    def handler(request):
        endpoint = request.url.path.rsplit("/", 1)[-1]
        if endpoint == "getArtists.view":
            return envelope(
                {"artists": {"index": [{"artist": [{"id": "a1", "name": state["name"]}]}]}}
            )
        if endpoint == "getAlbumList2.view":
            return envelope({"albumList2": {"album": state["album"]}})
        return envelope({})

    client = sync_client(settings, handler)
    sync_library(conn, client)
    state["name"] = "Nuevo"
    sync_library(conn, client)
    artist = conn.execute("SELECT name FROM artists WHERE navidrome_id='a1'").fetchone()
    assert artist["name"] == "Nuevo"


def test_sync_library_duplicate_artist_in_index(settings, conn):
    handler = make_sync_handler(
        {},
        artist_index=[
            {"artist": [{"id": "a1", "name": "First"}]},
            {"artist": [{"id": "a1", "name": "Second"}]},
        ],
    )
    client = sync_client(settings, handler)
    stats = sync_library(conn, client, with_tracks=False)
    assert stats["artists"] == 2
    assert conn.execute("SELECT COUNT(*) AS n FROM artists").fetchone()["n"] == 1
    artist = conn.execute("SELECT name FROM artists WHERE navidrome_id='a1'").fetchone()
    assert artist["name"] == "Second"


def test_sync_library_empty_library(settings, conn):
    client = sync_client(settings, make_sync_handler({}, artist_index=[]))
    stats = sync_library(conn, client)
    assert stats == {"artists": 0, "albums": 0, "tracks": 0}


def test_sync_track_without_album_id(settings, conn):
    import httpx

    from app.subsonic import SubsonicClient, sync_library

    def handler(request):
        endpoint = request.url.path.rsplit("/", 1)[-1]
        if endpoint == "getArtists.view":
            return httpx.Response(
                200, json={"subsonic-response": {"status": "ok", "artists": {"index": []}}}
            )
        if endpoint == "getAlbumList2.view":
            offset = int(request.url.params.get("offset", 0))
            payload = [{"id": "al1", "name": "A", "artistId": "a1"}] if offset == 0 else []
            return httpx.Response(
                200, json={"subsonic-response": {"status": "ok", "albumList2": {"album": payload}}}
            )
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
            return httpx.Response(
                200, json={"subsonic-response": {"status": "ok", "artists": {"index": []}}}
            )
        if endpoint == "getAlbumList2.view":
            offset = int(request.url.params.get("offset", 0))
            payload = [{"id": "al1", "name": "A", "artistId": "a1"}] if offset == 0 else []
            return httpx.Response(
                200, json={"subsonic-response": {"status": "ok", "albumList2": {"album": payload}}}
            )
        return httpx.Response(
            200,
            json={
                "subsonic-response": {
                    "status": "ok",
                    "album": {
                        "id": "al1",
                        "song": [{"id": "t1", "albumId": "al1", "artistId": ""}],
                    },
                }
            },
        )

    client = SubsonicClient(settings)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    sync_library(conn, client)
    track = conn.execute("SELECT artist_id FROM tracks WHERE navidrome_id='t1'").fetchone()
    # sin artista propio, hereda el artista del álbum
    assert track["artist_id"] == "artist:a1"


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
                        "artists": {"index": [{"artist": [{"id": "a2", "name": "Guest"}]}]},
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


def test_get_lyrics_parses_value(settings, monkeypatch):
    from app.subsonic import SubsonicClient

    client = SubsonicClient(settings)
    monkeypatch.setattr(
        client,
        "_request",
        lambda endpoint, **kwargs: {"lyrics": {"value": "la letra"}},
    )
    assert client.get_lyrics("A", "T") == "la letra"
    client.close()


def test_get_lyrics_empty(settings, monkeypatch):
    from app.subsonic import SubsonicClient

    client = SubsonicClient(settings)
    monkeypatch.setattr(client, "_request", lambda endpoint, **kwargs: {})
    assert client.get_lyrics("A", "T") == ""
    client.close()


def test_get_lyrics_by_song_id_structured(settings, monkeypatch):
    from app.subsonic import SubsonicClient

    payload = {
        "lyricsList": {
            "structuredLyrics": [
                {"line": [{"value": "linea 1"}, {"value": "linea 2"}]},
                {"line": [{"value": "linea 3"}]},
            ]
        }
    }
    client = SubsonicClient(settings)
    monkeypatch.setattr(client, "_request", lambda endpoint, **kwargs: payload)
    assert client.get_lyrics_by_song_id("t1") == "linea 1\nlinea 2\nlinea 3"
    client.close()


def test_get_lyrics_by_song_id_empty_and_garbage(settings, monkeypatch):
    from app.subsonic import SubsonicClient

    client = SubsonicClient(settings)
    monkeypatch.setattr(client, "_request", lambda endpoint, **kwargs: {})
    assert client.get_lyrics_by_song_id("t1") == ""
    monkeypatch.setattr(
        client,
        "_request",
        lambda endpoint, **kwargs: {"lyricsList": {"structuredLyrics": [{"line": []}, "basura"]}},
    )
    assert client.get_lyrics_by_song_id("t1") == ""
    client.close()
