from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.enrich.lastfm import API_ROOT, LastFmClient


def transport(payload=None, status=200, fail=False):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert str(request.url).startswith(API_ROOT)
        assert request.url.params["api_key"] == "key123"
        assert request.url.params["format"] == "json"
        if fail:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(status, json=payload or {})

    return httpx.MockTransport(handler), calls


@pytest.fixture()
def no_sleep(monkeypatch):
    monkeypatch.setattr("app.enrich.lastfm.time.sleep", lambda s: None)


def make_client(settings, t, key="key123") -> LastFmClient:
    client = LastFmClient(settings, api_key=key)
    client._client = httpx.Client(transport=t)
    return client


def test_available_requires_key(settings):
    assert LastFmClient(settings, api_key="x").available is True
    assert LastFmClient(settings, api_key="").available is False


def test_unavailable_returns_empty(settings, monkeypatch):
    monkeypatch.setenv("LASTFM_API_KEY", "")
    client = make_client(settings, transport()[0], key="")
    assert client._get("artist.getTopTags", artist="X") == {}
    client.close()


def test_artist_top_tags_filters_and_limits(settings, no_sleep):
    payload = {
        "toptags": {
            "tag": [
                {"name": "folk metal", "count": 100},
                {"name": "sin count"},
                {"name": "zero", "count": "0"},
                {"name": "power metal", "count": "50"},
                {"name": "", "count": 10},
            ]
        }
    }
    t, _ = transport(payload)
    client = make_client(settings, t)
    tags = client.artist_top_tags("Wind Rose")
    assert tags == ["folk metal", "power metal"]
    client.close()


def test_artist_top_tags_limit(settings, no_sleep):
    payload = {"toptags": {"tag": [{"name": f"t{i}", "count": 1} for i in range(20)]}}
    t, _ = transport(payload)
    client = make_client(settings, t)
    assert len(client.artist_top_tags("X", limit=5)) == 5
    client.close()


def test_track_top_tags(settings, no_sleep):
    payload = {"toptags": {"tag": [{"name": "drinking song", "count": 9}]}}
    t, _ = transport(payload)
    client = make_client(settings, t)
    assert client.track_top_tags("Wind Rose", "Drunken Dwarves") == ["drinking song"]
    client.close()


def test_artist_top_tags_empty_payloads(settings, no_sleep):
    for payload in ({}, {"toptags": {}}, {"toptags": {"tag": None}}):
        t, _ = transport(payload)
        client = make_client(settings, t)
        assert client.artist_top_tags("X") == []
        client.close()


def test_http_error_returns_empty(settings, no_sleep):
    t, _ = transport(fail=True)
    client = make_client(settings, t)
    assert client.artist_top_tags("X") == []
    client.close()


def test_http_status_error_returns_empty(settings, no_sleep):
    t, _ = transport(status=500)
    client = make_client(settings, t)
    assert client.artist_top_tags("X") == []
    client.close()


def test_invalid_json_returns_empty(settings, no_sleep):
    def handler(request):
        return httpx.Response(200, text="<html>nope</html>")

    client = make_client(settings, httpx.MockTransport(handler))
    assert client.artist_top_tags("X") == []
    client.close()


def test_rate_limit_sleeps_between_calls(settings, monkeypatch):
    sleeps = []
    monkeypatch.setattr("app.enrich.lastfm.time.sleep", lambda s: sleeps.append(s))
    t, _ = transport({"toptags": {"tag": []}})
    client = make_client(settings, t)
    client.artist_top_tags("A")
    client.artist_top_tags("B")
    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= 0.25
    client.close()


def test_fetch_for_caches(settings, no_sleep, conn):
    t, calls = transport({"toptags": {"tag": [{"name": "folk", "count": 1}]}})
    client = make_client(settings, t)
    first = client.fetch_for(conn, "artist", "a1", "Wind Rose")
    assert first == ["folk"]
    second = client.fetch_for(conn, "artist", "a1", "Wind Rose")
    assert second == ["folk"]
    assert calls["n"] == 1
    client.close()


def test_fetch_for_track_uses_track_endpoint(settings, no_sleep, conn):
    captured = {}

    def handler(request):
        captured.update(request.url.params)
        return httpx.Response(200, json={"toptags": {"tag": [{"name": "party", "count": 1}]}})

    client = make_client(settings, httpx.MockTransport(handler))
    tags = client.fetch_for(conn, "track", "t1", "Wind Rose", track="Drunken Dwarves")
    assert tags == ["party"]
    assert captured["method"] == "track.getTopTags"
    assert captured["track"] == "Drunken Dwarves"
    client.close()


def test_fetch_for_expired_cache_refetches(settings, no_sleep, conn):
    t, calls = transport({"toptags": {"tag": [{"name": "folk", "count": 1}]}})
    client = make_client(settings, t)
    client.fetch_for(conn, "artist", "a1", "X")
    old = (datetime.now(UTC) - timedelta(days=40)).isoformat(timespec="seconds")
    conn.execute("UPDATE lastfm_cache SET fetched_at = ? WHERE entity_key = 'a1'", (old,))
    conn.commit()
    client.fetch_for(conn, "artist", "a1", "X")
    assert calls["n"] == 2
    client.close()


def test_fetch_for_corrupt_cache_refetches(settings, no_sleep, conn):
    t, calls = transport({"toptags": {"tag": [{"name": "folk", "count": 1}]}})
    client = make_client(settings, t)
    client.fetch_for(conn, "artist", "a1", "X")
    conn.execute("UPDATE lastfm_cache SET fetched_at = 'no-es-fecha' WHERE entity_key='a1'")
    conn.commit()
    client.fetch_for(conn, "artist", "a1", "X")
    assert calls["n"] == 2
    client.close()


def test_fetch_for_corrupt_payload_refetches(settings, no_sleep, conn):
    t, calls = transport({"toptags": {"tag": [{"name": "folk", "count": 1}]}})
    client = make_client(settings, t)
    client.fetch_for(conn, "artist", "a1", "X")
    conn.execute("UPDATE lastfm_cache SET payload = 'no-json' WHERE entity_key='a1'")
    conn.commit()
    assert client.fetch_for(conn, "artist", "a1", "X") == ["folk"]
    assert calls["n"] == 2
    client.close()


def test_fetch_for_empty_tags_cached(settings, no_sleep, conn):
    t, calls = transport({"toptags": {"tag": []}})
    client = make_client(settings, t)
    assert client.fetch_for(conn, "artist", "a1", "X") == []
    assert client.fetch_for(conn, "artist", "a1", "X") == []
    assert calls["n"] == 1
    client.close()
