from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import db as db_mod
from app import main as main_mod
from app.web.i18n import detect_lang, translations
from tests.conftest import FakeSubsonic, seed_library


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from app.config import get_settings

    get_settings.cache_clear()
    db_mod._conn = None
    test_client = TestClient(main_mod.app)
    yield test_client
    test_client.close()
    db_mod._conn = None
    get_settings.cache_clear()


@pytest.fixture()
def seeded(client):
    conn = db_mod.get_conn()
    db_mod.init_db(conn)
    seed_library(conn)
    return conn


# ------------------------------------------------------------ i18n

def test_translations_spanish_default():
    t = translations("es")
    assert t["library"].startswith("📚")


def test_translations_english():
    assert translations("en")["library"].startswith("📚")


def test_translations_unknown_falls_back():
    assert translations("fr") == translations("es")


def test_detect_lang_from_header():
    class FakeRequest:
        headers = {"accept-language": "en-US,en;q=0.9"}
        query_params: dict = {}

    assert detect_lang(FakeRequest()) == "en"


def test_detect_lang_default_es():
    class FakeRequest:
        headers = {"accept-language": "es-AR"}
        query_params: dict = {}

    assert detect_lang(FakeRequest()) == "es"


def test_detect_lang_query_override():
    class FakeRequest:
        headers = {"accept-language": "en"}
        query_params = {"lang": "es"}

    assert detect_lang(FakeRequest()) == "es"


def test_detect_lang_broken_request():
    class Broken:
        @property
        def headers(self):
            raise RuntimeError("no headers")

    assert detect_lang(Broken()) == "es"


def test_all_strings_have_same_keys():
    assert set(translations("es")) == set(translations("en"))


# ------------------------------------------------------------ páginas

def test_dashboard(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Bardo" in response.text


def test_dashboard_english(client):
    response = client.get("/", headers={"accept-language": "en-US"})
    assert "Library" in response.text


def test_dashboard_status_indicators(client, monkeypatch):
    async def ok(self):
        return True

    monkeypatch.setattr("app.ollama.OllamaClient.ping", ok)

    class OkSubsonic(FakeSubsonic):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("app.subsonic.SubsonicClient", lambda s: OkSubsonic())
    response = client.get("/")
    assert response.status_code == 200


def test_library_page(client, seeded):
    response = client.get("/library")
    assert response.status_code == 200


def test_run_detail(client, seeded):
    conn = seeded
    cur = conn.execute(
        "INSERT INTO runs(module, kind, started_at, stats) VALUES ('janitor', 'pretend', 'now', '{\"a\": 1}')"
    )
    run_id = cur.lastrowid
    conn.execute(
        "INSERT INTO import_log(run_id, file, old_tags, new_tags, match_score, status) "
        "VALUES (?, '/music/a.flac', '{\"artist\": \"?\"}', '{\"artist\": \"X\"}', 0.9, 'APPLY')",
        (run_id,),
    )
    conn.commit()
    response = client.get(f"/library/runs/{run_id}")
    assert response.status_code == 200
    assert "/music/a.flac" in response.text
    assert "artist" in response.text


def test_run_detail_corrupt_json(client, seeded):
    conn = seeded
    cur = conn.execute(
        "INSERT INTO runs(module, started_at, stats) VALUES ('janitor', 'now', 'no-json')"
    )
    run_id = cur.lastrowid
    conn.execute(
        "INSERT INTO import_log(run_id, file, old_tags, new_tags, status) "
        "VALUES (?, '/a', 'no-json', 'tambien malo', 'X')",
        (run_id,),
    )
    conn.commit()
    response = client.get(f"/library/runs/{run_id}")
    assert response.status_code == 200


def test_run_detail_404(client):
    assert client.get("/library/runs/99999").status_code == 404


def test_library_fix_starts_background(client, monkeypatch):
    called = {}

    def fake_janitor(**kwargs):
        called.update(kwargs)

        class Result:
            status = "ok"

            def as_dict(self):
                return {}

        return Result()

    monkeypatch.setattr("app.janitor.runner.run_janitor", fake_janitor)
    response = client.post("/library/fix", data={"pretend": "true"})
    assert response.status_code == 200
    assert "simulación" in response.text


def test_library_fix_real_mode(client, monkeypatch):
    monkeypatch.setattr(
        "app.janitor.runner.run_janitor",
        lambda **kwargs: type("R", (), {"status": "ok", "as_dict": lambda self: {}})(),
    )
    response = client.post("/library/fix", data={})
    assert response.status_code == 200
    assert "real" in response.text


def test_library_fix_background_error_logged(client, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("janitor roto")

    monkeypatch.setattr("app.janitor.runner.run_janitor", boom)
    response = client.post("/library/fix", data={"pretend": "true"})
    assert response.status_code == 200


# ------------------------------------------------------------ playlists

def test_playlists_page(client, monkeypatch):
    class OkSubsonic(FakeSubsonic):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("app.subsonic.SubsonicClient", lambda s: OkSubsonic())
    response = client.get("/playlists")
    assert response.status_code == 200


def test_playlists_page_subsonic_down(client, monkeypatch):
    class BadSubsonic(FakeSubsonic):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("app.subsonic.SubsonicClient", lambda s: BadSubsonic(fail=True))
    response = client.get("/playlists")
    assert response.status_code == 200


def test_playlist_generate_preview(client, seeded, monkeypatch):
    async def fake_generate(conn, prompt, **kwargs):
        return {
            "playlist_name": "Taberna",
            "track_ids": ["t1", "t2"],
            "reasoning": "ok",
            "tool_calls": 2,
            "trace": [{"tool": "search_candidates", "arguments": {"query": prompt}}],
            "created_playlists": [],
            "validation": [],
            "saved": False,
            "mode": "agent",
        }

    class OkSubsonic(FakeSubsonic):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("app.subsonic.SubsonicClient", lambda s: OkSubsonic())
    monkeypatch.setattr(main_mod, "generate_playlist", fake_generate)
    response = client.post(
        "/playlists/generate", data={"prompt": "taberna", "use_agent": "true"}
    )
    assert response.status_code == 200
    assert "Taberna" in response.text
    assert 'name="track_ids" value="t1,t2"' in response.text


def test_playlist_generate_error(client, seeded, monkeypatch):
    async def fail(conn, prompt, **kwargs):
        raise RuntimeError("ollama caído")

    class OkSubsonic(FakeSubsonic):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("app.subsonic.SubsonicClient", lambda s: OkSubsonic())
    monkeypatch.setattr(main_mod, "generate_playlist", fail)
    response = client.post("/playlists/generate", data={"prompt": "x"})
    assert response.status_code == 200
    assert "ollama caído" in response.text


def test_playlist_save_ok(client, monkeypatch):
    class OkSubsonic(FakeSubsonic):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("app.subsonic.SubsonicClient", lambda s: OkSubsonic())
    response = client.post(
        "/playlists/save", data={"name": "P", "track_ids": "t1, t2 ,t3"}
    )
    assert response.status_code == 200
    assert "Navidrome" in response.text


def test_playlist_save_empty_ids(client):
    response = client.post(
        "/playlists/save", data={"name": "P", "track_ids": " , , "}
    )
    assert response.status_code == 200
    assert "track_ids" in response.text


def test_playlist_save_error(client, monkeypatch):
    class BadSubsonic(FakeSubsonic):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("app.subsonic.SubsonicClient", lambda s: BadSubsonic(fail=True))
    response = client.post("/playlists/save", data={"name": "P", "track_ids": "t1"})
    assert response.status_code == 200
    assert "subsonic down" in response.text


# ------------------------------------------------------------ settings

def test_settings_page(client):
    response = client.get("/settings")
    assert response.status_code == 200


def test_settings_page_shows_models(client, monkeypatch):
    async def models(self):
        return ["gemma4:cloud", "nomic-embed-text:v1.5"]

    monkeypatch.setattr("app.ollama.OllamaClient.list_models", models)
    response = client.get("/settings")
    assert "gemma4:cloud" in response.text


def test_settings_save_persists(client):
    response = client.post(
        "/settings/save",
        data={"subsonic_url": "http://new:4533", "embed_dim": "512"},
    )
    assert response.status_code == 200
    conn = db_mod.get_conn()
    stored = db_mod.all_settings(conn)
    assert stored["subsonic_url"] == "http://new:4533"
    assert stored["embed_dim"] == "512"
    from app.config import runtime_settings

    assert runtime_settings(conn).subsonic_url == "http://new:4533"


def test_settings_save_ignores_unknown_fields(client):
    client.post("/settings/save", data={"campo_malo": "x"})
    conn = db_mod.get_conn()
    assert "campo_malo" not in db_mod.all_settings(conn)


def test_settings_post_redirect(client):
    response = client.post("/settings", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/settings"


# ------------------------------------------------------------ API

def test_api_health(client):
    body = client.get("/api/health").json()
    assert body["version"]
    assert "library" in body
    assert "janitor_enabled" in body


def test_api_synonyms(client):
    body = client.get("/api/synonyms").json()
    assert "joda" in body["synonyms"]


def test_api_sync(client, monkeypatch):
    def fake_sync(conn, client, with_tracks=True):
        return {"artists": 1, "albums": 1, "tracks": 1}

    class OkSubsonic(FakeSubsonic):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("app.subsonic.sync_library", fake_sync)
    monkeypatch.setattr("app.subsonic.SubsonicClient", lambda s: OkSubsonic())
    body = client.post("/api/sync").json()
    assert body == {"artists": 1, "albums": 1, "tracks": 1}


def test_static_css(client):
    assert client.get("/static/bardo.css").status_code == 200


def test_check_subsonic_false_on_error(monkeypatch):
    monkeypatch.setattr(
        "app.subsonic.SubsonicClient",
        lambda s: (_ for _ in ()).throw(RuntimeError("x")),
    )
    from app.config import get_settings

    assert main_mod._check_subsonic(get_settings()) is False


def test_list_models_error_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "app.ollama.OllamaClient.list_models",
        lambda self: (_ for _ in ()).throw(RuntimeError("x")),
    )
    from app.config import get_settings

    assert main_mod._list_models(get_settings()) == []
