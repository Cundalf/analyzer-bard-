from __future__ import annotations

import sqlite3

import pytest

from app import db as db_mod


def test_init_creates_all_tables(conn):
    names = {
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        ).fetchall()
    }
    expected = {
        "meta",
        "settings",
        "artists",
        "albums",
        "tracks",
        "fichas",
        "runs",
        "import_log",
        "synonyms",
        "lastfm_cache",
        "web_cache",
        "fts_entities",
        "vec_fichas",
    }
    assert expected <= names


def test_init_idempotent(conn):
    db_mod.init_db(conn)
    db_mod.init_db(conn)
    version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[
        "value"
    ]
    assert version == str(db_mod.SCHEMA_VERSION)


def test_synonyms_seeded_once(conn):
    n1 = conn.execute("SELECT COUNT(*) AS n FROM synonyms").fetchone()["n"]
    assert n1 == len(db_mod.DEFAULT_SYNONYMS)
    db_mod.init_db(conn)
    n2 = conn.execute("SELECT COUNT(*) AS n FROM synonyms").fetchone()["n"]
    assert n2 == n1


def test_vec_dim_change_recreates_table(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("EMBED_DIM", "768")
    get_settings.cache_clear()
    c = db_mod.connect(tmp_path / "a.db")
    db_mod.init_db(c)
    db_mod.vec_upsert(c, "artist", "x", [0.0] * 768)
    assert c.execute("SELECT COUNT(*) AS n FROM vec_fichas").fetchone()["n"] == 1

    monkeypatch.setenv("EMBED_DIM", "1024")
    get_settings.cache_clear()
    db_mod.init_db(c)
    assert c.execute("SELECT COUNT(*) AS n FROM vec_fichas").fetchone()["n"] == 0
    dim = c.execute("SELECT value FROM meta WHERE key = 'vec_dim'").fetchone()["value"]
    assert dim == "1024"
    c.close()
    get_settings.cache_clear()


def test_vec_dim_same_keeps_data(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("EMBED_DIM", "768")
    get_settings.cache_clear()
    c = db_mod.connect(tmp_path / "b.db")
    db_mod.init_db(c)
    db_mod.vec_upsert(c, "artist", "x", [0.0] * 768)
    db_mod.init_db(c)
    assert c.execute("SELECT COUNT(*) AS n FROM vec_fichas").fetchone()["n"] == 1
    c.close()
    get_settings.cache_clear()


def test_vec_upsert_replaces_same_id(conn):
    db_mod.vec_upsert(conn, "track", "t1", [1.0] * 768)
    db_mod.vec_upsert(conn, "track", "t1", [0.5] * 768)
    rows = conn.execute("SELECT COUNT(*) AS n FROM vec_fichas").fetchone()["n"]
    assert rows == 1


def test_vec_search_type_filter(conn):
    db_mod.vec_upsert(conn, "track", "t1", [1.0] * 768)
    db_mod.vec_upsert(conn, "album", "al1", [1.0] * 768)
    all_hits = db_mod.vec_search(conn, [1.0] * 768, limit=10)
    assert {h["entity_type"] for h in all_hits} == {"track", "album"}
    only_album = db_mod.vec_search(conn, [1.0] * 768, limit=10, entity_type="album")
    assert len(only_album) == 1
    assert only_album[0]["entity_id"] == "al1"


def test_vec_search_empty(conn):
    assert db_mod.vec_search(conn, [0.0] * 768, limit=5) == []


def test_fts_upsert_replaces(conn):
    db_mod.fts_upsert(conn, "album", "al1", "folk metal enanos")
    db_mod.fts_upsert(conn, "album", "al1", "power metal dragones")
    hits = db_mod.fts_search(conn, "enanos")
    assert hits == []
    hits2 = db_mod.fts_search(conn, "dragones")
    assert len(hits2) == 1


def test_fts_search_empty_query(conn):
    assert db_mod.fts_search(conn, "") == []
    assert db_mod.fts_search(conn, "   ") == []
    assert db_mod.fts_search(conn, '""') == []


def test_fts_search_quotes_are_stripped(conn):
    db_mod.fts_upsert(conn, "artist", "a1", "wind rose folk")
    hits = db_mod.fts_search(conn, '"wind" rose')
    assert hits


def test_fts_search_type_filter(conn):
    db_mod.fts_upsert(conn, "artist", "a1", "wind rose")
    db_mod.fts_upsert(conn, "album", "al1", "wind saga")
    hits = db_mod.fts_search(conn, "wind", entity_type="album")
    assert [h["entity_type"] for h in hits] == ["album"]


def test_fts_search_diacritics(conn):
    db_mod.fts_upsert(conn, "artist", "a1", "señor de los anillos")
    assert db_mod.fts_search(conn, "senor")
    assert db_mod.fts_search(conn, "SEÑOR")


def test_fts_prefix_match(conn):
    db_mod.fts_upsert(conn, "artist", "a1", "nightwish symphonic")
    assert db_mod.fts_search(conn, "sympho")


def test_runs_lifecycle(conn):
    run_id = db_mod.start_run(conn, "janitor", "import")
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row["status"] == "running"
    assert row["started_at"]
    db_mod.finish_run(conn, run_id, "ok", {"x": 1})
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row["status"] == "ok"
    assert row["finished_at"]
    assert '"x": 1' in row["stats"]


def test_finish_run_default_stats(conn):
    run_id = db_mod.start_run(conn, "enrich")
    db_mod.finish_run(conn, run_id)
    row = conn.execute("SELECT stats FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row["stats"] == "{}"


def test_settings_get_set_and_update(conn):
    assert db_mod.all_settings(conn) == {}
    db_mod.set_setting(conn, "a", "1")
    db_mod.set_setting(conn, "b", "2")
    assert db_mod.all_settings(conn) == {"a": "1", "b": "2"}
    db_mod.set_setting(conn, "a", "9")
    assert db_mod.all_settings(conn)["a"] == "9"


def test_all_settings_missing_table():
    raw = sqlite3.connect(":memory:")
    raw.row_factory = sqlite3.Row
    assert db_mod.all_settings(raw) == {}
    raw.close()


def test_session_commits(conn, monkeypatch):
    monkeypatch.setattr(db_mod, "get_conn", lambda: conn)
    with db_mod.session() as s:
        s.execute("INSERT INTO settings(key, value) VALUES ('k', 'v')")
    assert db_mod.all_settings(conn)["k"] == "v"


def test_session_rolls_back(conn, monkeypatch):
    monkeypatch.setattr(db_mod, "get_conn", lambda: conn)
    with pytest.raises(RuntimeError):
        with db_mod.session() as s:
            s.execute("INSERT INTO settings(key, value) VALUES ('k2', 'v')")
            raise RuntimeError("boom")
    assert "k2" not in db_mod.all_settings(conn)


def test_connect_with_explicit_path(tmp_path):
    c = db_mod.connect(tmp_path / "explicit.db")
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    c.close()


def test_get_conn_singleton(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    from app.config import get_settings

    get_settings.cache_clear()
    db_mod._conn = None
    a = db_mod.get_conn()
    b = db_mod.get_conn()
    assert a is b
    db_mod._conn = None
    get_settings.cache_clear()


def test_utcnow_format():
    value = db_mod.utcnow()
    assert "T" in value and "+00:00" in value


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
