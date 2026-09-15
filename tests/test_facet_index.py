from __future__ import annotations

import json

from app.db import (
    facet_list,
    facet_rebuild_all,
    facet_search,
    facet_upsert,
)
from tests.conftest import seed_library


def test_facet_upsert_and_list(conn):
    facet_upsert(
        conn,
        "album",
        "al1",
        {"genres": ["Folk Metal"], "language": "Español", "country": "AR"},
    )
    conn.commit()
    genres = facet_list(conn, "album", "genres")
    assert genres[0]["value"] == "folk metal"
    assert genres[0]["n"] == 1
    langs = facet_list(conn, "album", "languages")
    assert langs[0]["value"] == "es"


def test_facet_upsert_replaces(conn):
    facet_upsert(conn, "album", "al1", {"genres": ["Rock"]})
    facet_upsert(conn, "album", "al1", {"genres": ["Pop"]})
    conn.commit()
    genres = facet_list(conn, "album", "genres")
    assert [g["value"] for g in genres] == ["pop"]


def test_facet_upsert_dedupes(conn):
    facet_upsert(conn, "album", "al1", {"genres": ["Rock", "rock", "ROCK"]})
    conn.commit()
    assert len(facet_list(conn, "album", "genres")) == 1


def test_facet_search_single_facet(conn):
    facet_upsert(conn, "album", "al1", {"languages": ["es"]})
    facet_upsert(conn, "album", "al2", {"languages": ["en"]})
    conn.commit()
    assert facet_search(conn, entity_type="album", facets={"languages": ["es"]}) == {"al1"}
    assert facet_search(
        conn, entity_type="album", facets={"languages": ["es", "en"]}
    ) == {"al1", "al2"}


def test_facet_search_multi_facet_requires_all(conn):
    facet_upsert(conn, "album", "al1", {"languages": ["es"], "countries": ["AR"]})
    facet_upsert(conn, "album", "al2", {"languages": ["es"], "countries": ["MX"]})
    conn.commit()
    result = facet_search(
        conn, entity_type="album", facets={"languages": ["es"], "countries": ["AR"]}
    )
    assert result == {"al1"}


def test_facet_search_numeric_range(conn):
    facet_upsert(conn, "album", "al1", {"energy": 0.9})
    facet_upsert(conn, "album", "al2", {"energy": 0.2})
    conn.commit()
    high = facet_search(conn, entity_type="album", numeric={"energy": (0.7, None)})
    assert high == {"al1"}
    low = facet_search(conn, entity_type="album", numeric={"energy": (None, 0.3)})
    assert low == {"al2"}
    mid = facet_search(conn, entity_type="album", numeric={"energy": (0.0, 1.0)})
    assert mid == {"al1", "al2"}


def test_facet_search_numeric_missing_bounds(conn):
    facet_upsert(conn, "album", "al1", {"energy": 0.5})
    conn.commit()
    assert facet_search(conn, entity_type="album", numeric={"energy": (None, None)}) == set()


def test_facet_search_empty_filters(conn):
    facet_upsert(conn, "album", "al1", {"genres": ["rock"]})
    conn.commit()
    assert facet_search(conn, entity_type="album") == set()
    assert facet_search(conn, entity_type="album", facets={"genres": []}) == set()


def test_facet_search_type_isolation(conn):
    facet_upsert(conn, "album", "al1", {"languages": ["es"]})
    facet_upsert(conn, "track", "al1", {"languages": ["en"]})
    conn.commit()
    assert facet_search(conn, entity_type="track", facets={"languages": ["es"]}) == set()


def test_facet_rebuild_all(conn):
    seed_library(conn)
    conn.execute(
        """
        INSERT INTO fichas(entity_type, entity_id, facets, description, confidence, source, content_hash)
        VALUES ('artist', 'a1', ?, 'd', 0.9, 'llm', 'h')
        """,
        (json.dumps({"genres": ["Folk Metal"], "language": "es", "country": "AR"}),),
    )
    conn.commit()
    count = facet_rebuild_all(conn)
    assert count == 2
    assert facet_list(conn, "artist", "languages")[0]["value"] == "es"
    assert facet_list(conn, "artist", "countries")[0]["value"] == "AR"
    assert facet_list(conn, "artist", "genres")[0]["value"] == "folk metal"
    album_moods = {m["value"] for m in facet_list(conn, "album", "moods")}
    assert "fiesta" in album_moods


def test_facet_rebuild_handles_corrupt(conn):
    conn.execute(
        """
        INSERT INTO fichas(entity_type, entity_id, facets, description, confidence, source, content_hash)
        VALUES ('artist', 'x', 'no-json', 'd', 0.9, 'llm', 'h')
        """
    )
    conn.commit()
    assert facet_rebuild_all(conn) == 1


def test_save_ficha_updates_facet_index(conn):
    from app.enrich.artist import Ficha, save_ficha

    save_ficha(
        conn,
        Ficha("artist", "a1", {"genres": ["Rock"], "language": "en"}, "d", 0.9, "llm", "h"),
    )
    assert facet_list(conn, "artist", "languages")[0]["value"] == "en"
    save_ficha(
        conn,
        Ficha("artist", "a1", {"genres": ["Pop"], "language": "es"}, "d", 0.9, "llm", "h2"),
    )
    assert facet_list(conn, "artist", "languages")[0]["value"] == "es"
    assert [g["value"] for g in facet_list(conn, "artist", "genres")] == ["pop"]


def test_init_db_migrates_from_v1(tmp_path, monkeypatch):
    from app import db as db_mod
    from app.config import get_settings

    monkeypatch.setenv("EMBED_DIM", "768")
    get_settings.cache_clear()
    conn = db_mod.connect(tmp_path / "mig.db")
    conn.executescript(db_mod.SCHEMA)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES ('schema_version', '1')"
    )
    conn.execute(
        """
        INSERT INTO fichas(entity_type, entity_id, facets, description, confidence, source, content_hash)
        VALUES ('artist', 'a1', '{"language": "es"}', 'd', 0.9, 'llm', 'h')
        """
    )
    conn.commit()
    db_mod.init_db(conn)
    assert facet_list(conn, "artist", "languages")[0]["value"] == "es"
    conn.close()
    get_settings.cache_clear()
