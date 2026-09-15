from __future__ import annotations

import pytest

from app.agent import retrieval
from tests.conftest import seed_library


# ------------------------------------------------------------ is_unknown

@pytest.mark.parametrize(
    "value,expected",
    [
        ("[Unknown Artist]", True),
        ("[unknown album]", True),
        ("Unknown Artist", True),
        ("unknown album", True),
        ("Various Artists", True),
        ("various artists", True),
        ("", True),
        ("   ", True),
        (None, True),
        ("Wind Rose", False),
        ("The Unknowns", False),
        ("Unknown Mortal Orchestra", False),
    ],
)
def test_is_unknown(value, expected):
    assert retrieval.is_unknown(value) is expected


# ------------------------------------------------------------ search_by_terms

def test_search_by_terms_empty_terms(conn):
    assert retrieval.search_by_terms(conn, []) == []


def test_search_by_terms_matches_title(conn):
    seed_library(conn)
    results = retrieval.search_by_terms(conn, ["dwarves"])
    assert any(r["title"] == "Drunken Dwarves" for r in results)


def test_search_by_terms_matches_album(conn):
    seed_library(conn)
    results = retrieval.search_by_terms(conn, ["wintersaga"])
    assert {r["track_id"] for r in results} == {"t1", "t2"}


def test_search_by_terms_matches_artist(conn):
    seed_library(conn)
    results = retrieval.search_by_terms(conn, ["blind guardian"])
    assert all(r["artist"] == "Blind Guardian" for r in results)


def test_search_by_terms_matches_genre(conn):
    seed_library(conn)
    results = retrieval.search_by_terms(conn, ["folk metal"])
    assert {r["track_id"] for r in results} == {"t1", "t2"}


def test_search_by_terms_matches_facets(conn):
    seed_library(conn)
    results = retrieval.search_by_terms(conn, ["cerveza"])
    assert {r["track_id"] for r in results} == {"t1", "t2"}


def test_search_by_terms_strips_whitespace(conn):
    seed_library(conn)
    assert retrieval.search_by_terms(conn, ["  dwarves  "])


def test_search_by_terms_year_filters(conn):
    seed_library(conn)
    assert retrieval.search_by_terms(conn, ["night"], year_min=2000) == []
    assert retrieval.search_by_terms(conn, ["night"], year_max=2000)
    assert retrieval.search_by_terms(conn, ["night"], year_min=1990, year_max=2000)


def test_search_by_terms_year_filter_keeps_null_year(conn):
    conn.execute(
        "INSERT INTO tracks(id, navidrome_id, title) VALUES ('track:nn', 'nn', 'noyear night')"
    )
    conn.commit()
    results = retrieval.search_by_terms(conn, ["night"], year_min=2050)
    assert {r["track_id"] for r in results} == {"nn"}


def test_search_by_terms_exclude_genres(conn):
    seed_library(conn)
    results = retrieval.search_by_terms(
        conn, ["metal"], exclude_genres=["power metal"]
    )
    assert results
    assert all(r["genre"] != "power metal" for r in results)


def test_search_by_terms_limit(conn):
    seed_library(conn)
    assert len(retrieval.search_by_terms(conn, ["metal"], limit=1)) == 1


def test_search_by_terms_no_results(conn):
    seed_library(conn)
    assert retrieval.search_by_terms(conn, ["zzzz-no-existe"]) == []


def test_search_by_terms_or_semantics(conn):
    seed_library(conn)
    results = retrieval.search_by_terms(conn, ["dwarves", "nightfall"])
    ids = {r["track_id"] for r in results}
    assert ids == {"t1", "t3", "t4"}


# ------------------------------------------------------------ fts

def test_search_fts_hits_album_ficha(conn):
    seed_library(conn)
    results = retrieval.search_fts(conn, "enanos cerveza")
    assert {r["track_id"] for r in results} == {"t1", "t2"}
    assert all(r["source"] == "fts" for r in results)


def test_search_fts_no_hits(conn):
    seed_library(conn)
    assert retrieval.search_fts(conn, "zzz") == []


def test_search_fts_hits_track_ficha(conn):
    from app.db import fts_upsert

    seed_library(conn)
    fts_upsert(conn, "track", "t3", "balada epica dragones")
    conn.commit()
    results = retrieval.search_fts(conn, "dragones")
    assert {r["track_id"] for r in results} == {"t3"}


def test_search_fts_hits_artist_ficha(conn):
    from app.db import fts_upsert

    seed_library(conn)
    fts_upsert(conn, "artist", "a2", "power metal sinfonico")
    conn.commit()
    results = retrieval.search_fts(conn, "sinfonico")
    assert {r["track_id"] for r in results} == {"t3", "t4"}


# ------------------------------------------------------------ vectors

def test_search_vectors_orders_by_score(conn):
    from app.db import vec_upsert

    seed_library(conn)
    vec_upsert(conn, "album", "al1", [1.0] * 768)
    vec_upsert(conn, "album", "al2", [0.0] * 768)
    results = retrieval.search_vectors(conn, [1.0] * 768, limit=5)
    assert results[0]["album"] == "Wintersaga"
    assert results[0]["source"] == "vector"
    assert results[0]["score"] > results[-1]["score"]


def test_search_vectors_track_direct_hit(conn):
    from app.db import vec_upsert

    seed_library(conn)
    vec_upsert(conn, "track", "t3", [1.0] * 768)
    results = retrieval.search_vectors(conn, [1.0] * 768, limit=5)
    assert results[0]["track_id"] == "t3"
    assert results[0]["score"] == pytest.approx(1.0)


def test_search_vectors_artist_hit_drills_down(conn):
    from app.db import vec_upsert

    seed_library(conn)
    vec_upsert(conn, "artist", "a1", [1.0] * 768)
    results = retrieval.search_vectors(conn, [1.0] * 768, limit=5)
    assert {r["track_id"] for r in results} == {"t1", "t2"}


def test_search_vectors_empty(conn):
    seed_library(conn)
    assert retrieval.search_vectors(conn, [0.1] * 768, limit=5) == []


def test_search_vectors_only_unknown_entity(conn):
    from app.db import vec_upsert

    seed_library(conn)
    vec_upsert(conn, "album", "no-existe", [1.0] * 768)
    assert retrieval.search_vectors(conn, [1.0] * 768, limit=5) == []


# ------------------------------------------------------------ rrf

def test_rrf_merge_prefers_consensus(conn):
    seed_library(conn)
    a = retrieval.search_by_terms(conn, ["dwarves"])
    b = retrieval.search_fts(conn, "enanos")
    merged = retrieval.rrf_merge([a, b])
    assert merged[0]["track_id"] == "t1"
    assert merged[0]["source"] == "hybrid"


def test_rrf_merge_empty_sets():
    assert retrieval.rrf_merge([]) == []
    assert retrieval.rrf_merge([[], []]) == []


def test_rrf_merge_skips_blank_ids():
    merged = retrieval.rrf_merge([[{"track_id": "", "title": "x"}]])
    assert merged == []


def test_rrf_merge_keeps_first_occurrence_metadata():
    first = {"track_id": "t1", "title": "First", "moods": [], "themes": []}
    second = {"track_id": "t1", "title": "Second", "moods": [], "themes": []}
    merged = retrieval.rrf_merge([[first], [second]])
    assert merged[0]["title"] == "First"


# ------------------------------------------------------------ hydrate

def test_hydrate_invalid_facets_json(conn):
    seed_library(conn)
    conn.execute("UPDATE fichas SET facets = 'no-json' WHERE entity_id = 'al1'")
    conn.commit()
    results = retrieval.search_by_terms(conn, ["dwarves"])
    assert results[0]["moods"] == []
    assert results[0]["themes"] == []


def test_hydrate_track_ficha_takes_precedence(conn):
    seed_library(conn)
    conn.execute(
        """
        INSERT INTO fichas(entity_type, entity_id, facets, description, confidence, source, content_hash)
        VALUES ('track', 't1', '{"moods": ["propio"], "themes": ["track"]}', 'desc track', 0.9, 'llm', 'h')
        """
    )
    conn.commit()
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["moods"] == ["propio"]
    assert target["themes"] == ["track"]
    assert target["description"] == "desc track"


def test_hydrate_artist_fallback(conn):
    from app.enrich.artist import Ficha, save_ficha

    seed_library(conn)
    conn.execute("DELETE FROM fichas WHERE entity_type='album'")
    save_ficha(
        conn,
        Ficha("artist", "a1", {"moods": ["del artista"], "references": ["ref"]}, "", 0.9, "llm", "h"),
    )
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["moods"] == ["del artista"]
    assert target["themes"] == ["ref"]


def test_candidate_fields_defaults(conn):
    conn.execute(
        "INSERT INTO tracks(id, navidrome_id, title) VALUES ('track:x', 'x', 'X')"
    )
    conn.commit()
    results = retrieval.search_by_terms(conn, ["X"])
    assert results[0]["artist"] == ""
    assert results[0]["album"] == ""
    assert results[0]["year"] is None
    assert results[0]["score"] == 0.0
