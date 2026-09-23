from __future__ import annotations

import pytest
from tests.conftest import seed_library

from app.agent import retrieval
from app.agent.retrieval import (
    FacetFilters,
    _facet_numeric_params,
    _merge_filters,
)

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
    results = retrieval.search_by_terms(conn, ["metal"], exclude_genres=["power metal"])
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
        Ficha(
            "artist", "a1", {"moods": ["del artista"], "references": ["ref"]}, "", 0.9, "llm", "h"
        ),
    )
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["moods"] == ["del artista"]
    assert target["themes"] == ["ref"]


def test_candidate_fields_defaults(conn):
    conn.execute("INSERT INTO tracks(id, navidrome_id, title) VALUES ('track:x', 'x', 'X')")
    conn.commit()
    results = retrieval.search_by_terms(conn, ["X"])
    assert results[0]["artist"] == ""
    assert results[0]["album"] == ""
    assert results[0]["year"] is None
    assert results[0]["score"] == 0.0


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
        conn,
        "dwarves tolkien epico",
        year_min=1990,
        year_max=2020,
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


def test_hydrate_invalid_json_facets(conn):
    seed_library(conn)
    conn.execute("UPDATE fichas SET facets = '{roto' WHERE entity_type='album' AND entity_id='al1'")
    conn.commit()
    from app.agent import retrieval

    results = retrieval.search_by_terms(conn, ["dwarves"])
    assert results[0]["moods"] == []


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


def test_hydrate_artist_only_themes_missing(conn):
    """El álbum aporta moods, pero no themes: el artista completa themes."""
    from app.agent import retrieval
    from app.enrich.artist import Ficha, save_ficha

    seed_library(conn)
    conn.execute("DELETE FROM fichas WHERE entity_type='album'")
    save_ficha(
        conn,
        Ficha(
            "artist",
            "a1",
            {"moods": ["del artista"], "lyrical_themes": ["temas del artista"]},
            "",
            0.9,
            "llm",
            "h",
        ),
    )
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["moods"] == ["del artista"]
    assert target["themes"] == ["temas del artista"]


def test_hydrate_artist_moods_present_themes_from_references(conn):
    from app.agent import retrieval
    from app.enrich.artist import Ficha, save_ficha

    seed_library(conn)
    conn.execute("DELETE FROM fichas WHERE entity_type='album'")
    save_ficha(
        conn,
        Ficha("artist", "a1", {"moods": ["x"], "references": ["ref"]}, "", 0.9, "llm", "h"),
    )
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    # el álbum (borrado) ya no aporta; el artista da moods y references
    assert target["moods"] == ["x"]
    assert target["themes"] == ["ref"]


def test_hydrate_artist_lyrical_themes_short_circuit(conn):
    """lyrical_themes presente: no evalúa references (ramas 93->95)."""
    from app.agent import retrieval
    from app.enrich.artist import Ficha, save_ficha

    seed_library(conn)
    conn.execute("DELETE FROM fichas WHERE entity_type='album'")
    save_ficha(
        conn,
        Ficha(
            "artist",
            "a1",
            {"moods": ["m"], "lyrical_themes": ["letras"], "references": ["ref"]},
            "",
            0.9,
            "llm",
            "h",
        ),
    )
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["themes"] == ["letras"]


def test_hydrate_artist_themes_empty_uses_references(conn):
    """Sin lyrical_themes, cae a references (rama 95->99)."""
    from app.agent import retrieval
    from app.enrich.artist import Ficha, save_ficha

    seed_library(conn)
    conn.execute("DELETE FROM fichas WHERE entity_type='album'")
    save_ficha(
        conn,
        Ficha(
            "artist",
            "a1",
            {"moods": ["m"], "lyrical_themes": [], "references": ["ref"]},
            "",
            0.9,
            "llm",
            "h",
        ),
    )
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["themes"] == ["ref"]


def test_hydrate_track_only_moods_completes_themes(conn):
    """Ficha de track con moods pero sin themes: rama 93->95."""
    from app.agent import retrieval
    from app.enrich.artist import Ficha, save_ficha

    seed_library(conn)
    save_ficha(
        conn,
        Ficha("track", "t1", {"moods": ["solo moods"]}, "d", 0.9, "llm", "h"),
    )
    save_ficha(
        conn,
        Ficha(
            "artist",
            "a1",
            {"moods": ["del artista"], "lyrical_themes": ["letras"]},
            "",
            0.9,
            "llm",
            "h2",
        ),
    )
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["moods"] == ["solo moods"]
    assert target["themes"] == ["letras"]


def test_hydrate_track_only_themes_completes_moods(conn):
    """Ficha de track con themes pero sin moods: rama 95->99."""
    from app.agent import retrieval
    from app.enrich.artist import Ficha, save_ficha

    seed_library(conn)
    save_ficha(
        conn,
        Ficha("track", "t1", {"themes": ["solo themes"]}, "d", 0.9, "llm", "h"),
    )
    save_ficha(
        conn,
        Ficha("artist", "a1", {"moods": ["del artista"]}, "", 0.9, "llm", "h2"),
    )
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["moods"] == ["del artista"]
    assert target["themes"] == ["solo themes"]


def test_hydrate_themes_present_skips_artist_fallback(conn):
    """El álbum aporta moods y themes: no consulta al artista (rama 93->99)."""
    from app.agent import retrieval

    seed_library(conn)
    results = retrieval.search_by_terms(conn, ["dwarves"])
    target = next(r for r in results if r["track_id"] == "t1")
    assert target["moods"] == ["fiesta", "epico"]
    assert target["themes"] == ["cerveza", "enanos"]


def test_as_list_ignores_non_list_types():
    from app.agent.retrieval import _as_list

    assert _as_list({"no": "dict"}) == []
    assert _as_list(7) == []
    assert _as_list([1, 2]) == ["1", "2"]


def test_facet_numeric_params_low_only():
    params = _facet_numeric_params("energy", 0.5, None)
    assert params == ["energy", 0.5, "energy", 0.5, "energy", 0.5]


def test_merge_filters_all_legacy_params():
    filters = FacetFilters(exclude_genres=["pop"])
    merged = _merge_filters(filters, 1980, 2000, ["rock", "pop"])
    assert merged.year_min == 1980
    assert merged.year_max == 2000
    assert merged.exclude_genres == ["pop", "rock"]
