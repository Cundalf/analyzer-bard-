from __future__ import annotations

import sqlite3

from app.agent.retrieval import FacetFilters, search_by_terms, search_fts, search_vectors
from app.db import facet_upsert, fts_upsert, vec_upsert
from tests.conftest import seed_library


def seed_facet_library(conn: sqlite3.Connection) -> None:
    """Biblioteca con idiomas/países/décadas/energía variados."""
    seed_library(conn)
    # Wind Rose (folk metal, italiano, en inglés, 2019, enérgico)
    facet_upsert(
        conn,
        "artist",
        "a1",
        {"language": "en", "country": "IT", "energy": 0.9, "genres": ["Folk Metal"]},
    )
    # Blind Guardian (power metal, alemán, en inglés, 1998)
    facet_upsert(
        conn,
        "artist",
        "a2",
        {"language": "en", "country": "DE", "energy": 0.8, "genres": ["Power Metal"]},
    )
    # álbum latino en español para probar filtro de idioma
    conn.execute(
        "INSERT INTO artists(id, navidrome_id, name) VALUES ('artist:a3', 'a3', 'Soda Stereo')"
    )
    conn.execute(
        """
        INSERT INTO albums(id, navidrome_id, artist_id, name, year, genre)
        VALUES ('album:al3', 'al3', 'artist:a3', 'Signos', 1986, 'Rock en Español')
        """
    )
    conn.execute(
        """
        INSERT INTO tracks(id, navidrome_id, album_id, artist_id, title, duration)
        VALUES ('track:t5', 't5', 'album:al3', 'artist:a3', 'Persiana Americana', 260)
        """
    )
    facet_upsert(
        conn,
        "artist",
        "a3",
        {"language": "es", "country": "AR", "energy": 0.6, "genres": ["Rock en Español"]},
    )
    facet_upsert(conn, "album", "al3", {"language": "es", "country": "AR", "decades": ["80s"]})
    # instrumental
    conn.execute(
        """
        INSERT INTO tracks(id, navidrome_id, album_id, artist_id, title, duration)
        VALUES ('track:t6', 't6', 'album:al3', 'artist:a3', 'Intro Instrumental', 90)
        """
    )
    facet_upsert(conn, "track", "t6", {"is_instrumental": True, "language": "es"})
    conn.commit()


# ------------------------------------------------------------ languages

def test_filter_languages_tracks_via_artist(conn):
    seed_facet_library(conn)
    filters = FacetFilters(languages=["es"])
    results = search_by_terms(conn, ["rock"], filters=filters)
    assert {r["track_id"] for r in results} == {"t5", "t6"}


def test_filter_languages_excludes_others(conn):
    seed_facet_library(conn)
    filters = FacetFilters(languages=["en"])
    results = search_by_terms(conn, ["metal"], filters=filters)
    assert {r["track_id"] for r in results} == {"t1", "t2", "t3", "t4"}


def test_filter_languages_via_album(conn):
    seed_facet_library(conn)
    # sólo el álbum al3 tiene facets de idioma; sus tracks deben matchear
    filters = FacetFilters(languages=["es"])
    results = search_by_terms(conn, ["persiana"], filters=filters)
    assert {r["track_id"] for r in results} == {"t5"}


def test_filter_language_no_match(conn):
    seed_facet_library(conn)
    filters = FacetFilters(languages=["ja"])
    assert search_by_terms(conn, ["metal", "rock"], filters=filters) == []


# ------------------------------------------------------------ countries

def test_filter_countries(conn):
    seed_facet_library(conn)
    filters = FacetFilters(countries=["AR"])
    results = search_by_terms(conn, ["rock"], filters=filters)
    assert {r["track_id"] for r in results} == {"t5", "t6"}


def test_filter_countries_multiple(conn):
    seed_facet_library(conn)
    filters = FacetFilters(countries=["AR", "DE"])
    results = search_by_terms(conn, ["metal", "rock"], filters=filters)
    assert {"t3", "t4", "t5", "t6"} <= {r["track_id"] for r in results}


# ------------------------------------------------------------ decades

def test_filter_decades_album(conn):
    seed_facet_library(conn)
    filters = FacetFilters(decades=[1980])
    results = search_by_terms(conn, ["rock"], filters=filters)
    assert {r["track_id"] for r in results} == {"t5", "t6"}


def test_filter_decades_no_match(conn):
    seed_facet_library(conn)
    filters = FacetFilters(decades=[2020])
    assert search_by_terms(conn, ["metal", "rock"], filters=filters) == []


# ------------------------------------------------------------ energy

def test_filter_energy_min(conn):
    seed_facet_library(conn)
    filters = FacetFilters(energy_min=0.85)
    results = search_by_terms(conn, ["metal", "rock"], filters=filters)
    assert {r["track_id"] for r in results} == {"t1", "t2"}


def test_filter_energy_max_only(conn):
    seed_facet_library(conn)
    filters = FacetFilters(energy_max=0.7)
    results = search_by_terms(conn, ["metal", "rock"], filters=filters)
    assert {r["track_id"] for r in results} == {"t5", "t6"}


def test_filter_energy_range(conn):
    seed_facet_library(conn)
    filters = FacetFilters(energy_min=0.5, energy_max=0.7)
    results = search_by_terms(conn, ["rock"], filters=filters)
    assert {r["track_id"] for r in results} == {"t5", "t6"}


# ------------------------------------------------------------ genres

def test_filter_include_genres(conn):
    seed_facet_library(conn)
    filters = FacetFilters(include_genres=["power metal"])
    results = search_by_terms(conn, ["nightfall"], filters=filters)
    assert {r["track_id"] for r in results} == {"t3", "t4"}


def test_filter_include_genres_via_facet_not_album_tag(conn):
    seed_facet_library(conn)
    # el álbum al1 tiene genre 'folk metal' en el espejo; al2 'power metal'
    filters = FacetFilters(include_genres=["folk metal"])
    results = search_by_terms(conn, ["wintersaga"], filters=filters)
    assert {r["track_id"] for r in results} == {"t1", "t2"}


def test_filter_exclude_genres_via_facet(conn):
    seed_facet_library(conn)
    filters = FacetFilters(exclude_genres=["power metal"])
    results = search_by_terms(conn, ["metal"], filters=filters)
    assert all(r["album"] != "Nightfall in Middle-Earth" for r in results)


# ------------------------------------------------------------ flags

def test_filter_instrumental_true(conn):
    seed_facet_library(conn)
    filters = FacetFilters(instrumental=True)
    results = search_by_terms(conn, ["intro", "persiana"], filters=filters)
    assert {r["track_id"] for r in results} == {"t6"}


def test_filter_instrumental_false(conn):
    seed_facet_library(conn)
    filters = FacetFilters(instrumental=False)
    results = search_by_terms(conn, ["intro", "persiana"], filters=filters)
    assert {r["track_id"] for r in results} == {"t5"}


def test_filter_ballad(conn):
    seed_facet_library(conn)
    facet_upsert(conn, "track", "t5", {"is_ballad": True})
    conn.commit()
    filters = FacetFilters(ballad=True)
    results = search_by_terms(conn, ["persiana"], filters=filters)
    assert {r["track_id"] for r in results} == {"t5"}


def test_filter_concept_album(conn):
    seed_facet_library(conn)
    facet_upsert(conn, "album", "al2", {"is_concept_album": True})
    conn.commit()
    filters = FacetFilters(concept_album=True)
    results = search_by_terms(conn, ["nightfall"], filters=filters)
    assert {r["track_id"] for r in results} == {"t3", "t4"}


# ------------------------------------------------------------ combinaciones

def test_filters_combined(conn):
    seed_facet_library(conn)
    filters = FacetFilters(languages=["es"], countries=["AR"], decades=[1980])
    results = search_by_terms(conn, ["rock"], filters=filters)
    assert {r["track_id"] for r in results} == {"t5", "t6"}


def test_filters_combined_no_match(conn):
    seed_facet_library(conn)
    filters = FacetFilters(languages=["es"], countries=["DE"])
    assert search_by_terms(conn, ["rock"], filters=filters) == []


def test_year_and_facet_filters_together(conn):
    seed_facet_library(conn)
    filters = FacetFilters(languages=["es"], year_min=2000)
    assert search_by_terms(conn, ["rock"], filters=filters) == []


# ------------------------------------------------------------ fts y vectores

def test_search_fts_applies_facet_filters(conn):
    seed_facet_library(conn)
    fts_upsert(conn, "track", "t5", "persiana americana rock latino")
    fts_upsert(conn, "track", "t3", "nightfall power metal")
    conn.commit()
    es = search_fts(conn, "persiana nightfall", filters=FacetFilters(languages=["es"]))
    assert {r["track_id"] for r in es} == {"t5"}
    en = search_fts(conn, "persiana nightfall", filters=FacetFilters(languages=["en"]))
    assert {r["track_id"] for r in en} == {"t3"}


def test_search_vectors_applies_facet_filters(conn):
    seed_facet_library(conn)
    vec_upsert(conn, "track", "t5", [1.0] * 768)
    vec_upsert(conn, "track", "t3", [1.0] * 768)
    results = search_vectors(
        conn, [1.0] * 768, filters=FacetFilters(languages=["es"])
    )
    assert {r["track_id"] for r in results} == {"t5"}


# ------------------------------------------------------------ from_dict

def test_facet_filters_from_dict_full():
    filters = FacetFilters.from_dict(
        {
            "year_min": "1980",
            "year_max": 1999,
            "exclude_genres": ["pop"],
            "include_genres": ["rock"],
            "languages": ["Español", "english"],
            "countries": ["argentina", "DE"],
            "decades": ["80s", 1990],
            "energy_min": "alta",
            "energy_max": 0.99,
            "instrumental": "true",
            "ballad": "no",
            "concept_album": True,
        }
    )
    assert filters.year_min == 1980
    assert filters.year_max == 1999
    assert filters.exclude_genres == ["pop"]
    assert filters.include_genres == ["rock"]
    assert filters.languages == ["es", "en"]
    assert filters.countries == ["AR", "DE"]
    assert filters.decades == [1980, 1990]
    assert filters.energy_min == 0.75
    assert filters.energy_max == 0.99
    assert filters.instrumental is True
    assert filters.ballad is False
    assert filters.concept_album is True


def test_facet_filters_from_dict_empty_and_garbage():
    assert FacetFilters.from_dict(None) == FacetFilters()
    filters = FacetFilters.from_dict(
        {
            "year_min": "no-num",
            "languages": None,
            "instrumental": "quizas",
            "decades": None,
        }
    )
    assert filters.year_min is None
    assert filters.languages == []
    assert filters.instrumental is None
    assert filters.decades == []


def test_facet_filters_from_dict_single_string_genre():
    # un string suelto se acepta como un único valor
    filters = FacetFilters.from_dict({"exclude_genres": "no-lista"})
    assert filters.exclude_genres == ["no-lista"]


def test_facet_filters_from_dict_strings_lists():
    filters = FacetFilters.from_dict(
        {
            "exclude_genres": "pop",
            "include_genres": ["rock", "  "],
            "languages": "espanol",
            "countries": "AR",
        }
    )
    assert filters.exclude_genres == ["pop"]
    assert filters.include_genres == ["rock"]
    assert filters.languages == ["es"]
    assert filters.countries == ["AR"]


def test_facet_filters_sql_empty():
    assert FacetFilters().sql() == ("", [])


def test_facet_filters_sql_all():
    filters = FacetFilters(
        year_min=1980,
        year_max=2000,
        include_genres=["rock"],
        exclude_genres=["pop"],
        languages=["es"],
        countries=["AR"],
        decades=[1980],
        energy_min=0.5,
        energy_max=0.9,
        instrumental=True,
        ballad=False,
        concept_album=True,
    )
    sql, params = filters.sql()
    assert sql.startswith(" AND ")
    assert sql.count("EXISTS") >= 6
    assert params


def test_search_candidates_note_when_filters_too_strict(settings, conn):
    import asyncio

    from app.agent.tools import ToolContext, execute_tool
    from app.enrich.canonicalize import Canonicalizer
    from tests.conftest import FakeOllama

    seed_facet_library(conn)
    ctx = ToolContext(
        conn=conn,
        ollama=FakeOllama(),
        settings=settings,
        canon=Canonicalizer.from_db(conn),
    )
    out = asyncio.run(
        execute_tool(
            "search_candidates",
            {"query": "zzzz", "languages": ["es"], "energy_min": 0.99},
            ctx,
        )
    )
    assert out["count"] == 0
    assert "note" in out
    assert "filtro" in out["note"]


def test_search_candidates_no_note_with_results(settings, conn):
    import asyncio

    from app.agent.tools import ToolContext, execute_tool
    from app.enrich.canonicalize import Canonicalizer
    from tests.conftest import FakeOllama

    seed_facet_library(conn)
    ctx = ToolContext(
        conn=conn,
        ollama=FakeOllama(),
        settings=settings,
        canon=Canonicalizer.from_db(conn),
    )
    out = asyncio.run(
        execute_tool(
            "search_candidates", {"query": "rock", "languages": ["es"]}, ctx
        )
    )
    assert out["count"] >= 1
    assert "note" not in out
