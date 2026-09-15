from __future__ import annotations

import pytest

from app.agent.retrieval import FacetFilters, _facet_numeric_params
from app.agent.runner import recall_candidates
from app.enrich.canonicalize import Canonicalizer, canonical_facets
from app.enrich.merge import merge_hard_facets
from tests.conftest import FakeOllama, seed_library


# ------------------------------------------------------------ retrieval gaps

def test_as_list_ignores_non_list_types():
    from app.agent.retrieval import _as_list

    assert _as_list({"no": "dict"}) == []
    assert _as_list(7) == []
    assert _as_list([1, 2]) == ["1", "2"]


def test_facet_numeric_params_low_only():
    params = _facet_numeric_params("energy", 0.5, None)
    assert params == ["energy", 0.5, "energy", 0.5, "energy", 0.5]


def test_merge_filters_all_legacy_params():
    from app.agent.retrieval import _merge_filters

    filters = FacetFilters(exclude_genres=["pop"])
    merged = _merge_filters(filters, 1980, 2000, ["rock", "pop"])
    assert merged.year_min == 1980
    assert merged.year_max == 2000
    assert merged.exclude_genres == ["pop", "rock"]


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


# ------------------------------------------------------------ canonicalize gaps

def test_canonical_facets_removes_invalid_language():
    out = canonical_facets({"language": "klingon"}, Canonicalizer())
    assert "language" not in out


def test_canonical_facets_multi_languages():
    out = canonical_facets({"language": ["es", "english"]}, Canonicalizer())
    assert out["language"] == "es"
    assert out["languages"] == ["es", "en"]


def test_canonical_facets_removes_invalid_country_and_energy():
    out = canonical_facets({"country": "Narnia", "energy": "muchísima"}, Canonicalizer())
    assert "country" not in out
    assert "energy" not in out


def test_canonical_facets_multi_countries():
    out = canonical_facets({"country": ["AR", "br"]}, Canonicalizer())
    assert out["country"] == "AR"
    assert out["countries"] == ["AR", "BR"]


def test_canonical_facets_drops_non_list_languages_field():
    out = canonical_facets({"languages": "es"}, Canonicalizer())
    assert "languages" not in out
    assert out["language"] == "es"


def test_canonical_facets_drops_non_list_countries_field():
    out = canonical_facets({"countries": "AR"}, Canonicalizer())
    assert "countries" not in out
    assert out["country"] == "AR"


def test_canonical_facets_normalizes_energy_ranges():
    assert canonical_facets({"energy": 8}, Canonicalizer())["energy"] == 0.8
    assert canonical_facets({"energy": "alta"}, Canonicalizer())["energy"] == 0.75


# ------------------------------------------------------------ merge gaps

def test_merge_hard_facets_multi_language():
    out = merge_hard_facets({}, language=["es", "en"])
    assert out["language"] == "es"
    assert out["languages"] == ["es", "en"]


# ------------------------------------------------------------ pipeline gaps

def test_merge_lastfm_missing_ficha_is_noop(conn):
    from app.enrich.pipeline import _merge_lastfm_into_ficha

    _merge_lastfm_into_ficha(conn, "artist", "no-existe", ["tag"])
    assert conn.execute("SELECT COUNT(*) AS n FROM fichas").fetchone()["n"] == 0


def test_merge_lastfm_into_existing_ficha(conn):
    from app.enrich.artist import Ficha, get_ficha, save_ficha
    from app.enrich.pipeline import _merge_lastfm_into_ficha

    save_ficha(conn, Ficha("artist", "a1", {"genres": []}, "d", 0.9, "llm", "h"))
    _merge_lastfm_into_ficha(conn, "artist", "a1", ["folk metal", "drinking song"])
    ficha = get_ficha(conn, "artist", "a1")
    assert ficha["facets"]["lastfm_tags"] == ["folk metal", "drinking song"]
    assert ficha["facets"]["genres"] == ["folk metal", "drinking song"]
    # no debe perder descripción/confianza
    assert ficha["description"] == "d"
    assert ficha["confidence"] == 0.9


# ------------------------------------------------------------ prompts gaps

def test_rerank_messages_with_moods_and_reference():
    from app.enrich.prompts import rerank_messages

    messages = rerank_messages(
        "taberna", [{"track_id": "t1"}], 10, moods=["fiesta"], reference="tolkien"
    )
    content = messages[1]["content"]
    assert "Moods buscados: fiesta" in content
    assert "Referencia del mundo: tolkien" in content


def test_rerank_messages_without_extras():
    from app.enrich.prompts import rerank_messages

    content = rerank_messages("taberna", [], 5)[1]["content"]
    assert "Moods" not in content
    assert "Referencia" not in content


# ------------------------------------------------------------ vocab gap

def test_decade_of_out_of_range_string():
    from app.enrich.vocab import decade_of

    assert decade_of("5000") is None
    assert decade_of(5000) is None


# ------------------------------------------------------------ cli facets

def test_cmd_facets_rebuild(monkeypatch, capsys, tmp_path):
    from app import cli

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from app.config import get_settings

    get_settings.cache_clear()
    code = cli.main(["facets", "--rebuild"])
    out = capsys.readouterr().out
    assert code == 0
    assert "rebuilt" in out


def test_cmd_facets_summary(monkeypatch, capsys, tmp_path):
    from app import cli
    from app.db import facet_upsert, get_conn, init_db

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from app.config import get_settings

    get_settings.cache_clear()
    conn = get_conn()
    init_db(conn)
    facet_upsert(conn, "track", "t1", {"languages": ["es"]})
    conn.commit()
    code = cli.main(["facets", "--entity-type", "track"])
    out = capsys.readouterr().out
    assert code == 0
    assert '"es"' in out
    assert '"languages"' in out


# ------------------------------------------------------------ heredad con fichas reales

@pytest.mark.anyio
async def test_pipeline_inherits_language_and_country(conn, settings):
    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    conn.execute("UPDATE albums SET genre = ''")
    conn.commit()
    # Orden real del pipeline: artistas y álbumes ORDER BY name.
    # Artistas: Blind Guardian (a2), Wind Rose (a1).
    # Álbumes: Nightfall in Middle-Earth (al2), Wintersaga (al1).
    ollama = FakeOllama(
        chat_responses=[
            {"name": "Blind Guardian", "genres": [], "moods": [], "description": "d", "confidence": 0.8},
            {
                "name": "Wind Rose",
                "genres": ["folk metal"],
                "moods": ["epico"],
                "language": "Italiano",
                "country": "Italia",
                "energy": "alta",
                "description": "d",
                "confidence": 0.9,
            },
            {"artist": "Blind Guardian", "album": "Nightfall", "themes": [], "moods": [], "description": "d", "confidence": 0.8},
            {
                "artist": "Wind Rose",
                "album": "Wintersaga",
                "themes": ["enanos"],
                "moods": ["fiesta"],
                "language": "en",
                "decades": ["2010s"],
                "description": "d",
                "confidence": 0.9,
            },
        ]
    )
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=ollama,
        lastfm=type("L", (), {"fetch_for": lambda *a, **k: [], "close": lambda self: None})(),
        force=True,
    )
    assert result["tracks"] == 4
    from app.enrich.artist import get_ficha

    track = get_ficha(conn, "track", "t1")
    assert track["facets"]["language"] == "en"
    assert track["facets"]["country"] == "IT"
    assert track["facets"]["energy"] == 0.75
    other = get_ficha(conn, "track", "t3")
    assert other is not None
    assert other["facets"].get("country") in (None, "DE")
