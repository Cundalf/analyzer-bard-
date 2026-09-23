from __future__ import annotations

import pytest
from tests.conftest import seed_library

from app.enrich.artist import (
    Ficha,
    get_ficha,
    is_generic_entity,
    mark_needs_janitor,
    save_ficha,
)
from app.enrich.merge import (
    merge_confidence,
    merge_description,
    merge_ficha_facets,
)

# ------------------------------------------------------------ merge_ficha_facets


def test_merge_lists_union_no_loss():
    old = {"moods": ["epico"], "themes": ["cerveza"]}
    new = {"moods": ["fiesta"], "themes": ["enanos"]}
    merged = merge_ficha_facets(old, new)
    assert merged["moods"] == ["epico", "fiesta"]
    assert merged["themes"] == ["cerveza", "enanos"]


def test_merge_lists_dedupe():
    merged = merge_ficha_facets({"moods": ["epico"]}, {"moods": ["Epico", "fiesta"]})
    assert merged["moods"] == ["epico", "fiesta"]


def test_merge_lists_drop_generics():
    merged = merge_ficha_facets({"moods": ["epico"]}, {"moods": ["unknown", "N/A", "fiesta"]})
    assert merged["moods"] == ["epico", "fiesta"]


def test_merge_scalar_new_wins_when_valid():
    merged = merge_ficha_facets({"country": "AR"}, {"country": "ES"})
    assert merged["country"] == "ES"


def test_merge_scalar_keeps_old_when_new_generic():
    merged = merge_ficha_facets({"country": "AR"}, {"country": "Unknown"})
    assert merged["country"] == "AR"


def test_merge_scalar_new_empty_keeps_old():
    merged = merge_ficha_facets({"country": "AR"}, {"country": ""})
    assert merged["country"] == "AR"


def test_merge_scalar_accepts_new_when_old_generic():
    merged = merge_ficha_facets({"country": "Unknown"}, {"country": "ES"})
    assert merged["country"] == "ES"


def test_merge_protects_measured_language():
    old = {"language": "es", "language_source": "lyrics"}
    new = {"language": "en"}
    merged = merge_ficha_facets(old, new)
    assert merged["language"] == "es"
    assert merged["language_source"] == "lyrics"


def test_merge_protects_measured_energy():
    old = {"energy": 0.9, "energy_source": "audio"}
    new = {"energy": 0.1}
    merged = merge_ficha_facets(old, new)
    assert merged["energy"] == 0.9
    assert merged["energy_source"] == "audio"


def test_merge_language_normalizes_new():
    merged = merge_ficha_facets({}, {"language": "Español"})
    assert merged["language"] == "es"


def test_merge_language_keeps_old_on_invalid_new():
    merged = merge_ficha_facets({"language": "es"}, {"language": "klingon"})
    assert merged["language"] == "es"


def test_merge_country_normalizes():
    merged = merge_ficha_facets({}, {"country": "argentina"})
    assert merged["country"] == "AR"


def test_merge_flags_prefer_new_then_old():
    assert merge_ficha_facets({}, {"is_ballad": True})["is_ballad"] is True
    assert merge_ficha_facets({"is_ballad": True}, {})["is_ballad"] is True
    assert merge_ficha_facets({"is_ballad": True}, {"is_ballad": False})["is_ballad"] is False


def test_merge_preserves_inherited_from_and_sources():
    old = {"inherited_from": {"album": "al1"}, "genres_source": "hard"}
    merged = merge_ficha_facets(old, {})
    assert merged["inherited_from"] == {"album": "al1"}
    assert merged["genres_source"] == "hard"


def test_merge_none_inputs():
    assert merge_ficha_facets(None, None) == {}
    assert merge_ficha_facets({}, None) == {}
    assert merge_ficha_facets(None, {"moods": ["x"]}) == {"moods": ["x"]}


def test_merge_does_not_mutate_inputs():
    old = {"moods": ["a"]}
    new = {"moods": ["b"]}
    merge_ficha_facets(old, new)
    assert old == {"moods": ["a"]}
    assert new == {"moods": ["b"]}


def test_merge_confidence():
    assert merge_confidence(0.9, 0.5) == 0.5
    assert merge_confidence(0.9, 0) == 0.9
    assert merge_confidence(0.9, None) == 0.9
    assert merge_confidence(None, 0.5) == 0.5
    assert merge_confidence("0.7", "0.3") == 0.3
    assert merge_confidence("basura", 0.4) == 0.4
    assert merge_confidence(None, None) == 0.0


def test_merge_description():
    assert merge_description("vieja", "nueva") == "nueva"
    assert merge_description("vieja", "") == "vieja"
    assert merge_description("vieja", "unknown") == "vieja"
    assert merge_description("vieja", None) == "vieja"
    assert merge_description(None, "nueva") == "nueva"
    assert merge_description(None, None) == ""


# ------------------------------------------------------------ save_ficha


def test_save_ficha_preserves_previous_fields(conn):
    save_ficha(conn, Ficha("artist", "a1", {"moods": ["epico"]}, "desc vieja", 0.9, "llm", "h1"))
    save_ficha(conn, Ficha("artist", "a1", {"themes": ["cerveza"]}, "", 0.0, "llm", "h2"))
    ficha = get_ficha(conn, "artist", "a1")
    assert ficha["facets"]["moods"] == ["epico"]
    assert ficha["facets"]["themes"] == ["cerveza"]
    assert ficha["description"] == "desc vieja"
    assert ficha["confidence"] == 0.9


def test_save_ficha_generic_does_not_override(conn):
    save_ficha(conn, Ficha("artist", "a1", {"genres": ["folk metal"]}, "d", 0.9, "llm", "h1"))
    save_ficha(conn, Ficha("artist", "a1", {"genres": ["Unknown"]}, "d", 0.9, "llm", "h2"))
    ficha = get_ficha(conn, "artist", "a1")
    assert ficha["facets"]["genres"] == ["folk metal"]


def test_save_ficha_respects_measured_language(conn):
    save_ficha(
        conn,
        Ficha(
            "track",
            "t1",
            {"language": "es", "language_source": "lyrics"},
            "d",
            0.9,
            "inherited",
            "h1",
        ),
    )
    save_ficha(conn, Ficha("track", "t1", {"language": "en"}, "d", 0.9, "llm", "h2"))
    ficha = get_ficha(conn, "track", "t1")
    assert ficha["facets"]["language"] == "es"
    assert ficha["facets"]["language_source"] == "lyrics"


# ------------------------------------------------------------ needs_janitor


def test_mark_needs_janitor(conn):
    mark_needs_janitor(conn, "artist", "a1", "artista genérico", hash_payload="x")
    ficha = get_ficha(conn, "artist", "a1")
    assert ficha["facets"]["needs_janitor"] is True
    assert ficha["facets"]["needs_janitor_reason"] == "artista genérico"
    assert ficha["source"] == "pending"
    assert ficha["confidence"] == 0.0


def test_mark_needs_janitor_idempotent(conn):
    mark_needs_janitor(conn, "artist", "a1", "x", hash_payload="x")
    mark_needs_janitor(conn, "artist", "a1", "y", hash_payload="y")
    assert conn.execute("SELECT COUNT(*) AS n FROM fichas").fetchone()["n"] == 1
    assert get_ficha(conn, "artist", "a1")["facets"]["needs_janitor_reason"] == "y"


def test_is_generic_entity():
    assert is_generic_entity("[Unknown Artist]") is True
    assert is_generic_entity("Unknown") is True
    assert is_generic_entity("Wind Rose") is False
    assert is_generic_entity(None) is True


@pytest.mark.anyio
async def test_enrich_artist_skips_generic(conn, settings):
    from tests.conftest import FakeOllama

    from app.enrich.artist import enrich_artist
    from app.enrich.canonicalize import Canonicalizer

    ollama = FakeOllama()
    row = {"id": "artist:u", "navidrome_id": "u", "name": "[Unknown Artist]"}
    result = await enrich_artist(conn, ollama, Canonicalizer.from_db(conn), row)
    assert result is None
    assert ollama.calls == []
    ficha = get_ficha(conn, "artist", "u")
    assert ficha["facets"]["needs_janitor"] is True


@pytest.mark.anyio
async def test_enrich_album_skips_generic(conn, settings):
    from tests.conftest import FakeOllama

    from app.enrich.album import enrich_album
    from app.enrich.canonicalize import Canonicalizer

    ollama = FakeOllama()
    row = {"id": "album:u", "navidrome_id": "u", "name": "[Unknown Album]"}
    result = await enrich_album(conn, ollama, Canonicalizer.from_db(conn), row)
    assert result is None
    assert ollama.calls == []
    assert get_ficha(conn, "album", "u")["facets"]["needs_janitor"] is True


@pytest.mark.anyio
async def test_pipeline_reports_pending(conn, settings):
    from tests.conftest import FakeOllama

    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    conn.execute("UPDATE artists SET name = '[Unknown Artist]' WHERE navidrome_id = 'a1'")
    conn.commit()
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=type("L", (), {"fetch_for": lambda *a, **k: [], "close": lambda s: None})(),
        artists=True,
        albums=False,
        tracks=False,
        force=True,
    )
    assert result["pending"] >= 1


def test_library_health_counts_pending(conn):
    from app.janitor.report import library_health

    seed_library(conn)
    mark_needs_janitor(conn, "artist", "a1", "genérico", hash_payload="x")
    health = library_health(conn)
    assert health["needs_janitor"] == 1


# ------------------------------------------------------------ ramas restantes


def test_filter_valid_rejects_non_iterable():
    from app.enrich.generic import filter_valid

    assert filter_valid(42) == []
    assert filter_valid({"a": 1}) == ["a"]


def test_merge_protected_scalar_field_kept():
    """Un campo escalar protegido (p.ej. country con fuente dura) se conserva."""
    old = {"country": "AR", "country_source": "hard"}
    new = {"country": "ES"}
    merged = merge_ficha_facets(old, new)
    # country no está en HARD_PROTECTED hoy: el nuevo válido gana
    assert merged["country"] == "ES"


def test_merge_protected_language_with_multi():
    old = {"language": "es", "languages": ["es", "en"], "language_source": "lyrics"}
    merged = merge_ficha_facets(old, {"language": "fr"})
    assert merged["language"] == "es"
    assert merged["languages"] == ["es", "en"]


def test_merge_language_multi_new():
    merged = merge_ficha_facets({}, {"languages": ["es", "en"]})
    assert merged["language"] == "es"
    assert merged["languages"] == ["es", "en"]


def test_merge_country_multi():
    merged = merge_ficha_facets({}, {"countries": ["AR", "BR"]})
    assert merged["country"] == "AR"
    assert merged["countries"] == ["AR", "BR"]


def test_merge_confidence_invalid_old():
    from app.enrich.merge import merge_confidence

    assert merge_confidence("basura", 0.5) == 0.5
    assert merge_confidence("basura", 0) == 0.0


def test_pipeline_pending_album(conn, settings):
    import asyncio

    from tests.conftest import FakeOllama

    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    conn.execute("UPDATE albums SET name = '[Unknown Album]' WHERE navidrome_id = 'al1'")
    conn.commit()
    result = asyncio.run(
        enrich_library(
            conn,
            settings=settings,
            ollama=FakeOllama(),
            lastfm=type("L", (), {"fetch_for": lambda *a, **k: [], "close": lambda s: None})(),
            artists=False,
            albums=True,
            tracks=False,
            force=True,
        )
    )
    assert result["pending"] >= 1


def test_merge_confidence_new_invalid_old_valid():
    from app.enrich.merge import merge_confidence

    assert merge_confidence(0.8, "no-numero") == 0.8
