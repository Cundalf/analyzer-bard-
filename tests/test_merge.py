from __future__ import annotations

from app.enrich.merge import (
    apply_hard_genre_to_artist,
    merge_hard_facets,
)
from app.enrich.track import inherit_track_ficha


# ------------------------------------------------------------ merge_hard_facets

def test_merge_hard_genre_replaces_llm():
    # En el flujo real, la ficha ya pasó por canonical_facets (minúsculas).
    facets = {"genres": ["folk metal"], "moods": ["epico"]}
    out = merge_hard_facets(facets, genre="Rock en Español")
    assert out["genres"][0] == "Rock en Español"
    assert "folk metal" in out["genres"]
    assert out["moods"] == ["epico"]


def test_merge_hard_genre_adds_when_missing():
    out = merge_hard_facets({}, genre="Tango")
    assert out["genres"] == ["Tango"]


def test_merge_hard_genre_dedupes_case_insensitive():
    out = merge_hard_facets({"genres": ["rock"]}, genre="ROCK")
    assert len([g for g in out["genres"] if g.lower() == "rock"]) == 1


def test_merge_lastfm_tags_as_fallback():
    out = merge_hard_facets({}, lastfm_tags=["drinking song", "folk metal"])
    assert out["genres"] == ["drinking song", "folk metal"]
    assert out["lastfm_tags"] == ["drinking song", "folk metal"]


def test_merge_lastfm_does_not_override_llm_genres():
    out = merge_hard_facets({"genres": ["Tango"]}, lastfm_tags=["unknown tag"])
    assert out["genres"] == ["Tango"]
    assert out["lastfm_tags"] == ["unknown tag"]


def test_merge_language_hard_wins():
    out = merge_hard_facets({"language": "en"}, language="Español")
    assert out["language"] == "es"


def test_merge_language_multi():
    out = merge_hard_facets({}, language="portuguese")
    assert out["language"] == "pt"


def test_merge_normalizes_country_and_energy():
    out = merge_hard_facets({"country": "argentina", "energy": 0.7})
    assert out["country"] == "AR"
    assert out["energy"] == 0.7


def test_merge_does_not_mutate_input():
    original = {"genres": ["Rock"], "moods": ["x"]}
    merge_hard_facets(original, genre="Pop", language="es")
    assert original == {"genres": ["Rock"], "moods": ["x"]}


def test_merge_empty_inputs():
    out = merge_hard_facets({}, genre="", lastfm_tags=[], language=None)
    assert out == {}


def test_merge_cleans_whitespace_and_empty_tags():
    out = merge_hard_facets({}, genre="  Rock   Nacional ", lastfm_tags=["", "  "])
    assert out["genres"] == ["Rock Nacional"]
    assert "lastfm_tags" not in out


# ------------------------------------------------------------ apply_hard_genre

def test_apply_hard_genre_prioritizes_llm_canonical_form():
    # La ficha del LLM llega canonicalizada: "power metal".
    facets = {"genres": ["folk metal", "power metal"]}
    out = apply_hard_genre_to_artist(facets, ["power metal", "nacional"])
    assert out["genres"][0] == "power metal"
    assert "folk metal" in out["genres"]
    assert "nacional" in out["genres"]
    assert out["genres_source"] == "hard"


def test_apply_hard_genre_empty_returns_copy():
    facets = {"genres": ["Rock"]}
    out = apply_hard_genre_to_artist(facets, [])
    assert out == facets
    assert out is not facets


def test_apply_hard_genre_with_canonicalizer():
    from app.enrich.canonicalize import Canonicalizer

    canon = Canonicalizer({"rock nacional": "rock"})
    out = apply_hard_genre_to_artist({}, ["Rock Nacional"], canon)
    assert out["genres"] == ["rock"]


def test_apply_hard_genre_limits_to_12():
    out = apply_hard_genre_to_artist({}, [f"g{i}" for i in range(30)])
    assert len(out["genres"]) == 12


def test_apply_hard_genre_no_llm_genres():
    out = apply_hard_genre_to_artist({"moods": ["x"]}, ["Tango"])
    assert out["genres"] == ["Tango"]
    assert out["moods"] == ["x"]


def test_apply_hard_genre_non_list_genres():
    out = apply_hard_genre_to_artist({"genres": "no-lista"}, ["Tango"])
    assert out["genres"] == ["Tango"]


def test_apply_hard_genre_ignores_generic_tags():
    out = apply_hard_genre_to_artist({"genres": ["folk metal"]}, ["Unknown", "N/A"])
    assert out["genres"] == ["folk metal"]
    # todos los tags eran genéricos: no hay dato duro, no se marca la fuente
    assert "genres_source" not in out


def test_merge_hard_facets_ignores_generic_genre():
    out = merge_hard_facets({"genres": ["rock"]}, genre="[Unknown Artist]")
    assert out["genres"] == ["rock"]


def test_merge_hard_facets_ignores_generic_tags():
    out = merge_hard_facets({}, lastfm_tags=["unknown", "N/A", "rock"])
    assert out["genres"] == ["rock"]
    assert out["lastfm_tags"] == ["rock"]


# ------------------------------------------------------------ herencia completa

def make_album_ficha(facets: dict) -> dict:
    return {
        "entity_id": "al1",
        "facets": facets,
        "description": "Album desc",
        "confidence": 0.9,
        "content_hash": "ha",
    }


def make_artist_ficha(facets: dict) -> dict:
    return {
        "entity_id": "a1",
        "facets": facets,
        "description": "Artist desc",
        "confidence": 0.8,
        "content_hash": "hb",
    }


def test_inherit_language_country_instrumentation():
    album = make_album_ficha(
        {"language": "es", "country": "AR", "instrumentation": ["guitarra"]}
    )
    artist = make_artist_ficha({"country": "AR", "vocal_style": "suave"})
    track = {"navidrome_id": "t1", "title": "Canción"}
    ficha = inherit_track_ficha(None, track, album, artist)
    assert ficha.facets["language"] == "es"
    assert ficha.facets["country"] == "AR"
    assert ficha.facets["instrumentation"] == ["guitarra"]
    assert ficha.facets["vocal_style"] == "suave"


def test_inherit_album_wins_over_artist_scalar():
    album = make_album_ficha({"language": "es"})
    artist = make_artist_ficha({"language": "en"})
    ficha = inherit_track_ficha(
        None, {"navidrome_id": "t1", "title": "T"}, album, artist
    )
    assert ficha.facets["language"] == "es"


def test_inherit_artist_fills_missing_scalar():
    album = make_album_ficha({"moods": ["x"]})
    artist = make_artist_ficha({"language": "de", "energy": 0.7})
    ficha = inherit_track_ficha(
        None, {"navidrome_id": "t1", "title": "T"}, album, artist
    )
    assert ficha.facets["language"] == "de"
    assert ficha.facets["energy"] == 0.7


def test_inherit_genres_and_concept_flag():
    album = make_album_ficha({"genres": ["Tango"], "is_concept_album": True})
    artist = make_artist_ficha({"for_fans_of": ["Piazzolla"]})
    ficha = inherit_track_ficha(
        None, {"navidrome_id": "t1", "title": "T"}, album, artist
    )
    assert ficha.facets["genres"] == ["Tango"]
    assert ficha.facets["is_concept_album"] is True
    assert ficha.facets["for_fans_of"] == ["Piazzolla"]


def test_inherit_genres_from_subgenres():
    album = make_album_ficha({"subgenres": ["Neo Tango"]})
    ficha = inherit_track_ficha(
        None, {"navidrome_id": "t1", "title": "T"}, album, None
    )
    assert ficha.facets["genres"] == ["Neo Tango"]


def test_inherit_energy_default_zero():
    album = make_album_ficha({"moods": ["x"]})
    ficha = inherit_track_ficha(
        None, {"navidrome_id": "t1", "title": "T"}, album, None
    )
    assert ficha.facets["energy"] == 0.0


def test_inherit_no_optional_fields_absent():
    album = make_album_ficha({})
    ficha = inherit_track_ficha(
        None, {"navidrome_id": "t1", "title": "T"}, album, None
    )
    for field in ("language", "country", "vocal_style", "instrumentation"):
        assert field not in ficha.facets or ficha.facets[field] in ([], "", None)
