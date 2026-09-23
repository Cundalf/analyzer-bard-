from __future__ import annotations

import pytest

from app.enrich.canonicalize import (
    Canonicalizer,
    canonical_facets,
    content_hash,
    ficha_text,
    normalize,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  Fiesta MEDIEVAL ", "fiesta medieval"),
        ("Épico", "epico"),
        ("drinking_song", "drinking song"),
        ("Rock---n---Roll", "rock n roll"),
        ("  many   spaces  ", "many spaces"),
        ("Ñandú", "nandu"),
        ("", ""),
        ("   ", ""),
        ("TABERNA", "taberna"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_canonical_identity():
    canon = Canonicalizer()
    assert canon.canonical("Fiesta") == "fiesta"
    assert canon.canonical("Fiesta") == canon.canonical("Fiesta")


def test_canonical_chain():
    canon = Canonicalizer({"a": "b", "b": "c"})
    assert canon.canonical("a") == "c"


def test_canonical_cycle_does_not_loop():
    canon = Canonicalizer({"a": "b", "b": "a"})
    assert canon.canonical("a") in {"a", "b"}


def test_canonical_display_preserves_accent():
    canon = Canonicalizer({"tolkien": "señor de los anillos"})
    assert canon.canonical("tolkien") == "señor de los anillos"
    assert canon.canonical("Señor de los Anillos") == "señor de los anillos"


def test_add_updates_display():
    canon = Canonicalizer()
    canon.add("epic", "Épico")
    assert canon.canonical("epic") == "Épico"
    assert canon.canonical("epico") == "Épico"


def test_canonical_list_dedupes_and_skips_empty():
    canon = Canonicalizer({"party": "fiesta"})
    assert canon.canonical_list(["Party", "", "  ", "fiesta"]) == ["fiesta"]


def test_canonical_list_none():
    canon = Canonicalizer()
    assert canon.canonical_list(None) == []


def test_expand_terms_and_synonyms():
    canon = Canonicalizer({"party": "fiesta", "joda": "fiesta"})
    expanded = canon.expand(["fiesta"])
    assert set(expanded) >= {"fiesta", "party", "joda"}


def test_expand_unknown_term():
    canon = Canonicalizer()
    assert canon.expand(["rareza"]) == ["rareza"]


def test_expand_empty():
    canon = Canonicalizer()
    assert canon.expand([]) == []
    assert canon.expand(["", "  "]) == []


def test_expand_dedupes():
    canon = Canonicalizer({"party": "fiesta"})
    expanded = canon.expand(["fiesta", "Fiesta", "party"])
    assert len(expanded) == len(set(expanded))


def test_canonical_facets_all_list_fields():
    canon = Canonicalizer({"epic": "epico", "tolkien": "señor de los anillos"})
    out = canonical_facets(
        {
            "genres": ["Epic", "Folk Metal"],
            "subgenres": ["Celtic"],
            "lyrical_themes": ["War"],
            "moods": ["Epic"],
            "references": ["Tolkien"],
            "themes": ["Battle"],
            "instrumentation": ["Fiddle"],
        },
        canon,
    )
    assert out["genres"] == ["epico", "folk metal"]
    assert out["moods"] == ["epico"]
    assert out["references"][0] == "señor de los anillos"


def test_canonical_facets_ignores_non_lists_and_missing():
    canon = Canonicalizer()
    out = canonical_facets({"moods": "no-lista", "country": "AR"}, canon)
    assert out["moods"] == "no-lista"
    assert out["country"] == "AR"


def test_canonical_facets_does_not_mutate_input():
    canon = Canonicalizer({"epic": "epico"})
    original = {"moods": ["Epic"]}
    canonical_facets(original, canon)
    assert original["moods"] == ["Epic"]


def test_ficha_text_full():
    text = ficha_text(
        {"genres": ["folk metal"], "energy": 0.8, "confidence": 0.9},
        "Folk metal festivo.",
    )
    assert "Folk metal festivo." in text
    assert "genres: folk metal" in text
    assert "energy: 0.8" in text
    assert "confidence" not in text


def test_ficha_text_only_description():
    assert ficha_text({}, "Solo prosa") == "Solo prosa"


def test_ficha_text_only_tags():
    text = ficha_text({"moods": ["epico"]}, "")
    assert text == "moods: epico"


def test_ficha_text_empty_facets():
    assert ficha_text({}, "") == ""
    assert ficha_text(None, None) == ""


def test_ficha_text_skips_zero_energy_and_empty_lists():
    text = ficha_text({"moods": [], "energy": 0.0}, "")
    assert text == ""


def test_content_hash_stable_and_order_independent():
    a = content_hash({"x": [1, 2], "y": "z"})
    b = content_hash({"y": "z", "x": [1, 2]})
    assert a == b
    assert a != content_hash({"x": [1, 3], "y": "z"})


def test_content_hash_unicode():
    assert content_hash({"x": "ñandú"}) == content_hash({"x": "ñandú"})


def test_from_db_and_unknown_synonym(conn):
    canon = Canonicalizer.from_db(conn)
    assert canon.canonical("joda") == "fiesta"
    assert canon.canonical("termino-inexistente") == "termino inexistente"


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
