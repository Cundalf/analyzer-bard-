from __future__ import annotations

import pytest

from app.enrich.vocab import (
    decade_of,
    decades_of,
    display_country,
    display_language,
    facet_values_from_facets,
    normalize_countries,
    normalize_country,
    normalize_energy,
    normalize_language,
    normalize_languages,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Español", "es"),
        ("español", "es"),
        ("CASTELLANO", "es"),
        ("spanish", "es"),
        ("es", "es"),
        ("Argentino", "es"),
        ("Inglés", "en"),
        ("japanese", "ja"),
        ("Português", "pt"),
        ("brasileño", "pt"),
        ("instrumental", "zxx"),
        ("xyz", "xyz"),
        ("", None),
        (None, None),
        ("  ", None),
        ("klingon", None),
    ],
)
def test_normalize_language(raw, expected):
    assert normalize_language(raw) == expected


def test_normalize_language_unknown_iso_like():
    assert normalize_language("nah") == "nah"


def test_normalize_languages_variants():
    assert normalize_languages(["Español", "castellano", "spanish"]) == ["es"]
    assert normalize_languages("ingles") == ["en"]
    assert normalize_languages(None) == []
    assert normalize_languages([]) == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Argentina", "AR"),
        ("argentina", "AR"),
        ("España", "ES"),
        ("spain", "ES"),
        ("eeuu", "US"),
        ("United States", "US"),
        ("Reino Unido", "GB"),
        ("uk", "GB"),
        ("japon", "JP"),
        ("mx", "MX"),
        ("", None),
        ("Narnia", None),
        (None, None),
    ],
)
def test_normalize_country(raw, expected):
    assert normalize_country(raw) == expected


def test_normalize_country_unknown_iso_like():
    assert normalize_country("zz") == "ZZ"


def test_normalize_countries_dedupes():
    assert normalize_countries(["Argentina", "AR"]) == ["AR"]
    assert normalize_countries("Brasil") == ["BR"]
    assert normalize_countries(None) == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("80s", 1980),
        ("1980s", 1980),
        ("1985", 1980),
        ("años 80", 1980),
        ("late 90s", 1990),
        ("90", 1990),
        ("00s", 2000),
        ("2003", 2000),
        ("20s", 2020),
        (1985, 1980),
        (2024, 2020),
        ("", None),
        (None, None),
        ("sin numeros", None),
        (True, None),
        (500, None),
        (9999, None),
        (29, 2020),
    ],
)
def test_decade_of(raw, expected):
    assert decade_of(raw) == expected


def test_decades_of_variants():
    assert decades_of(["80s", "1985", "90s"]) == [1980, 1990]
    assert decades_of("70s") == [1970]
    assert decades_of(None) == []
    assert decades_of([]) == []
    assert decades_of(1980) == [1980]


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.5, 0.5),
        (1, 1.0),
        (0, 0.0),
        (5, 0.5),
        (10, 1.0),
        (50, 0.5),
        (100, 1.0),
        (150, 1.0),
        (-3, 0.0),
        ("alta", 0.75),
        ("very high", 0.95),
        ("baja", 0.25),
        ("0.8", 0.8),
        ("no-numero", None),
        (None, None),
        (True, None),
        ([], None),
    ],
)
def test_normalize_energy(raw, expected):
    assert normalize_energy(raw) == expected


def test_facet_values_from_facets_full():
    facets = {
        "genres": ["Folk Metal", "Folk Metal"],
        "moods": ["Epico"],
        "themes": ["Cerveza"],
        "references": ["Tolkien"],
        "instrumentation": ["Fiddle"],
        "lyrical_themes": ["Dwarves"],
        "for_fans_of": ["Alestorm"],
        "lastfm_tags": ["drinking song"],
        "language": "Español",
        "country": "Argentina",
        "era": "90s",
        "energy": "alta",
        "is_ballad": False,
        "is_instrumental": True,
        "is_concept_album": True,
    }
    values = facet_values_from_facets(facets)
    assert values["genres"] == [("folk metal", None), ("folk metal", None)]
    assert values["moods"] == [("epico", None)]
    assert values["languages"] == [("es", None)]
    assert values["countries"] == [("AR", None)]
    assert values["decades"] == [("1990", 1990.0)]
    assert values["energy"] == [("0.750", 0.75)]
    assert values["is_ballad"] == [("0", 0.0)]
    assert values["is_instrumental"] == [("1", 1.0)]
    assert values["is_concept_album"] == [("1", 1.0)]


def test_facet_values_from_facets_lists_variants():
    values = facet_values_from_facets(
        {"languages": ["Español", "english"], "countries": ["AR", "br"], "decades": ["80s", "90s"]}
    )
    assert values["languages"] == [("es", None), ("en", None)]
    assert values["countries"] == [("AR", None), ("BR", None)]
    assert values["decades"] == [("1980", 1980.0), ("1990", 1990.0)]


def test_facet_values_from_facets_empty_and_garbage():
    assert facet_values_from_facets({}) == {}
    assert facet_values_from_facets({"genres": "no-lista"}) == {}
    assert facet_values_from_facets({"language": "", "energy": None}) == {}
    values = facet_values_from_facets({"genres": ["", "  ", "Rock"]})
    assert values["genres"] == [("rock", None)]


def test_display_helpers():
    assert display_language("es") == "Español"
    assert display_language("xx") == "xx"
    assert display_country("AR") == "Argentina"
    assert display_country("ZZ") == "ZZ"
