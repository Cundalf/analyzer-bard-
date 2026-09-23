from __future__ import annotations

import pytest

from app.enrich.generic import filter_valid, is_generic, is_valid


# ------------------------------------------------------------ genéricos

@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "   ",
        "Unknown",
        "unknown",
        "UNKNOWN ARTIST",
        "[Unknown Artist]",
        "[unknown album]",
        "(Unknown)",
        "Unknown Album",
        "Artista Desconocido",
        "undefined",
        "NULL",
        "nil",
        "None",
        "N/A",
        "n/a",
        "NA",
        "Various Artists",
        "various",
        "Varios Artistas",
        "VA",
        "VV AA",
        "Untitled",
        "no title",
        "Sin Título",
        "-",
        "--",
        ".",
        "?",
        "??",
        "0",
        "00",
        "Track",
        "track 01",
        "Track 1",
        "Pista 04",
        "01",
        "04 -",
        "compilation",
        "Miscellaneous",
        "Other",
        "Otros",
        "genre",
        "Unknown Genre",
    ],
)
def test_is_generic_true(value):
    assert is_generic(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "Wind Rose",
        "Unknown Mortal Orchestra",  # banda real: no matchea exacto
        "The Unknowns",
        "1979",  # título real de Smashing Pumpkins
        "Track of Time",
        "Compilation of Souls",
        "Otro Lugar",
        "N/A (Live)",  # tiene más contexto
        "Various Moods",
        "Rock",
        "Folk Metal",
        "Soda Stereo",
        "Ñandú",
        "Xibalba",
        "Zero 7",
        "Track 1 - Intro",
    ],
)
def test_is_generic_false(value):
    assert is_generic(value) is False


def test_is_generic_collections_and_numbers():
    assert is_generic([]) is True
    assert is_generic(["rock"]) is True
    assert is_generic({}) is True
    assert is_generic({"a": 1}) is True
    assert is_generic(0) is True
    assert is_generic(0.0) is True
    assert is_generic(float("nan")) is True
    assert is_generic(1) is False
    assert is_generic(2019) is False
    assert is_generic(True) is False


def test_is_valid_inverse():
    assert is_valid("Rock") is True
    assert is_valid("unknown") is False
    assert is_valid(None) is False


# ------------------------------------------------------------ filter_valid

def test_filter_valid_removes_generics_and_dedupes():
    assert filter_valid(["Rock", "unknown", "rock", "N/A", "Pop"]) == ["Rock", "Pop"]


def test_filter_valid_strips_and_skips_empty():
    assert filter_valid(["  folk metal  ", "", "   ", None]) == ["folk metal"]


def test_filter_valid_keeps_order():
    assert filter_valid(["metal", "folk", "power"]) == ["metal", "folk", "power"]


def test_filter_valid_empty_inputs():
    assert filter_valid([]) == []
    assert filter_valid(None) == []
    assert filter_valid(["unknown", "N/A", "-"]) == []


def test_filter_valid_non_string_values():
    # un número puro de hasta 3 cifras es placeholder ("Track 01")
    assert filter_valid([123, "Rock", 0]) == ["Rock"]
    # un año de 4 cifras sí es información válida
    assert filter_valid([2019]) == ["2019"]


def test_filter_valid_normalizes_accents_for_dedupe():
    assert filter_valid(["Ñandú", "nandu"]) == ["Ñandú"]


def test_filter_valid_track_number_placeholders():
    assert filter_valid(["Track 01", "Real Song", "04", "Otro Tema"]) == [
        "Real Song",
        "Otro Tema",
    ]
