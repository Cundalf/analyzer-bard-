"""Vocabulario controlado para facetas.

Normaliza idiomas a ISO 639-1, países a ISO 3166-1 alpha-2, eras a décadas
numéricas y energía a un float 0..1. Así "español", "castellano" y "spanish"
matchean el mismo filtro.
"""

from __future__ import annotations

import re
from typing import Any

from app.enrich.canonicalize import normalize

# ---------------------------------------------------------------- idiomas

LANGUAGE_ALIASES: dict[str, str] = {
    "espanol": "es",
    "castellano": "es",
    "spanish": "es",
    "es": "es",
    "argentino": "es",
    "mexicano": "es",
    "latino": "es",
    "ingles": "en",
    "english": "en",
    "en": "en",
    "portugues": "pt",
    "portuguese": "pt",
    "pt": "pt",
    "brasileno": "pt",
    "frances": "fr",
    "french": "fr",
    "fr": "fr",
    "aleman": "de",
    "german": "de",
    "de": "de",
    "italiano": "it",
    "italian": "it",
    "it": "it",
    "japones": "ja",
    "japanese": "ja",
    "ja": "ja",
    "coreano": "ko",
    "korean": "ko",
    "ko": "ko",
    "chino": "zh",
    "chinese": "zh",
    "mandarin": "zh",
    "zh": "zh",
    "ruso": "ru",
    "russian": "ru",
    "ru": "ru",
    "ucraniano": "uk",
    "ukrainian": "uk",
    "uk": "uk",
    "polaco": "pl",
    "polish": "pl",
    "pl": "pl",
    "sueco": "sv",
    "swedish": "sv",
    "sv": "sv",
    "noruego": "no",
    "norwegian": "no",
    "no": "no",
    "danes": "da",
    "danish": "da",
    "da": "da",
    "fines": "fi",
    "finnish": "fi",
    "fi": "fi",
    "holandes": "nl",
    "dutch": "nl",
    "nl": "nl",
    "griego": "el",
    "greek": "el",
    "el": "el",
    "turco": "tr",
    "turkish": "tr",
    "tr": "tr",
    "arabe": "ar",
    "arabic": "ar",
    "ar": "ar",
    "hebreo": "he",
    "hebrew": "he",
    "he": "he",
    "hindi": "hi",
    "hi": "hi",
    "tailandes": "th",
    "thai": "th",
    "th": "th",
    "vietnamita": "vi",
    "vietnamese": "vi",
    "vi": "vi",
    "indonesio": "id",
    "indonesian": "id",
    "id": "id",
    "checo": "cs",
    "czech": "cs",
    "cs": "cs",
    "hungaro": "hu",
    "hungarian": "hu",
    "hu": "hu",
    "rumano": "ro",
    "romanian": "ro",
    "ro": "ro",
    "catalan": "ca",
    "ca": "ca",
    "gallego": "gl",
    "galician": "gl",
    "gl": "gl",
    "euskera": "eu",
    "vasco": "eu",
    "basque": "eu",
    "eu": "eu",
    "latín": "la",
    "latin": "la",
    "la": "la",
    "islandes": "is",
    "icelandic": "is",
    "is": "is",
    "irlandes": "ga",
    "irish": "ga",
    "ga": "ga",
    "instrumental": "zxx",
    "zxx": "zxx",
}

LANGUAGE_NAMES: dict[str, str] = {
    "es": "Español",
    "en": "Inglés",
    "pt": "Portugués",
    "fr": "Francés",
    "de": "Alemán",
    "it": "Italiano",
    "ja": "Japonés",
    "ko": "Coreano",
    "zh": "Chino",
    "ru": "Ruso",
    "uk": "Ucraniano",
    "la": "Latín",
    "zxx": "Instrumental",
}

# ---------------------------------------------------------------- países

COUNTRY_ALIASES: dict[str, str] = {
    "argentina": "AR",
    "ar": "AR",
    "españa": "ES",
    "espana": "ES",
    "spain": "ES",
    "es": "ES",
    "mexico": "MX",
    "méxico": "MX",
    "mx": "MX",
    "chile": "CL",
    "cl": "CL",
    "colombia": "CO",
    "co": "CO",
    "peru": "PE",
    "perú": "PE",
    "pe": "PE",
    "uruguay": "UY",
    "uy": "UY",
    "brasil": "BR",
    "brazil": "BR",
    "br": "BR",
    "estados unidos": "US",
    "eeuu": "US",
    "usa": "US",
    "united states": "US",
    "us": "US",
    "reino unido": "GB",
    "inglaterra": "GB",
    "uk": "GB",
    "united kingdom": "GB",
    "gb": "GB",
    "irlanda": "IE",
    "ireland": "IE",
    "ie": "IE",
    "francia": "FR",
    "france": "FR",
    "fr": "FR",
    "alemania": "DE",
    "germany": "DE",
    "de": "DE",
    "italia": "IT",
    "italy": "IT",
    "it": "IT",
    "suecia": "SE",
    "sweden": "SE",
    "se": "SE",
    "noruega": "NO",
    "norway": "NO",
    "no": "NO",
    "finlandia": "FI",
    "finland": "FI",
    "fi": "FI",
    "dinamarca": "DK",
    "denmark": "DK",
    "dk": "DK",
    "islandia": "IS",
    "iceland": "IS",
    "is": "IS",
    "paises bajos": "NL",
    "holanda": "NL",
    "netherlands": "NL",
    "nl": "NL",
    "belgica": "BE",
    "belgium": "BE",
    "be": "BE",
    "portugal": "PT",
    "pt": "PT",
    "grecia": "GR",
    "greece": "GR",
    "gr": "GR",
    "polonia": "PL",
    "poland": "PL",
    "pl": "PL",
    "rusia": "RU",
    "russia": "RU",
    "ru": "RU",
    "ucrania": "UA",
    "ukraine": "UA",
    "ua": "UA",
    "canada": "CA",
    "canadá": "CA",
    "ca": "CA",
    "australia": "AU",
    "au": "AU",
    "nueva zelanda": "NZ",
    "new zealand": "NZ",
    "nz": "NZ",
    "japon": "JP",
    "japón": "JP",
    "japan": "JP",
    "jp": "JP",
    "corea": "KR",
    "korea": "KR",
    "kr": "KR",
    "china": "CN",
    "cn": "CN",
    "taiwan": "TW",
    "tw": "TW",
    "india": "IN",
    "in": "IN",
    "israel": "IL",
    "il": "IL",
    "turquia": "TR",
    "turkey": "TR",
    "tr": "TR",
    "egipto": "EG",
    "egypt": "EG",
    "eg": "EG",
    "sudafrica": "ZA",
    "south africa": "ZA",
    "za": "ZA",
    "jamaica": "JM",
    "jm": "JM",
    "cuba": "CU",
    "cu": "CU",
    "puerto rico": "PR",
    "pr": "PR",
    "venezuela": "VE",
    "ve": "VE",
    "ecuador": "EC",
    "ec": "EC",
    "bolivia": "BO",
    "bo": "BO",
    "paraguay": "PY",
    "py": "PY",
    "costa rica": "CR",
    "cr": "CR",
    "panama": "PA",
    "panamá": "PA",
    "pa": "PA",
    "republica dominicana": "DO",
    "dominican republic": "DO",
    "do": "DO",
    "guatemala": "GT",
    "gt": "GT",
    "honduras": "HN",
    "hn": "HN",
    "el salvador": "SV",
    "sv": "SV",
    "nicaragua": "NI",
    "ni": "NI",
}

COUNTRY_NAMES: dict[str, str] = {
    "AR": "Argentina",
    "ES": "España",
    "MX": "México",
    "CL": "Chile",
    "CO": "Colombia",
    "PE": "Perú",
    "UY": "Uruguay",
    "BR": "Brasil",
    "US": "Estados Unidos",
    "GB": "Reino Unido",
    "IE": "Irlanda",
    "FR": "Francia",
    "DE": "Alemania",
    "IT": "Italia",
    "SE": "Suecia",
    "NO": "Noruega",
    "FI": "Finlandia",
    "JP": "Japón",
    "KR": "Corea",
    "CN": "China",
    "RU": "Rusia",
}

# ---------------------------------------------------------------- energía

ENERGY_LABELS: dict[str, float] = {
    "muy baja": 0.1,
    "very low": 0.1,
    "baja": 0.25,
    "low": 0.25,
    "tranquila": 0.2,
    "calma": 0.2,
    "calm": 0.2,
    "media": 0.5,
    "medium": 0.5,
    "mid": 0.5,
    "moderada": 0.5,
    "alta": 0.75,
    "high": 0.75,
    "energica": 0.9,
    "energetica": 0.9,
    "energetic": 0.9,
    "muy alta": 0.95,
    "very high": 0.95,
    "frenetica": 1.0,
    "frantic": 1.0,
}

_DECADE_RE = re.compile(r"(\d{2,4})")


def normalize_language(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    key = normalize(text)
    if key in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[key]
    if re.fullmatch(r"[a-z]{2,3}", key):
        return key
    return None


def normalize_languages(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        values = [values]
    out: list[str] = []
    for value in values:
        code = normalize_language(value)
        if code and code not in out:
            out.append(code)
    return out


def normalize_country(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    key = normalize(text)
    if key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key]
    if re.fullmatch(r"[a-z]{2}", key):
        return key.upper()
    return None


def normalize_countries(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        values = [values]
    out: list[str] = []
    for value in values:
        code = normalize_country(value)
        if code and code not in out:
            out.append(code)
    return out


def decade_of(value: Any) -> int | None:
    """Convierte '80s', 'años 80', '1985' o 'late 90s' a 1980/1990."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        year = int(value)
        if 1000 <= year <= 2999:
            return (year // 10) * 10
        if 0 <= year < 100:
            base = year - (year % 10)
            return 2000 + base if year < 30 else 1900 + base
        return None
    match = _DECADE_RE.search(str(value))
    if not match:
        return None
    number = int(match.group(1))
    if number >= 1000:
        if 1000 <= number <= 2999:
            return (number // 10) * 10
        return None
    base = number - (number % 10)
    if number < 30:
        return 2000 + base
    return 1900 + base


def decades_of(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, (str, bytes, int, float)) and not isinstance(values, bool):
        values = [values]
    out: list[int] = []
    for value in values:
        decade = decade_of(value)
        if decade and decade not in out:
            out.append(decade)
    return out


def normalize_energy(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        key = normalize(value)
        if key in ENERGY_LABELS:
            return ENERGY_LABELS[key]
        try:
            value = float(key)
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number < 0:
        return 0.0
    if number <= 1:
        return round(number, 3)
    if number <= 10:
        return round(min(number / 10.0, 1.0), 3)
    if number <= 100:
        return round(min(number / 100.0, 1.0), 3)
    return 1.0


def display_language(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def display_country(code: str) -> str:
    return COUNTRY_NAMES.get(code, code)


def facet_values_from_facets(facets: dict[str, Any]) -> dict[str, list[tuple[str, float | None]]]:
    """Expande facetas crudas a valores normalizados por faceta.

    Devuelve {facet: [(value, num)]}. Incluye idiomas/países ISO, décadas,
    energía y booleanos como '0'/'1'.
    """
    out: dict[str, list[tuple[str, float | None]]] = {}
    list_facets = (
        "genres",
        "subgenres",
        "moods",
        "themes",
        "references",
        "instrumentation",
        "lyrical_themes",
        "for_fans_of",
        "lastfm_tags",
    )
    for facet in list_facets:
        values = facets.get(facet)
        if not isinstance(values, list):
            continue
        for raw in values:
            value = normalize(str(raw))
            if value:
                out.setdefault(facet, []).append((value, None))

    languages = normalize_languages(facets.get("languages") or facets.get("language"))
    for code in languages:
        out.setdefault("languages", []).append((code, None))

    countries = normalize_countries(facets.get("countries") or facets.get("country"))
    for code in countries:
        out.setdefault("countries", []).append((code, None))

    for decade in decades_of(facets.get("decades") or facets.get("era")):
        out.setdefault("decades", []).append((str(decade), float(decade)))

    energy = normalize_energy(facets.get("energy"))
    if energy is not None:
        out.setdefault("energy", []).append((f"{energy:.3f}", energy))

    for flag in ("is_ballad", "is_instrumental", "is_concept_album"):
        if facets.get(flag) is True:
            out.setdefault(flag, []).append(("1", 1.0))
        elif facets.get(flag) is False:
            out.setdefault(flag, []).append(("0", 0.0))
    return out
