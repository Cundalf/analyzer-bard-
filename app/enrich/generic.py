"""Detección de valores genéricos/basura en metadata.

Un valor genérico ("Unknown", "Various Artists", "N/A", "-") no es
información: nunca debe pisar un dato válido ni llegar a una ficha.

Cuidado con los falsos positivos: "Unknown Mortal Orchestra" es una banda
real. Por eso se comparan valores exactos (normalizados), no substrings,
salvo los corchetes explícitos tipo "[Unknown Artist]".
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from typing import Any

GENERIC_EXACT: frozenset[str] = frozenset(
    {
        "unknown",
        "unknown artist",
        "unknown album",
        "unknown title",
        "unknown genre",
        "unknown year",
        "unknown track",
        "artista desconocido",
        "album desconocido",
        "desconocido",
        "sin artista",
        "sin album",
        "sin titulo",
        "sin genero",
        "undefined",
        "null",
        "nil",
        "none",
        "n/a",
        "na",
        "various",
        "various artists",
        "varios artistas",
        "varios",
        "va",
        "vv aa",
        "vv. aa.",
        "compilation",
        "compilado",
        "untitled",
        "no title",
        "track",
        "track 00",
        "audio track",
        "pista",
        "genero",
        "genre",
        "misc",
        "miscellaneous",
        "otros",
        "other",
        "otras",
        "-",
        "--",
        "---",
        ".",
        "..",
        "?",
        "??",
        "???",
        "0",
        "00",
    }
)

GENERIC_PREFIXES: tuple[str, ...] = (
    "[unknown",
    "(unknown",
    "[unknown artist]",
    "[unknown album]",
    "[no artist]",
    "[no album]",
    "[sin artista]",
    "[sin album]",
)


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.split())


def is_generic(value: Any) -> bool:
    """True si el valor es basura/placeholder (no aporta información)."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 0 or (isinstance(value, float) and value != value)
    text = _normalize(value)
    if not text:
        return True
    if text in GENERIC_EXACT:
        return True
    if any(text.startswith(prefix) for prefix in GENERIC_PREFIXES):
        return True
    return bool(_is_track_number(text))


def _is_track_number(text: str) -> bool:
    """'track 01', 'pista 3', '01', '04 -' son placeholders, no títulos."""
    import re

    # Un número suelto de hasta 3 cifras es placeholder ("1979" tiene 4 y se
    # conserva: podría ser un título real).
    return bool(
        re.fullmatch(
            r"(?:track|pista|track no\.?|cancion|song)?\s*0*\d{1,3}\s*[-.]?",
            text,
        )
    )


def is_valid(value: Any) -> bool:
    """Inverso de is_generic, más explícito para leer en el flujo."""
    return not is_generic(value)


def filter_valid(values: Iterable[Any]) -> list[str]:
    """Limpia espacios y descarta genéricos, preservando el orden.

    Acepta un string suelto (lo trata como un único valor, no como
    iterable de caracteres).
    """
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        values = [values]
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value).split()).strip()
        if not text or is_generic(text):
            continue
        key = _normalize(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out
