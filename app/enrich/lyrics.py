"""Detección de idioma por letra (offline, opcional).

Usa py3langid (139 idiomas, BSD-3) si está instalado. Es un dato duro:
pisa lo que haya dicho el LLM sobre el idioma de la canción. Sin la
dependencia, todo degrada a no-op.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger("bardo.enrich.lyrics")

MIN_TEXT_LENGTH = 40
MIN_CONFIDENCE = 0.5

_identifier = None
_load_failed = False

# idiomas que el vocabulario de Bardo conoce; si el detector devuelve otro,
# se guarda igual (son códigos ISO válidos) salvo los no-idioma.
SKIP_CODES = {"und", "zxx", ""}


def available() -> bool:
    global _identifier, _load_failed
    if _identifier is not None:
        return True
    if _load_failed:
        return False
    try:
        import py3langid

        _identifier = py3langid
        return True
    except ImportError:
        _load_failed = True
        log.info("py3langid no instalado; detección de idioma deshabilitada")
        return False


def detect_language(text: str) -> tuple[str, float] | None:
    """Devuelve (código ISO 639-1, confianza) o None si no hay señal."""
    if not text:
        return None
    clean = _clean_lyrics(text)
    if len(clean) < MIN_TEXT_LENGTH:
        return None
    if not available():
        return None
    try:
        code, score = _identifier.classify(clean)
    except Exception as exc:  # pragma: no cover - defensivo
        log.warning("language detection failed: %s", exc)
        return None
    if not code or code in SKIP_CODES:
        return None
    confidence = _normalize_confidence(score)
    if confidence is not None and confidence < MIN_CONFIDENCE:
        return None
    return str(code), confidence if confidence is not None else 0.5


def _normalize_confidence(score: Any) -> float | None:
    """langid devuelve log-prob (negativo) o probabilidad según configuración."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    if value > 1:
        return 1.0
    return value


_CREDIT_RE = re.compile(
    r"^\s*(\[|\(|\{)?\s*(verse|chorus|bridge|intro|outro|pre-?chorus|"
    r"hook|refrain|solo|instrumental|spoken)\b.*$",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean_lyrics(text: str) -> str:
    """Saca marcadores de sección, timestamps y HTML de la letra."""
    out_lines: list[str] = []
    for line in str(text).splitlines():
        line = _TAG_RE.sub(" ", line)
        line = re.sub(r"\[\d{1,2}:\d{2}(?:\.\d+)?\]", " ", line)
        if _CREDIT_RE.match(line):
            continue
        if line.strip():
            out_lines.append(line.strip())
    return "\n".join(out_lines)
