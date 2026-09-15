from __future__ import annotations

import json
import logging
import unicodedata
from typing import Any, Iterable

log = logging.getLogger("bardo.enrich.canonicalize")


def normalize(term: str) -> str:
    text = unicodedata.normalize("NFKD", str(term).strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


class Canonicalizer:
    def __init__(self, synonyms: dict[str, str] | None = None):
        self.synonyms = {
            normalize(k): normalize(v) for k, v in (synonyms or {}).items()
        }
        self._display: dict[str, str] = {}
        for value in (synonyms or {}).values():
            self._display.setdefault(normalize(value), str(value).strip())

    @classmethod
    def from_db(cls, conn: Any) -> "Canonicalizer":
        rows = conn.execute("SELECT term, canonical FROM synonyms").fetchall()
        return cls({r["term"]: r["canonical"] for r in rows})

    def add(self, term: str, canonical: str) -> None:
        self.synonyms[normalize(term)] = normalize(canonical)
        self._display.setdefault(normalize(canonical), str(canonical).strip())

    def canonical(self, term: str) -> str:
        key = normalize(term)
        seen: set[str] = set()
        while key in self.synonyms and key not in seen:
            seen.add(key)
            key = self.synonyms[key]
        return self._display.get(key, key)

    def canonical_list(
        self, terms: Iterable[str], *, keep_order: bool = True
    ) -> list[str]:
        out: list[str] = []
        for term in terms or []:
            canon = self.canonical(term)
            if canon and canon not in out:
                out.append(canon)
        return out

    def expand(self, terms: Iterable[str]) -> list[str]:
        """Devuelve los términos originales + sus canónicos + sinónimos."""
        result: list[str] = []
        normalized = [normalize(t) for t in terms if str(t).strip()]
        reverse: dict[str, list[str]] = {}
        for term, canon in self.synonyms.items():
            reverse.setdefault(canon, []).append(term)
        for term in normalized:
            candidates = [term, self.canonical(term)]
            candidates.extend(reverse.get(self.canonical(term), []))
            for cand in candidates:
                if cand and cand not in result:
                    result.append(cand)
        return result


def canonical_facets(facets: dict[str, Any], canon: Canonicalizer) -> dict[str, Any]:
    """Normaliza los campos de una ficha: listas, idioma, país, década, energía."""
    from app.enrich.vocab import (
        normalize_countries,
        normalize_energy,
        normalize_languages,
    )

    list_fields = (
        "genres",
        "subgenres",
        "lyrical_themes",
        "moods",
        "references",
        "themes",
        "instrumentation",
        "for_fans_of",
    )
    out = dict(facets)
    for field in list_fields:
        if field in out and isinstance(out[field], list):
            out[field] = canon.canonical_list(out[field])

    languages = normalize_languages(out.get("language"))
    if not languages:
        languages = normalize_languages(out.get("languages"))
    if languages:
        out["language"] = languages[0]
    elif "language" in out:
        out.pop("language")
    if "languages" in out and not isinstance(out.get("languages"), list):
        out.pop("languages")
    if languages and len(languages) > 1:
        out["languages"] = languages

    country = normalize_countries(out.get("country"))
    if not country:
        country = normalize_countries(out.get("countries"))
    if country:
        out["country"] = country[0]
    elif "country" in out:
        out.pop("country")
    if "countries" in out and not isinstance(out.get("countries"), list):
        out.pop("countries")
    if country and len(country) > 1:
        out["countries"] = country

    energy = normalize_energy(out.get("energy"))
    if energy is not None:
        out["energy"] = energy
    elif "energy" in out:
        out.pop("energy")
    return out


def ficha_text(facets: dict[str, Any], description: str = "") -> str:
    """Texto que se embeddea: description + tags canónicos."""
    parts: list[str] = []
    if description:
        parts.append(description)
    for key, value in (facets or {}).items():
        if key in {"description", "confidence"}:
            continue
        if isinstance(value, list) and value:
            parts.append(f"{key}: {', '.join(str(v) for v in value)}")
        elif isinstance(value, str) and value.strip():
            parts.append(f"{key}: {value.strip()}")
        elif isinstance(value, (int, float)) and key == "energy" and value:
            parts.append(f"energy: {value}")
    return "\n".join(parts)


def content_hash(payload: Any) -> str:
    import hashlib

    data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:32]
