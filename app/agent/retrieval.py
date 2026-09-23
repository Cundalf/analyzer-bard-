from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.enrich.artist import get_ficha

log = logging.getLogger("bardo.agent.retrieval")


@dataclass
class FacetFilters:
    """Filtros duros del recall. Los facetas se resuelven en facet_index y la
    coincidencia es a nivel track, álbum o artista (herencia en consulta)."""

    year_min: int | None = None
    year_max: int | None = None
    exclude_genres: list[str] = field(default_factory=list)
    include_genres: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    decades: list[int] = field(default_factory=list)
    energy_min: float | None = None
    energy_max: float | None = None
    instrumental: bool | None = None
    ballad: bool | None = None
    concept_album: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FacetFilters:
        from app.enrich.vocab import (
            decades_of,
            normalize_countries,
            normalize_energy,
            normalize_languages,
        )

        if not data:
            return cls()
        return cls(
            year_min=_as_int(data.get("year_min")),
            year_max=_as_int(data.get("year_max")),
            exclude_genres=_as_list(data.get("exclude_genres")),
            include_genres=_as_list(data.get("include_genres")),
            languages=normalize_languages(data.get("languages")),
            countries=normalize_countries(data.get("countries")),
            decades=decades_of(data.get("decades")),
            energy_min=normalize_energy(data.get("energy_min")),
            energy_max=normalize_energy(data.get("energy_max")),
            instrumental=_as_bool(data.get("instrumental")),
            ballad=_as_bool(data.get("ballad")),
            concept_album=_as_bool(data.get("concept_album")),
        )

    def sql(self) -> tuple[str, list[Any]]:
        """Devuelve ('' o ' AND (...)') + parámetros para BASE_TRACK_QUERY."""
        conditions: list[str] = []
        params: list[Any] = []

        if self.year_min:
            conditions.append("(al.year IS NULL OR al.year >= ?)")
            params.append(int(self.year_min))
        if self.year_max:
            conditions.append("(al.year IS NULL OR al.year <= ?)")
            params.append(int(self.year_max))

        if self.include_genres:
            genre_clauses = []
            for genre in self.include_genres:
                genre_clauses.append("al.genre LIKE ? OR " + _facet_sql("genres", [genre]))
                params.append(f"%{genre}%")
                params.extend(_facet_params("genres", [genre]))
            conditions.append("(" + " OR ".join(genre_clauses) + ")")

        if self.exclude_genres:
            genre_clauses = []
            for genre in self.exclude_genres:
                genre_clauses.append(
                    "COALESCE(al.genre, '') LIKE ? OR " + _facet_sql("genres", [genre])
                )
                params.append(f"%{genre}%")
                params.extend(_facet_params("genres", [genre]))
            conditions.append("NOT (" + " OR ".join(genre_clauses) + ")")

        for facet, values in (
            ("languages", self.languages),
            ("countries", list(self.countries)),
        ):
            if values:
                conditions.append(_facet_sql(facet, values))
                params.extend(_facet_params(facet, values))

        if self.decades:
            conditions.append(_facet_sql("decades", [str(d) for d in self.decades]))
            params.extend(_facet_params("decades", [str(d) for d in self.decades]))

        if self.energy_min is not None or self.energy_max is not None:
            conditions.append(_facet_numeric_sql("energy", self.energy_min, self.energy_max))
            params.extend(_facet_numeric_params("energy", self.energy_min, self.energy_max))

        for facet, flag in (
            ("is_instrumental", self.instrumental),
            ("is_ballad", self.ballad),
            ("is_concept_album", self.concept_album),
        ):
            if flag is True:
                conditions.append(_facet_sql(facet, ["1"]))
                params.extend(_facet_params(facet, ["1"]))
            elif flag is False:
                # negativo: "no tiene la marca", aunque no exista el registro
                conditions.append(f"NOT {_facet_sql(facet, ['1'])}")
                params.extend(_facet_params(facet, ["1"]))

        if not conditions:
            return "", []
        return " AND " + " AND ".join(conditions), params


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value if isinstance(value, bool) else None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return []


def _facet_refs(facet: str, values: list[str]) -> tuple[str, list[Any]]:
    marks = ",".join("?" * len(values))
    params: list[Any] = []
    for _ in range(3):
        params.append(facet)
        params.extend(values)
    sql = (
        "EXISTS (SELECT 1 FROM facet_index fx WHERE fx.entity_type = 'track' "
        f"AND fx.entity_id = t.navidrome_id AND fx.facet = ? "
        f"AND fx.value IN ({marks})) "
        "OR EXISTS (SELECT 1 FROM facet_index fx WHERE fx.entity_type = 'album' "
        f"AND fx.entity_id = al.navidrome_id AND fx.facet = ? "
        f"AND fx.value IN ({marks})) "
        "OR EXISTS (SELECT 1 FROM facet_index fx WHERE fx.entity_type = 'artist' "
        f"AND fx.entity_id = ar.navidrome_id AND fx.facet = ? "
        f"AND fx.value IN ({marks}))"
    )
    return sql, params


def _facet_sql(facet: str, values: list[str]) -> str:
    sql, _ = _facet_refs(facet, values)
    return f"({sql})"


def _facet_params(facet: str, values: list[str]) -> list[Any]:
    _, params = _facet_refs(facet, values)
    return params


def _facet_numeric_refs(facet: str, low: float | None, high: float | None) -> tuple[str, list[Any]]:
    clauses = ["fx.facet = ?", "fx.num IS NOT NULL"]
    extras: list[Any] = []
    if low is not None:
        clauses.append("fx.num >= ?")
        extras.append(float(low))
    if high is not None:
        clauses.append("fx.num <= ?")
        extras.append(float(high))
    inner = " AND ".join(clauses)
    params: list[Any] = []
    sql_parts = []
    for entity, ref in (("track", "t"), ("album", "al"), ("artist", "ar")):
        sql_parts.append(
            f"EXISTS (SELECT 1 FROM facet_index fx WHERE fx.entity_type = '{entity}' "
            f"AND fx.entity_id = {ref}.navidrome_id AND {inner})"
        )
        params.append(facet)
        params.extend(extras)
    return " OR ".join(sql_parts), params


def _facet_numeric_sql(facet: str, low: float | None, high: float | None) -> str:
    sql, _ = _facet_numeric_refs(facet, low, high)
    return f"({sql})"


def _facet_numeric_params(facet: str, low: float | None, high: float | None) -> list[Any]:
    _, params = _facet_numeric_refs(facet, low, high)
    return params


def is_unknown(value: str | None) -> bool:
    from app.enrich.generic import is_generic

    return is_generic(value)


def _row_to_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "track_id": row.get("navidrome_id") or "",
        "title": row.get("title") or "",
        "artist": row.get("artist_name") or "",
        "artist_id": row.get("artist_nid") or "",
        "album": row.get("album_name") or "",
        "album_id": row.get("album_nid") or "",
        "year": row.get("year"),
        "genre": row.get("genre"),
        "duration": row.get("duration"),
        "moods": row.get("moods") or [],
        "themes": row.get("themes") or [],
        "description": row.get("description") or "",
        "score": row.get("score") or 0.0,
        "source": row.get("source") or "",
    }


BASE_TRACK_QUERY = """
SELECT t.navidrome_id, t.title, t.duration,
       ar.name AS artist_name, ar.navidrome_id AS artist_nid,
       al.name AS album_name, al.navidrome_id AS album_nid,
       al.year, al.genre,
       f.facets AS facets, f.description AS description, f.source AS source
FROM tracks t
LEFT JOIN artists ar ON ar.id = t.artist_id
LEFT JOIN albums al ON al.id = t.album_id
LEFT JOIN fichas f
  ON f.entity_type = 'track'
 AND f.entity_id = t.navidrome_id
LEFT JOIN fichas fa
  ON fa.entity_type = 'album'
 AND fa.entity_id = al.navidrome_id
LEFT JOIN fichas fr
  ON fr.entity_type = 'artist'
 AND fr.entity_id = ar.navidrome_id
"""


def _hydrate(conn: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import json

    enriched: list[dict[str, Any]] = []
    album_fichas: dict[str, dict[str, Any]] = {}
    artist_fichas: dict[str, dict[str, Any]] = {}
    for row in rows:
        facets: dict[str, Any] = {}
        if row.get("facets"):
            try:
                facets = json.loads(row["facets"])
            except json.JSONDecodeError:
                facets = {}
        if not facets and row.get("album_nid"):
            if row["album_nid"] not in album_fichas:
                album_fichas[row["album_nid"]] = get_ficha(conn, "album", row["album_nid"]) or {}
            album_ficha = album_fichas[row["album_nid"]]
            af = album_ficha.get("facets", {})
            facets = {
                "moods": af.get("moods", []),
                "themes": af.get("themes", []),
            }
            if not row.get("description"):
                row["description"] = album_ficha.get("description", "")
        needs_moods = not facets.get("moods")
        needs_themes = not facets.get("themes")
        if row.get("artist_nid") and (needs_moods or needs_themes):
            if row["artist_nid"] not in artist_fichas:
                artist_fichas[row["artist_nid"]] = (
                    get_ficha(conn, "artist", row["artist_nid"]) or {}
                )
            arf = artist_fichas[row["artist_nid"]].get("facets", {})
            if needs_moods:
                facets["moods"] = arf.get("moods", [])
            if needs_themes:
                facets["themes"] = arf.get("lyrical_themes", []) or arf.get("references", [])
        row["moods"] = facets.get("moods", [])
        row["themes"] = facets.get("themes", [])
        enriched.append(_row_to_candidate(row))
    return enriched


def search_by_terms(
    conn: Any,
    terms: list[str],
    *,
    limit: int = 60,
    filters: FacetFilters | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    exclude_genres: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Recall determinístico: LIKE sobre tracks/álbumes/artistas + género."""
    if not terms:
        return []
    clauses: list[str] = []
    params: list[Any] = []
    for term in terms:
        pattern = f"%{term.strip()}%"
        clauses.append(
            "(t.title LIKE ? OR al.name LIKE ? OR ar.name LIKE ? "
            "OR al.genre LIKE ? OR f.facets LIKE ? OR fa.facets LIKE ? "
            "OR fr.facets LIKE ?)"
        )
        params.extend([pattern] * 7)
    where = "(" + " OR ".join(clauses) + ")"
    effective = _merge_filters(filters, year_min, year_max, exclude_genres)
    extra, extra_params = effective.sql()
    params.extend(extra_params)
    params.append(limit)
    rows = conn.execute(f"{BASE_TRACK_QUERY} WHERE {where}{extra} LIMIT ?", params).fetchall()
    return _hydrate(conn, [dict(r) for r in rows])


def search_fts(
    conn: Any,
    query: str,
    *,
    limit: int = 60,
    filters: FacetFilters | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    exclude_genres: list[str] | None = None,
) -> list[dict[str, Any]]:
    from app.db import fts_search

    hits = fts_search(conn, query, limit=limit * 3)
    track_ids = [h["entity_id"] for h in hits if h["entity_type"] == "track"]
    album_ids = [h["entity_id"] for h in hits if h["entity_type"] == "album"]
    artist_ids = [h["entity_id"] for h in hits if h["entity_type"] == "artist"]

    conditions: list[str] = []
    params: list[Any] = []
    if track_ids:
        marks = ",".join("?" * len(track_ids))
        conditions.append(f"t.navidrome_id IN ({marks})")
        params.extend(track_ids)
    if album_ids:
        marks = ",".join("?" * len(album_ids))
        conditions.append(f"al.navidrome_id IN ({marks})")
        params.extend(album_ids)
    if artist_ids:
        marks = ",".join("?" * len(artist_ids))
        conditions.append(f"ar.navidrome_id IN ({marks})")
        params.extend(artist_ids)
    if not conditions:
        return []
    effective = _merge_filters(filters, year_min, year_max, exclude_genres)
    extra, extra_params = effective.sql()
    params.extend(extra_params)
    params.append(limit)
    rows = conn.execute(
        f"{BASE_TRACK_QUERY} WHERE ({' OR '.join(conditions)}){extra} LIMIT ?",
        params,
    ).fetchall()
    out = _hydrate(conn, [dict(r) for r in rows])
    for candidate in out:
        candidate["source"] = "fts"
    return out


def search_vectors(
    conn: Any,
    embedding: list[float],
    *,
    limit: int = 60,
    filters: FacetFilters | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    exclude_genres: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Búsqueda vectorial + drill-down a tracks."""
    from app.db import vec_search

    hits = vec_search(conn, embedding, limit=limit)
    track_ids = [h["entity_id"] for h in hits if h["entity_type"] == "track"]
    album_ids = [h["entity_id"] for h in hits if h["entity_type"] == "album"]
    artist_ids = [h["entity_id"] for h in hits if h["entity_type"] == "artist"]
    distances = {h["entity_id"]: h["distance"] for h in hits}

    conditions: list[str] = []
    params: list[Any] = []
    if track_ids:
        marks = ",".join("?" * len(track_ids))
        conditions.append(f"t.navidrome_id IN ({marks})")
        params.extend(track_ids)
    if album_ids:
        marks = ",".join("?" * len(album_ids))
        conditions.append(f"al.navidrome_id IN ({marks})")
        params.extend(album_ids)
    if artist_ids:
        marks = ",".join("?" * len(artist_ids))
        conditions.append(f"ar.navidrome_id IN ({marks})")
        params.extend(artist_ids)
    if not conditions:
        return []
    effective = _merge_filters(filters, year_min, year_max, exclude_genres)
    extra, extra_params = effective.sql()
    params.extend(extra_params)
    rows = conn.execute(
        f"{BASE_TRACK_QUERY} WHERE ({' OR '.join(conditions)}){extra} LIMIT 200",
        params,
    ).fetchall()
    out = _hydrate(conn, [dict(r) for r in rows])
    for candidate in out:
        direct = distances.get(candidate["track_id"])
        album_d = distances.get(candidate["album_id"])
        artist_d = distances.get(candidate["artist_id"])
        distance = min([d for d in (direct, album_d, artist_d) if d is not None] or [9.0])
        candidate["score"] = round(1.0 / (1.0 + distance), 4)
        candidate["source"] = "vector"
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def _merge_filters(
    filters: FacetFilters | None,
    year_min: int | None,
    year_max: int | None,
    exclude_genres: list[str] | None,
) -> FacetFilters:
    """Combina filtros estructurados con los parámetros legacy."""
    if filters is None:
        return FacetFilters(
            year_min=year_min,
            year_max=year_max,
            exclude_genres=list(exclude_genres or []),
        )
    if year_min is not None:
        filters.year_min = year_min
    if year_max is not None:
        filters.year_max = year_max
    if exclude_genres:
        merged = list(filters.exclude_genres)
        for genre in exclude_genres:
            if genre not in merged:
                merged.append(genre)
        filters.exclude_genres = merged
    return filters


def rrf_merge(result_sets: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion de varios rankings de candidatos."""
    scores: dict[str, float] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for results in result_sets:
        for rank, candidate in enumerate(results, 1):
            key = candidate["track_id"]
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(key, candidate)
    merged = []
    for key, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        candidate = dict(by_id[key])
        candidate["score"] = round(score, 6)
        candidate["source"] = "hybrid"
        merged.append(candidate)
    return merged
