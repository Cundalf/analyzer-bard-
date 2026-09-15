from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import sqlite_vec

from app.config import get_settings

_local = threading.local()

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);

-- Espejo de Navidrome
CREATE TABLE IF NOT EXISTS artists (
  id TEXT PRIMARY KEY,
  navidrome_id TEXT UNIQUE,
  name TEXT,
  synced_at TEXT
);
CREATE TABLE IF NOT EXISTS albums (
  id TEXT PRIMARY KEY,
  navidrome_id TEXT UNIQUE,
  artist_id TEXT,
  name TEXT,
  year INT,
  genre TEXT,
  synced_at TEXT
);
CREATE TABLE IF NOT EXISTS tracks (
  id TEXT PRIMARY KEY,
  navidrome_id TEXT UNIQUE,
  album_id TEXT,
  artist_id TEXT,
  title TEXT,
  duration INT,
  path TEXT,
  synced_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_albums_artist ON albums(artist_id);
CREATE INDEX IF NOT EXISTS idx_tracks_album ON tracks(album_id);
CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist_id);

-- Fichas (artista | album | track)
CREATE TABLE IF NOT EXISTS fichas (
  entity_type TEXT,
  entity_id TEXT,
  facets TEXT,
  description TEXT,
  confidence REAL,
  source TEXT,
  content_hash TEXT,
  updated_at TEXT,
  PRIMARY KEY (entity_type, entity_id)
);

-- Logs / runs
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  module TEXT,
  kind TEXT,
  status TEXT DEFAULT 'running',
  started_at TEXT,
  finished_at TEXT,
  stats TEXT
);
CREATE TABLE IF NOT EXISTS import_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INT,
  file TEXT,
  old_tags TEXT,
  new_tags TEXT,
  match_score REAL,
  album_id TEXT,
  status TEXT,
  created_at TEXT
);

-- Vocabulario controlado
CREATE TABLE IF NOT EXISTS synonyms (term TEXT PRIMARY KEY, canonical TEXT);

CREATE TABLE IF NOT EXISTS lastfm_cache (
  entity_type TEXT, entity_key TEXT, payload TEXT, fetched_at TEXT,
  PRIMARY KEY (entity_type, entity_key)
);

CREATE TABLE IF NOT EXISTS web_cache (
  query TEXT PRIMARY KEY, payload TEXT, fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS lyrics_cache (
  track_id TEXT PRIMARY KEY, lyrics TEXT, fetched_at TEXT
);

-- Full-text search sobre el espejo (para recall sin vectores)
CREATE VIRTUAL TABLE IF NOT EXISTS fts_entities USING fts5(
  entity_type,
  entity_id UNINDEXED,
  text,
  tokenize = 'unicode61 remove_diacritics 2'
);

-- Índice de facetas normalizadas (filtros duros: idioma, país, década, etc.)
CREATE TABLE IF NOT EXISTS facet_index (
  entity_type TEXT,
  entity_id TEXT,
  facet TEXT,
  value TEXT,
  num REAL,
  PRIMARY KEY (entity_type, entity_id, facet, value)
);
CREATE INDEX IF NOT EXISTS idx_facet_lookup ON facet_index(facet, value);
CREATE INDEX IF NOT EXISTS idx_facet_entity ON facet_index(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_facet_num ON facet_index(facet, num);
"""

VEC_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS vec_fichas USING vec0(
  entity_id TEXT PRIMARY KEY,
  entity_type TEXT,
  embedding FLOAT[{dim}]
);
"""


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    settings = get_settings()
    if path is None:
        settings.ensure_dirs()
        path = settings.db_path
    conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _load_vec(conn)
    return conn


_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def get_conn() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            _conn = connect()
            init_db(_conn)
        return _conn


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'vec_dim'"
    ).fetchone()
    current = row["value"] if row else None
    if current is not None and int(current) == dim:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_fichas'"
        ).fetchone()
        if exists:
            return
    conn.execute("DROP TABLE IF EXISTS vec_fichas")
    conn.execute(VEC_SCHEMA.format(dim=dim))
    conn.execute(
        "INSERT INTO meta(key, value) VALUES ('vec_dim', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(dim),),
    )


def init_db(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    conn = conn or get_conn()
    previous = None
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        previous = int(row["value"]) if row else None
    except (sqlite3.OperationalError, TypeError, ValueError):
        previous = None
    conn.executescript(SCHEMA)
    settings = get_settings()
    _ensure_vec_table(conn, settings.embed_dim)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES ('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(SCHEMA_VERSION),),
    )
    seed_synonyms(conn)
    if previous is not None and previous < 2:
        facet_rebuild_all(conn)
    conn.commit()
    return conn


DEFAULT_SYNONYMS: dict[str, str] = {
    "joda": "fiesta",
    "party": "fiesta",
    "festejo": "fiesta",
    "celebracion": "fiesta",
    "tavern": "taberna",
    "pub": "taberna",
    "posada": "taberna",
    "drinking song": "cancion de taberna",
    "epic": "epico",
    "epica": "epico",
    "triste": "melancolico",
    "sad": "melancolico",
    "melancholic": "melancolico",
    "energetic": "energico",
    "fiesta medieval": "taberna",
    "tolkien": "señor de los anillos",
    "lotr": "señor de los anillos",
    "middle earth": "señor de los anillos",
    "tierra media": "señor de los anillos",
    "christmas": "navidad",
    "navideño": "navidad",
    "halloween": "terror",
    "spooky": "terror",
    "love": "amor",
    "romance": "amor",
}


def seed_synonyms(conn: sqlite3.Connection) -> None:
    count = conn.execute("SELECT COUNT(*) AS n FROM synonyms").fetchone()["n"]
    if count:
        return
    conn.executemany(
        "INSERT OR IGNORE INTO synonyms(term, canonical) VALUES (?, ?)",
        DEFAULT_SYNONYMS.items(),
    )


def fts_upsert(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: str,
    text: str,
) -> None:
    conn.execute(
        "DELETE FROM fts_entities WHERE entity_type = ? AND entity_id = ?",
        (entity_type, entity_id),
    )
    conn.execute(
        "INSERT INTO fts_entities(entity_type, entity_id, text) VALUES (?, ?, ?)",
        (entity_type, entity_id, text),
    )


def facet_upsert(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: str,
    facets: dict[str, Any],
) -> None:
    """Reemplaza el índice de facetas normalizadas de una entidad."""
    from app.enrich.vocab import facet_values_from_facets

    conn.execute(
        "DELETE FROM facet_index WHERE entity_type = ? AND entity_id = ?",
        (entity_type, entity_id),
    )
    values = facet_values_from_facets(facets or {})
    rows: list[tuple[str, str, str, str, float | None]] = []
    for facet, pairs in values.items():
        seen: set[str] = set()
        for value, num in pairs:
            if value in seen:
                continue
            seen.add(value)
            rows.append((entity_type, entity_id, facet, value, num))
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO facet_index"
            "(entity_type, entity_id, facet, value, num) VALUES (?, ?, ?, ?, ?)",
            rows,
        )


def facet_search(
    conn: sqlite3.Connection,
    *,
    entity_type: str,
    facets: dict[str, list[str]] | None = None,
    numeric: dict[str, tuple[float | None, float | None]] | None = None,
    require_all: bool = True,
) -> set[str]:
    """IDs de entidades que cumplen TODOS los filtros de facetas.

    `facets`: {facet: [valores]} — match por cualquiera de los valores dados.
    `numeric`: {facet: (min, max)} — rango sobre la columna numérica.
    """
    result: set[str] | None = None
    for facet, values in (facets or {}).items():
        if not values:
            continue
        marks = ",".join("?" * len(values))
        rows = conn.execute(
            f"SELECT DISTINCT entity_id FROM facet_index "
            f"WHERE entity_type = ? AND facet = ? AND value IN ({marks})",
            [entity_type, facet, *values],
        ).fetchall()
        ids = {r["entity_id"] for r in rows}
        result = ids if result is None else (result & ids if require_all else result | ids)
    for facet, (low, high) in (numeric or {}).items():
        if low is None and high is None:
            continue
        clauses = ["entity_type = ?", "facet = ?", "num IS NOT NULL"]
        params: list[Any] = [entity_type, facet]
        if low is not None:
            clauses.append("num >= ?")
            params.append(float(low))
        if high is not None:
            clauses.append("num <= ?")
            params.append(float(high))
        rows = conn.execute(
            f"SELECT DISTINCT entity_id FROM facet_index WHERE {' AND '.join(clauses)}",
            params,
        ).fetchall()
        ids = {r["entity_id"] for r in rows}
        result = ids if result is None else result & ids
    return result if result is not None else set()


def facet_list(
    conn: sqlite3.Connection, entity_type: str, facet: str
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT value, COUNT(*) AS n FROM facet_index "
        "WHERE entity_type = ? AND facet = ? GROUP BY value ORDER BY n DESC",
        (entity_type, facet),
    ).fetchall()
    return [dict(r) for r in rows]


def facet_rebuild_all(conn: sqlite3.Connection) -> int:
    """Reconstruye facet_index desde las fichas. Devuelve fichas procesadas."""
    import json

    rows = conn.execute("SELECT entity_type, entity_id, facets FROM fichas").fetchall()
    conn.execute("DELETE FROM facet_index")
    count = 0
    for row in rows:
        try:
            facets = json.loads(row["facets"] or "{}")
        except json.JSONDecodeError:
            facets = {}
        facet_upsert(conn, row["entity_type"], row["entity_id"], facets)
        count += 1
    conn.commit()
    return count


def vec_upsert(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: str,
    embedding: Sequence[float],
) -> None:
    blob = sqlite_vec.serialize_float32([float(x) for x in embedding])
    conn.execute(
        "DELETE FROM vec_fichas WHERE entity_id = ?", (entity_id,)
    )
    conn.execute(
        "INSERT INTO vec_fichas(entity_id, entity_type, embedding) VALUES (?, ?, ?)",
        (entity_id, entity_type, blob),
    )


def vec_search(
    conn: sqlite3.Connection,
    embedding: Sequence[float],
    limit: int = 20,
    entity_type: str | None = None,
) -> list[dict[str, Any]]:
    blob = sqlite_vec.serialize_float32([float(x) for x in embedding])
    if entity_type:
        rows = conn.execute(
            """
            SELECT entity_id, entity_type, distance
            FROM vec_fichas
            WHERE embedding MATCH ? AND k = ? AND entity_type = ?
            ORDER BY distance
            """,
            (blob, limit, entity_type),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT entity_id, entity_type, distance
            FROM vec_fichas
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (blob, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    entity_type: str | None = None,
) -> list[dict[str, Any]]:
    terms = [t for t in query.replace('"', " ").split() if t]
    if not terms:
        return []
    match = " OR ".join(f'"{t}"*' for t in terms)
    params: list[Any] = [match]
    type_filter = ""
    if entity_type:
        type_filter = "AND entity_type = ?"
        params.append(entity_type)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT entity_type, entity_id, bm25(fts_entities) AS score
        FROM fts_entities
        WHERE fts_entities MATCH ? {type_filter}
        ORDER BY score
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def start_run(conn: sqlite3.Connection, module: str, kind: str = "") -> int:
    cur = conn.execute(
        "INSERT INTO runs(module, kind, started_at, stats) VALUES (?, ?, ?, '{}')",
        (module, kind, utcnow()),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str = "ok",
    stats: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        "UPDATE runs SET status = ?, finished_at = ?, stats = ? WHERE id = ?",
        (status, utcnow(), json.dumps(stats or {}, ensure_ascii=False), run_id),
    )
    conn.commit()


def all_settings(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {r["key"]: r["value"] for r in rows}


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO settings(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()
