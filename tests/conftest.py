from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from app import db as db_mod


@pytest.fixture(autouse=True)
def isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Aísla cada test: DATA_DIR temporal y endpoints muertos por defecto."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MUSIC_DIR", str(tmp_path / "music"))
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("SUBSONIC_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("SUBSONIC_USER", "admin")
    monkeypatch.setenv("SUBSONIC_PASS", "secret")
    monkeypatch.delenv("JANITOR_ENABLED", raising=False)
    monkeypatch.delenv("ACOUSTID_KEY", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    db_mod._conn = None
    yield
    if db_mod._conn is not None:
        db_mod._conn.close()
        db_mod._conn = None
    get_settings.cache_clear()


@pytest.fixture()
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = db_mod.connect(tmp_path / "test.db")
    db_mod.init_db(connection)
    yield connection
    connection.close()


@pytest.fixture()
def settings(tmp_path: Path):
    from app.config import Settings

    s = Settings(
        data_dir=str(tmp_path / "data"),
        subsonic_url="http://navidrome.test:4533",
        subsonic_user="admin",
        subsonic_pass="secret",
        ollama_url="http://ollama.test:11434",
        embed_dim=768,
    )
    s.ensure_dirs()
    return s


def seed_library(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO artists(id, navidrome_id, name) VALUES ('artist:a1', 'a1', 'Wind Rose')"
    )
    conn.execute(
        "INSERT INTO artists(id, navidrome_id, name) VALUES ('artist:a2', 'a2', 'Blind Guardian')"
    )
    conn.execute(
        """
        INSERT INTO albums(id, navidrome_id, artist_id, name, year, genre)
        VALUES ('album:al1', 'al1', 'artist:a1', 'Wintersaga', 2019, 'folk metal')
        """
    )
    conn.execute(
        """
        INSERT INTO albums(id, navidrome_id, artist_id, name, year, genre)
        VALUES ('album:al2', 'al2', 'artist:a2', 'Nightfall in Middle-Earth', 1998, 'power metal')
        """
    )
    tracks = [
        ("t1", "al1", "a1", "Drunken Dwarves", "artist:a1"),
        ("t2", "al1", "a1", "Mine Mine Mine!", "artist:a1"),
        ("t3", "al2", "a2", "Nightfall", "artist:a2"),
        ("t4", "al2", "a2", "Into the Storm", "artist:a2"),
    ]
    for tid, album_nid, artist_nid, title, artist_pk in tracks:
        conn.execute(
            """
            INSERT INTO tracks(id, navidrome_id, album_id, artist_id, title, duration)
            VALUES (?, ?, ?, ?, ?, 240)
            """,
            (
                f"track:{tid}",
                tid,
                {"al1": "album:al1", "al2": "album:al2"}[album_nid],
                artist_pk,
                title,
            ),
        )
    conn.execute(
        """
        INSERT INTO fichas(entity_type, entity_id, facets, description, confidence, source, content_hash)
        VALUES ('album', 'al1',
                '{"moods": ["fiesta", "epico"], "themes": ["cerveza", "enanos"]}',
                'Folk metal festivo de enanos con canciones para beber.',
                0.9, 'llm', 'h1')
        """
    )
    db_mod.fts_upsert(
        conn,
        "album",
        "al1",
        "Folk metal festivo de enanos con canciones para beber. "
        "moods: fiesta, epico themes: cerveza, enanos",
    )
    conn.commit()


class FakeOllama:
    """Doble de OllamaClient para tests de enriquecimiento/agente."""

    def __init__(self, chat_responses=None, embedding=None, text=None):
        self.chat_responses = list(chat_responses or [])
        self.embedding = embedding or [0.1] * 768
        self.text = text or ""
        self.calls: list[dict] = []
        self.closed = False

    async def chat(self, messages, **kwargs):
        self.calls.append({"kind": "chat", "messages": messages, **kwargs})
        if self.chat_responses:
            item = self.chat_responses.pop(0)
            if isinstance(item, Exception):
                raise item
            if isinstance(item, dict) and "message" in item:
                return item
            return {"message": item}
        return {"message": {"role": "assistant", "content": self.text}}

    async def chat_json(self, messages, **kwargs):
        self.calls.append({"kind": "chat_json", "messages": messages, **kwargs})
        if self.chat_responses:
            item = self.chat_responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return {}

    async def embed(self, texts, **kwargs):
        return [list(self.embedding) for _ in texts]

    async def embed_one(self, text, **kwargs):
        self.calls.append({"kind": "embed_one", "text": text})
        if self.chat_responses and isinstance(self.chat_responses[0], Exception):
            raise self.chat_responses.pop(0)
        return list(self.embedding)

    async def list_models(self):
        return ["test-model"]

    async def ping(self):
        return True

    async def close(self):
        self.closed = True


class FakeSubsonic:
    def __init__(self, playlists=None, fail: bool = False):
        self.playlists = list(playlists or [])
        self.fail = fail
        self.created: list[tuple[str, list[str]]] = []
        self.closed = False

    def ping(self):
        if self.fail:
            raise RuntimeError("subsonic down")
        return {"status": "ok"}

    def get_playlists(self):
        if self.fail:
            raise RuntimeError("subsonic down")
        return self.playlists

    def create_playlist(self, name, track_ids):
        if self.fail:
            raise RuntimeError("subsonic down")
        self.created.append((name, list(track_ids)))

        class P:
            id = "pl-test"
            song_count = len(track_ids)

        P.name = name
        return P()

    def start_scan(self, full=False):
        if self.fail:
            raise RuntimeError("scan failed")
        return {"scanning": True}

    def close(self):
        self.closed = True
