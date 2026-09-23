from __future__ import annotations

from pathlib import Path

from tests.conftest import seed_library

from app.janitor.report import (
    TagDiff,
    ingest_into_db,
    library_health,
    read_jsonl,
    write_jsonl,
)
from app.janitor.wav2flac import scan_wavs


def test_scan_wavs_finds_files(tmp_path: Path):
    (tmp_path / "a.wav").write_bytes(b"RIFF")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.WAV").write_bytes(b"RIFF")
    (tmp_path / "c.flac").write_bytes(b"fLaC")
    result = scan_wavs(tmp_path)
    assert result.count == 2
    assert result.total_size == 8
    names = {f.path.name for f in result.findings}
    assert names == {"a.wav", "b.WAV"}


def test_scan_wavs_missing_dir():
    import pytest

    with pytest.raises(FileNotFoundError):
        scan_wavs("/ruta/que/no/existe")


def test_tagdiff_changed_fields():
    diff = TagDiff(
        file="/music/x.flac",
        old_tags={"artist": "[Unknown]", "title": "Track 01", "year": 0},
        new_tags={"artist": "Wind Rose", "title": "Drunken Dwarves", "year": 2019},
        match_score=0.94,
    )
    assert diff.changed
    assert diff.changed_fields() == ["artist", "title", "year"]


def test_jsonl_roundtrip(tmp_path: Path):
    diff = TagDiff(
        file="/music/x.flac",
        old_tags={"artist": "?"},
        new_tags={"artist": "Wind Rose"},
        match_score=0.9,
        status="APPLY",
    )
    path = write_jsonl(type("R", (), {"items": [diff]})(), tmp_path / "run.jsonl")
    loaded = read_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0].file == "/music/x.flac"
    assert loaded[0].new_tags["artist"] == "Wind Rose"


def test_ingest_into_db(conn):
    diffs = [
        TagDiff(file="/a.flac", old_tags={}, new_tags={"artist": "X"}, status="APPLY"),
        TagDiff(file="/b.flac", old_tags={}, new_tags={}, status="ASIS"),
    ]
    count = ingest_into_db(conn, 1, diffs)
    assert count == 2
    rows = conn.execute("SELECT COUNT(*) AS n FROM import_log").fetchone()["n"]
    assert rows == 2


def test_library_health(conn):
    seed_library(conn)
    conn.execute(
        "INSERT INTO albums(id, navidrome_id, artist_id, name, year) VALUES "
        "('album:bad', 'bad', NULL, '   ', NULL)"
    )
    conn.commit()
    health = library_health(conn)
    assert health["albums"] == 3
    assert health["tracks"] == 4
    assert health["no_year"] == 1
    assert health["blank_name"] == 1
    assert 0 <= health["health"] <= 100
