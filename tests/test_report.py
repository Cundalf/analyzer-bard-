from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import seed_library

from app.janitor.report import (
    RunReport,
    TagDiff,
    ingest_into_db,
    library_health,
    read_jsonl,
    write_jsonl,
)

# ------------------------------------------------------------ TagDiff


def test_tagdiff_changed_and_fields():
    diff = TagDiff(
        file="/m/a.flac",
        old_tags={"artist": "?", "title": "T", "year": 0},
        new_tags={"artist": "X", "title": "T", "year": 1999},
    )
    assert diff.changed is True
    assert diff.changed_fields() == ["artist", "year"]


def test_tagdiff_no_changes():
    diff = TagDiff(file="/m/a.flac", old_tags={"a": 1}, new_tags={"a": 1})
    assert diff.changed is False
    assert diff.changed_fields() == []


def test_tagdiff_added_and_removed_fields():
    diff = TagDiff(file="/a", old_tags={"x": 1}, new_tags={"y": 2})
    assert diff.changed_fields() == ["x", "y"]


def test_tagdiff_roundtrip():
    diff = TagDiff(
        file="/a",
        old_tags={"a": 1},
        new_tags={"a": 2},
        match_score=0.5,
        album_id="al",
        status="APPLY",
    )
    restored = TagDiff.from_dict(diff.as_dict())
    assert restored == diff


def test_tagdiff_from_dict_defaults():
    restored = TagDiff.from_dict({})
    assert restored.file == ""
    assert restored.old_tags == {}
    assert restored.match_score is None
    assert restored.status == "unknown"


# ------------------------------------------------------------ JSONL


def test_write_and_read_jsonl(tmp_path: Path):
    report = RunReport(
        items=[
            TagDiff(file="/a", old_tags={}, new_tags={"artist": "X"}),
            TagDiff(file="/b", old_tags={}, new_tags={}, status="ASIS"),
        ]
    )
    path = write_jsonl(report, tmp_path / "sub" / "run.jsonl")
    assert path.exists()
    loaded = read_jsonl(path)
    assert [d.file for d in loaded] == ["/a", "/b"]


def test_read_jsonl_skips_blank_and_corrupt(tmp_path: Path):
    path = tmp_path / "r.jsonl"
    path.write_text(
        '{"file": "/a", "old_tags": {}, "new_tags": {}}\n'
        "\n"
        "no-es-json\n"
        '{"file": "/b", "old_tags": {}, "new_tags": {}}\n'
    )
    loaded = read_jsonl(path)
    assert [d.file for d in loaded] == ["/a", "/b"]


def test_read_jsonl_missing_file(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "nope.jsonl")


def test_report_summary_counts():
    report = RunReport()
    report.add(TagDiff(file="/a", old_tags={}, new_tags={}, match_score=0.9, status="APPLY"))
    report.add(TagDiff(file="/b", old_tags={}, new_tags={"x": 1}, match_score=0.5, status="ASIS"))
    report.add(TagDiff(file="/c", old_tags={}, new_tags={}, match_score=None, status="SKIP"))
    report.errors.append("e")
    summary = report.summary()
    assert summary["total"] == 3
    assert summary["changed"] == 1
    assert summary["strong_matches"] == 1
    assert summary["by_status"] == {"APPLY": 1, "ASIS": 1, "SKIP": 1}
    assert summary["errors"] == 1


def test_report_summary_empty():
    assert RunReport().summary()["total"] == 0


def test_ingest_into_db(conn):
    diffs = [
        TagDiff(
            file="/a",
            old_tags={"x": 1},
            new_tags={"x": 2},
            match_score=0.9,
            album_id="al",
            status="APPLY",
        ),
        TagDiff(file="/b", old_tags={}, new_tags={}, status="SKIP"),
    ]
    count = ingest_into_db(conn, 7, diffs)
    assert count == 2
    row = conn.execute("SELECT * FROM import_log WHERE run_id = 7 ORDER BY id").fetchone()
    assert row["file"] == "/a"
    assert json.loads(row["old_tags"]) == {"x": 1}
    assert row["match_score"] == 0.9
    assert row["album_id"] == "al"


def test_ingest_empty(conn):
    assert ingest_into_db(conn, 1, []) == 0


# ------------------------------------------------------------ health


def test_library_health_empty(conn):
    health = library_health(conn)
    assert health["albums"] == 0
    assert health["health"] == 100.0


def test_library_health_detects_problems(conn):
    seed_library(conn)
    conn.execute(
        "INSERT INTO albums(id, navidrome_id, artist_id, name, year) "
        "VALUES ('album:bad1', 'bad1', NULL, '   ', NULL)"
    )
    conn.execute(
        "INSERT INTO albums(id, navidrome_id, artist_id, name, year) "
        "VALUES ('album:bad2', 'bad2', NULL, 'Track 04', 2000)"
    )
    conn.commit()
    health = library_health(conn)
    assert health["albums"] == 4
    assert health["tracks"] == 4
    assert health["no_year"] == 1
    assert health["unknown_artist"] == 2
    assert health["blank_name"] == 1
    assert health["generic_name"] >= 1
    assert 0 <= health["health"] <= 100


def test_library_health_unknown_artist_by_name(conn):
    conn.execute(
        "INSERT INTO artists(id, navidrome_id, name) VALUES ('artist:u', 'u', '[Unknown Artist]')"
    )
    conn.execute(
        "INSERT INTO albums(id, navidrome_id, artist_id, name, year) "
        "VALUES ('album:u', 'u', 'artist:u', 'Album', 2000)"
    )
    conn.commit()
    health = library_health(conn)
    assert health["unknown_artist"] == 1


def test_library_health_includes_last_run(conn):
    from app.db import finish_run, start_run

    run_id = start_run(conn, "janitor")
    finish_run(conn, run_id, "ok", {"x": 1})
    health = library_health(conn)
    assert health["last_run"]["id"] == run_id


def test_library_health_no_unknown_collision(conn):
    conn.execute(
        "INSERT INTO artists(id, navidrome_id, name) VALUES ('artist:k', 'k', 'The Unknowns')"
    )
    conn.execute(
        "INSERT INTO albums(id, navidrome_id, artist_id, name, year) "
        "VALUES ('album:k', 'k', 'artist:k', 'Disco', 2000)"
    )
    conn.commit()
    health = library_health(conn)
    assert health["unknown_artist"] == 0
