from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("bardo.janitor.report")

TAG_FIELDS = (
    "artist",
    "albumartist",
    "album",
    "title",
    "year",
    "genre",
    "track",
    "disc",
    "mb_trackid",
    "mb_albumid",
    "mb_artistid",
)


@dataclass
class TagDiff:
    file: str
    old_tags: dict[str, Any] = field(default_factory=dict)
    new_tags: dict[str, Any] = field(default_factory=dict)
    match_score: float | None = None
    album_id: str = ""
    status: str = "unknown"

    @property
    def changed(self) -> bool:
        return self.old_tags != self.new_tags

    def changed_fields(self) -> list[str]:
        keys = set(self.old_tags) | set(self.new_tags)
        return sorted(
            k for k in keys if self.old_tags.get(k) != self.new_tags.get(k)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "old_tags": self.old_tags,
            "new_tags": self.new_tags,
            "match_score": self.match_score,
            "album_id": self.album_id,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TagDiff":
        return cls(
            file=data.get("file", ""),
            old_tags=data.get("old_tags", {}) or {},
            new_tags=data.get("new_tags", {}) or {},
            match_score=data.get("match_score"),
            album_id=data.get("album_id", "") or "",
            status=data.get("status", "unknown"),
        )


@dataclass
class RunReport:
    run_id: int | None = None
    module: str = "janitor"
    kind: str = ""
    items: list[TagDiff] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add(self, diff: TagDiff) -> None:
        self.items.append(diff)

    def summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for item in self.items:
            by_status[item.status] = by_status.get(item.status, 0) + 1
        matched = [i for i in self.items if i.match_score is not None]
        strong = [i for i in matched if (i.match_score or 0) >= 0.8]
        changed = [i for i in self.items if i.changed]
        return {
            "total": len(self.items),
            "changed": len(changed),
            "strong_matches": len(strong),
            "by_status": by_status,
            "errors": len(self.errors),
        }


def write_jsonl(report: RunReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for item in report.items:
            fh.write(json.dumps(item.as_dict(), ensure_ascii=False) + "\n")
    return target


def read_jsonl(path: str | Path) -> list[TagDiff]:
    items: list[TagDiff] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(TagDiff.from_dict(json.loads(line)))
            except json.JSONDecodeError:
                log.warning("skipping invalid jsonl line")
    return items


def ingest_into_db(conn: Any, run_id: int, diffs: Iterable[TagDiff]) -> int:
    from app.db import utcnow

    count = 0
    now = utcnow()
    for diff in diffs:
        conn.execute(
            """
            INSERT INTO import_log(run_id, file, old_tags, new_tags,
                                   match_score, album_id, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                diff.file,
                json.dumps(diff.old_tags, ensure_ascii=False),
                json.dumps(diff.new_tags, ensure_ascii=False),
                diff.match_score,
                diff.album_id,
                diff.status,
                now,
            ),
        )
        count += 1
    conn.commit()
    return count


def library_health(conn: Any) -> dict[str, Any]:
    total = conn.execute("SELECT COUNT(*) AS n FROM albums").fetchone()["n"]
    no_year = conn.execute(
        "SELECT COUNT(*) AS n FROM albums WHERE year IS NULL OR year = 0"
    ).fetchone()["n"]
    unknown_artist = conn.execute(
        """
        SELECT COUNT(*) AS n FROM albums
        WHERE artist_id IS NULL
           OR artist_id = ''
           OR artist_id IN (SELECT id FROM artists WHERE name LIKE '[%nknown%')
        """
    ).fetchone()["n"]
    blank_name = conn.execute(
        "SELECT COUNT(*) AS n FROM albums WHERE TRIM(COALESCE(name, '')) = ''"
    ).fetchone()["n"]
    generic_name = conn.execute(
        """
        SELECT COUNT(*) AS n FROM albums
        WHERE name LIKE 'Track %' OR name LIKE '%Unknown%'
           OR name LIKE '%track%' AND name NOT LIKE '%tracks%'
        """
    ).fetchone()["n"]
    tracks = conn.execute("SELECT COUNT(*) AS n FROM tracks").fetchone()["n"]
    last_run = conn.execute(
        "SELECT id, status, started_at, finished_at, stats FROM runs "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    issues = no_year + unknown_artist + blank_name + generic_name
    score = 100.0 if total == 0 else max(0.0, 100.0 * (1 - issues / total))
    return {
        "albums": total,
        "tracks": tracks,
        "no_year": no_year,
        "unknown_artist": unknown_artist,
        "blank_name": blank_name,
        "generic_name": generic_name,
        "issues": issues,
        "health": round(min(100.0, score), 1),
        "last_run": dict(last_run) if last_run else None,
    }
