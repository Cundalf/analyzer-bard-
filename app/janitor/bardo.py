"""Plugin de beets para Bardo: captura metadata antes/después en JSONL.

beets lo carga vía `pluginpath` (por defecto /app/app/janitor en el container).
Registra listeners en import_task_start (antes), import_task_choice (decisión),
write (después de escribir tags) e import_task_files (cierre del import).
Escribe un JSONL con {file, old_tags, new_tags, match_score, album_id, status}.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from beets import config
from beets.plugins import BeetsPlugin

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


def _log_path() -> Path:
    env_path = os.environ.get("BARDO_BEETS_LOG")
    if env_path:
        return Path(env_path)
    return Path(config["bardo"]["logpath"].as_str() or "/data/runs/beets.jsonl")


def _clean(value: Any) -> Any:
    if isinstance(value, bytes):
        return os.fsdecode(value)
    return value


def _tags_of(item: Any) -> dict[str, Any]:
    return {field: _clean(item.get(field)) for field in TAG_FIELDS}


def _distance_of(obj: Any) -> float | None:
    """Extrae la distancia normalizada (0..1) de un Match/Candidate.

    beets 2.14 cambió la API: `task.rec` es un enum `Recommendation` y la
    distancia vive en `task.match.distance` o `task.candidates[0].distance`
    (objetos `Distance` con propiedad `.distance`).
    """
    if obj is None:
        return None
    raw = getattr(obj, "distance", None)
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    value = getattr(raw, "distance", None)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _match_score(task: Any) -> float | None:
    for candidate in (
        getattr(task, "match", None),
        next(iter(getattr(task, "candidates", None) or []), None),
    ):
        score = _distance_of(candidate)
        if score is not None:
            return score
    return None


class BardoLogPlugin(BeetsPlugin):
    def __init__(self) -> None:
        super().__init__()
        self.config.add({"logpath": "/data/runs/beets.jsonl"})
        self._before: dict[str, dict[str, Any]] = {}
        self._match_scores: dict[str, float] = {}
        self._albums: dict[str, str] = {}
        self._statuses: dict[str, str] = {}
        self._emitted: set[str] = set()

        self.register_listener("import_task_created", self.on_task_created)
        self.register_listener("import_task_start", self.on_task_created)
        self.register_listener("import_task_choice", self.on_task_choice)
        self.register_listener("write", self.on_write)
        self.register_listener("import_task_files", self.on_task_files)

    def on_task_created(self, session: Any, task: Any) -> None:
        for item in task.items:
            self._before.setdefault(_clean(item.path), _tags_of(item))

    def on_task_choice(self, session: Any, task: Any) -> None:
        status = getattr(task.choice_flag, "name", None) or "unknown"
        score = _match_score(task)
        for item in task.items:
            path = _clean(item.path)
            self._statuses[path] = str(status)
            if score is not None:
                self._match_scores[path] = score
            album = getattr(task, "album", None)
            if not album:
                album = item.get("album", "")
            self._albums[path] = str(album or "")
        if getattr(task, "skip", False):
            for item in task.items:
                self._emit(_clean(item.path), _tags_of(item), status=str(status))

    def on_write(self, item: Any, path: Any, tags: dict[str, Any]) -> None:
        path_str = _clean(path)
        if path_str not in self._before:
            return
        self._emit(path_str, _tags_of(item))

    def on_task_files(self, session: Any, task: Any) -> None:
        for item in task.imported_items():
            path = _clean(item.path)
            if path in self._emitted:
                continue
            self._emit(path, _tags_of(item))

    def _emit(
        self,
        path: str,
        new_tags: dict[str, Any],
        *,
        status: str | None = None,
    ) -> None:
        if path in self._emitted:
            return
        old = self._before.get(path)
        if old is None:
            return
        record = {
            "file": path,
            "old_tags": old,
            "new_tags": new_tags,
            "match_score": self._match_scores.get(path),
            "album_id": self._albums.get(path, ""),
            "status": status or self._statuses.get(path, "imported"),
        }
        logpath = _log_path()
        logpath.parent.mkdir(parents=True, exist_ok=True)
        with logpath.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._emitted.add(path)

    def commands(self) -> list[Any]:
        return []
