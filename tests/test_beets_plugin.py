from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parents[1] / "app" / "janitor" / "bardo.py"


class FakeItem(dict):
    path = b"/music/cancion.flac"


class FakeRec:
    pass


class FakeTask:
    def __init__(self, skip: bool = False, score: float | None = None):
        self.items = [FakeItem(artist="[Unknown]", title="Track 01", year=0)]
        self.choice_flag = type("Flag", (), {"name": "APPLY"})()
        self.skip = skip
        self.match = (
            type("Match", (), {"distance": FakeRec()})() if score is not None else None
        )
        if self.match is not None:
            self.match.distance.distance = score
        self.candidates = []
        self.album = "Wishmaster"


@pytest.fixture()
def plugin(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("bardo_plugin", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bardo_plugin"] = module
    spec.loader.exec_module(module)
    monkeypatch.setenv("BARDO_BEETS_LOG", str(tmp_path / "beets.jsonl"))
    instance = module.BardoLogPlugin()
    return instance


def read_log(tmp_path: Path) -> list[dict]:
    import json

    log = tmp_path / "beets.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_distance_of_handles_old_and_new_api():
    from app.janitor import bardo as mod

    assert mod._distance_of(None) is None

    class OldMatch:
        distance = 0.87

    assert mod._distance_of(OldMatch()) == pytest.approx(0.87)

    class NewDistance:
        distance = 0.55

    class NewMatch:
        distance = NewDistance()

    assert mod._distance_of(NewMatch()) == pytest.approx(0.55)

    class Rec:
        pass

    assert mod._distance_of(type("X", (), {"distance": Rec()})()) is None


def test_match_score_uses_match_or_candidates():
    from app.janitor import bardo as mod

    task = FakeTask(score=0.9)
    assert mod._match_score(task) == pytest.approx(0.9)

    class C:
        distance = 0.3

    class TaskNoMatch:
        match = None
        candidates = [C()]

    assert mod._match_score(TaskNoMatch()) == pytest.approx(0.3)


def test_plugin_writes_diff(plugin, tmp_path):
    task = FakeTask(score=0.9)
    plugin.on_task_created(session=None, task=task)
    plugin.on_task_choice(session=None, task=task)
    item = task.items[0]
    item["artist"] = "Nightwish"
    item["album"] = "Wishmaster"
    plugin.on_task_files(session=None, task=type("T", (), {"imported_items": lambda self: [item]})())

    records = read_log(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record["old_tags"]["artist"] == "[Unknown]"
    assert record["new_tags"]["artist"] == "Nightwish"
    assert record["album_id"] == "Wishmaster"
    assert record["match_score"] == pytest.approx(0.9)
    assert record["status"] == "APPLY"


def test_plugin_skips_duplicates_emit_once(plugin, tmp_path):
    task = FakeTask()
    plugin.on_task_created(session=None, task=task)
    plugin.on_task_choice(session=None, task=task)
    item = task.items[0]

    class T:
        def imported_items(self):
            return [item]

    plugin.on_task_files(session=None, task=T())
    plugin.on_task_files(session=None, task=T())
    assert len(read_log(tmp_path)) == 1
