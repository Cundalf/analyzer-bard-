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
        self.match = type("Match", (), {"distance": FakeRec()})() if score is not None else None
        if self.match is not None:
            self.match.distance.distance = score
        self.candidates = []
        self.album = "Wishmaster"


@pytest.fixture()
def plugin(tmp_path, monkeypatch):
    """Instancia del plugin de beets con el log apuntando a tmp_path."""
    spec = importlib.util.spec_from_file_location("bardo_plugin", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bardo_plugin"] = module
    spec.loader.exec_module(module)
    monkeypatch.setenv("BARDO_BEETS_LOG", str(tmp_path / "beets.jsonl"))
    return module.BardoLogPlugin()


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
    plugin.on_task_files(
        session=None, task=type("T", (), {"imported_items": lambda self: [item]})()
    )

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


def test_plugin_log_path_env_first(plugin, tmp_path, monkeypatch):
    from app.janitor import bardo as mod

    monkeypatch.setenv("BARDO_BEETS_LOG", str(tmp_path / "override.jsonl"))
    assert mod._log_path() == tmp_path / "override.jsonl"


def test_plugin_clean_bytes():
    from app.janitor import bardo as mod

    assert mod._clean(b"/music/a.flac") == "/music/a.flac"
    assert mod._clean("texto") == "texto"
    assert mod._clean(7) == 7


def test_plugin_distance_of_none_and_plain():
    from app.janitor import bardo as mod

    assert mod._distance_of(None) is None

    class Plain:
        distance = 0.4

    assert mod._distance_of(Plain()) == pytest.approx(0.4)


@pytest.mark.anyio
async def test_plugin_empty_queue_noop(plugin, tmp_path):
    class Task:
        items: list = []
        choice_flag = None
        skip = False
        album = ""
        match = None
        candidates: list = []

    task = Task()
    plugin.on_task_created(session=None, task=task)
    plugin.on_task_choice(session=None, task=task)
    plugin.on_write(item={}, path=b"/x", tags={})
    assert plugin.commands() == []


def test_plugin_skip_status_recorded(plugin, tmp_path):
    import json

    class Item(dict):
        path = b"/music/skip.flac"

    item = Item(artist="?")

    class Task:
        items = [item]
        choice_flag = type("F", (), {"name": "SKIP"})()
        skip = True
        album = ""
        match = None
        candidates: list = []

    task = Task()
    plugin.on_task_created(session=None, task=task)
    plugin.on_task_choice(session=None, task=task)
    log = tmp_path / "beets.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["status"] == "SKIP"
    assert records[0]["file"] == "/music/skip.flac"


def test_plugin_album_fallback_from_item(plugin):
    class Item(dict):
        path = b"/m/a.flac"

    item = Item(album="DesdeItem")
    task = type(
        "T",
        (),
        {
            "items": [item],
            "choice_flag": type("F", (), {"name": "ASIS"})(),
            "skip": False,
            "album": "",
            "match": None,
            "candidates": [],
        },
    )()
    plugin.on_task_created(session=None, task=task)
    plugin.on_task_choice(session=None, task=task)
    assert plugin._albums["/m/a.flac"] == "DesdeItem"


def test_plugin_emit_ignores_path_without_before(plugin, tmp_path):
    plugin._emit("/desconocido", {"artist": "x"})
    assert not (tmp_path / "beets.jsonl").exists()


def test_plugin_log_path_default(plugin, monkeypatch):
    from beets import config

    from app.janitor import bardo as mod

    monkeypatch.delenv("BARDO_BEETS_LOG", raising=False)
    config["bardo"]["logpath"] = "/tmp/desde-config.jsonl"
    assert mod._log_path() == Path("/tmp/desde-config.jsonl")


def test_plugin_log_path_final_fallback(plugin, monkeypatch):
    from beets import config

    from app.janitor import bardo as mod

    monkeypatch.delenv("BARDO_BEETS_LOG", raising=False)
    config["bardo"]["logpath"] = ""
    assert mod._log_path() == Path("/data/runs/beets.jsonl")


def test_plugin_distance_not_number_not_object():
    from app.janitor import bardo as mod

    class Weird:
        distance = "texto"

    assert mod._distance_of(Weird()) is None


def test_plugin_on_write_unknown_path(plugin):
    plugin.on_write(item={}, path=b"/nuevo.flac", tags={})


def test_plugin_emit_twice_no_duplicate(plugin, tmp_path):
    plugin._before["/a"] = {"x": 1}
    plugin._emit("/a", {"x": 2})
    plugin._emit("/a", {"x": 3})
    lines = (tmp_path / "beets.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_plugin_distance_of_none_attr():
    from app.janitor import bardo as mod

    class NoDistance:
        pass

    assert mod._distance_of(NoDistance()) is None


def test_plugin_on_write_emits(plugin, tmp_path):
    plugin._before["/music/a.flac"] = {"artist": "?"}
    plugin.on_write(
        item={"artist": "Nuevo", "path": b"/music/a.flac"},
        path=b"/music/a.flac",
        tags={},
    )
    import json

    records = [json.loads(line) for line in (tmp_path / "beets.jsonl").read_text().splitlines()]
    assert records[0]["new_tags"]["artist"] == "Nuevo"
