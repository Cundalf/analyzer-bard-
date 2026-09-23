from __future__ import annotations

import json
import sys

import pytest
from tests.conftest import FakeSubsonic

from app import cli


def run(args, monkeypatch, capsys):
    code = cli.main(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ------------------------------------------------------------ parser


def test_parser_requires_command():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_parser_all_commands():
    parser = cli.build_parser()
    for argv in (
        ["init"],
        ["ping"],
        ["sync"],
        ["health"],
        ["janitor"],
        ["enrich"],
        ["index"],
        ["playlist", "x"],
        ["serve"],
    ):
        args = parser.parse_args(argv)
        assert args.func is not None


def test_janitor_flags():
    args = cli.build_parser().parse_args(
        ["janitor", "/music", "-p", "-t", "--skip-wav", "--no-rescan"]
    )
    assert args.music == "/music"
    assert args.pretend and args.timid and args.skip_wav and args.no_rescan


def test_enrich_flags():
    args = cli.build_parser().parse_args(
        ["enrich", "--no-artists", "--no-albums", "--no-tracks", "--limit", "5", "--force"]
    )
    assert args.no_artists and args.no_albums and args.no_tracks
    assert args.limit == 5 and args.force


def test_index_flags():
    args = cli.build_parser().parse_args(["index", "--force", "--limit", "3"])
    assert args.force and args.limit == 3


def test_playlist_flags():
    args = cli.build_parser().parse_args(["playlist", "taberna", "--save", "--no-agent"])
    assert args.prompt == "taberna" and args.save and args.no_agent


# ------------------------------------------------------------ comandos


def test_cmd_init(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from app.config import get_settings

    get_settings.cache_clear()
    code, out, _ = run(["init"], monkeypatch, capsys)
    assert code == 0
    assert "DB lista" in out


def test_cmd_ping(monkeypatch, capsys, settings):
    monkeypatch.setattr(
        "app.subsonic.SubsonicClient",
        lambda s=None: FakeSubsonicWithPing(),
    )
    code, out, _ = run(["ping"], monkeypatch, capsys)
    assert code == 0
    assert "ok" in out


class FakeSubsonicWithPing(FakeSubsonic):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def ping(self):
        return {"status": "ok", "type": "navidrome"}


def test_cmd_sync(monkeypatch, capsys):
    calls = {}

    def fake_sync(conn, client, with_tracks=True):
        calls["with_tracks"] = with_tracks
        return {"artists": 1, "albums": 2, "tracks": 3}

    monkeypatch.setattr("app.subsonic.sync_library", fake_sync)
    monkeypatch.setattr("app.subsonic.SubsonicClient", lambda s=None: FakeSubsonicWithPing())
    code, out, _ = run(["sync"], monkeypatch, capsys)
    assert code == 0
    assert json.loads(out) == {"artists": 1, "albums": 2, "tracks": 3}
    assert calls["with_tracks"] is True

    code, _, _ = run(["sync", "--no-tracks"], monkeypatch, capsys)
    assert calls["with_tracks"] is False


def test_cmd_health(monkeypatch, capsys):
    code, out, _ = run(["health"], monkeypatch, capsys)
    assert code == 0
    assert "health" in json.loads(out)


def test_cmd_janitor(monkeypatch, capsys):
    class Result:
        status = "ok"

        def as_dict(self):
            return {"status": "ok", "errors": []}

    def fake_janitor(**kwargs):
        if kwargs.get("progress"):
            kwargs["progress"]("start", {"run_id": 1})
        return Result()

    monkeypatch.setattr("app.janitor.runner.run_janitor", fake_janitor)
    code, out, _ = run(["janitor", "-p"], monkeypatch, capsys)
    assert code == 0
    assert "[start]" in out
    assert '"status": "ok"' in out


def test_cmd_janitor_error_exit_code(monkeypatch, capsys):
    class Result:
        status = "error"

        def as_dict(self):
            return {"status": "error", "errors": ["x"]}

    monkeypatch.setattr("app.janitor.runner.run_janitor", lambda **kwargs: Result())
    code, _, _ = run(["janitor", "-p"], monkeypatch, capsys)
    assert code == 1


def test_cmd_enrich(monkeypatch, capsys):
    async def fake_enrich(conn, **kwargs):
        return {"run_id": 1, "artists": 2, "albums": 1, "tracks": 0}

    monkeypatch.setattr("app.enrich.pipeline.enrich_library", fake_enrich)
    code, out, _ = run(["enrich", "--limit", "2"], monkeypatch, capsys)
    assert code == 0
    assert json.loads(out)["artists"] == 2


def test_cmd_index(monkeypatch, capsys):
    async def fake_index(conn, **kwargs):
        return {"total": 1, "embedded": 1}

    monkeypatch.setattr("app.index.build.build_index", fake_index)
    code, out, _ = run(["index"], monkeypatch, capsys)
    assert code == 0
    assert json.loads(out)["embedded"] == 1


def test_cmd_playlist_without_save(monkeypatch, capsys):
    async def fake_generate(conn, prompt, **kwargs):
        if kwargs.get("progress"):
            kwargs["progress"]("plan", {"intent": "playlist"})
        return {"playlist_name": "P", "track_ids": ["t1"], "mode": "rerank"}

    monkeypatch.setattr("app.agent.runner.generate_playlist", fake_generate)
    monkeypatch.setattr("app.subsonic.SubsonicClient", lambda s=None: FakeSubsonicWithPing())
    code, out, _ = run(["playlist", "taberna", "--no-agent"], monkeypatch, capsys)
    assert code == 0
    payload = out.split("]\n", 1)[-1] if "]\n" in out else out
    assert '"playlist_name": "P"' in payload


def test_cmd_serve(monkeypatch, capsys):
    captured = {}

    def fake_run(app_path, **kwargs):
        captured["app"] = app_path
        captured.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    code, _, _ = run(["serve", "--port", "9999", "--host", "127.0.0.1"], monkeypatch, capsys)
    assert code == 0
    assert captured["app"] == "app.main:app"
    assert captured["port"] == 9999


def test_main_handles_exception(monkeypatch, capsys):
    def boom(args):
        raise RuntimeError("algo salió mal")

    monkeypatch.setattr(cli, "cmd_health", boom)
    code, _, err = run(["health"], monkeypatch, capsys)
    assert code == 1
    assert "algo salió mal" in err


def test_main_handles_keyboard_interrupt(monkeypatch, capsys):
    def interrupt(args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "cmd_health", interrupt)
    code, _, _ = run(["health"], monkeypatch, capsys)
    assert code == 130


def test_setup_logging_verbose():
    import logging

    cli._setup_logging(True)
    assert logging.getLogger().level == logging.DEBUG
    cli._setup_logging(False)
    assert logging.getLogger().level == logging.INFO


def test_main_verbose_reraises(monkeypatch, capsys):
    from app import cli

    def boom(args):
        raise RuntimeError("detalle")

    monkeypatch.setattr(cli, "cmd_health", boom)
    with pytest.raises(RuntimeError):
        cli.main(["-v", "health"])


def test_cli_main_guard(monkeypatch):
    import runpy

    monkeypatch.setattr(sys, "argv", ["bardo", "health"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("app.cli", run_name="__main__")
    assert exc.value.code == 0


def test_cli_enrich_audio_flag(monkeypatch, capsys, tmp_path):
    from app import cli
    from app.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    captured = {}

    async def fake_enrich(conn, **kwargs):
        captured["settings"] = kwargs.get("settings")
        return {"run_id": 1}

    monkeypatch.setattr("app.enrich.pipeline.enrich_library", fake_enrich)
    code = cli.main(["enrich", "--audio", "--no-lyrics"])
    assert code == 0
    assert captured["settings"].analyze_audio is True


def test_cli_enrich_without_client_when_no_lyrics(monkeypatch, tmp_path):
    from app import cli
    from app.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    captured = {}

    async def fake_enrich(conn, **kwargs):
        captured["client"] = kwargs.get("client")
        return {"run_id": 1}

    monkeypatch.setattr("app.enrich.pipeline.enrich_library", fake_enrich)
    cli.main(["enrich", "--no-lyrics"])
    assert captured["client"] is None


def test_cli_enrich_client_creation_error(monkeypatch, capsys, tmp_path):
    from app import cli
    from app.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()

    def boom(settings):
        raise RuntimeError("navidrome caído")

    monkeypatch.setattr("app.subsonic.SubsonicClient", boom)

    async def fake_enrich(conn, **kwargs):
        return {"run_id": 1}

    monkeypatch.setattr("app.enrich.pipeline.enrich_library", fake_enrich)
    code = cli.main(["enrich"])
    out = capsys.readouterr().out
    assert code == 0
    assert "aviso" in out


def test_cmd_facets_rebuild(monkeypatch, capsys, tmp_path):
    from app import cli

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from app.config import get_settings

    get_settings.cache_clear()
    code = cli.main(["facets", "--rebuild"])
    out = capsys.readouterr().out
    assert code == 0
    assert "rebuilt" in out


def test_cmd_facets_summary(monkeypatch, capsys, tmp_path):
    from app import cli
    from app.db import facet_upsert, get_conn, init_db

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from app.config import get_settings

    get_settings.cache_clear()
    conn = get_conn()
    init_db(conn)
    facet_upsert(conn, "track", "t1", {"languages": ["es"]})
    conn.commit()
    code = cli.main(["facets", "--entity-type", "track"])
    out = capsys.readouterr().out
    assert code == 0
    assert '"es"' in out
    assert '"languages"' in out
