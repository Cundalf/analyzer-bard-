from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import FakeSubsonic

from app.config import Settings
from app.db import connect, init_db
from app.janitor import report as report_mod
from app.janitor.report import TagDiff
from app.janitor.runner import (
    JanitorError,
    JanitorResult,
    default_beets_config,
    ensure_beets_config,
    rescan_navidrome,
    run_beets,
    run_janitor,
)


def make_settings(tmp_path: Path, **kwargs) -> Settings:
    music = tmp_path / "music"
    music.mkdir(exist_ok=True)
    beetsdir = tmp_path / "config"
    beetsdir.mkdir(exist_ok=True)
    defaults = dict(
        data_dir=str(tmp_path / "data"),
        music_dir=str(music),
        beetsdir=str(beetsdir),
        janitor_enabled=True,
    )
    defaults.update(kwargs)
    s = Settings(**defaults)
    s.ensure_dirs()
    return s


# ------------------------------------------------------------ JanitorResult


def test_result_as_dict_empty():
    data = JanitorResult().as_dict()
    assert data["status"] == "ok"
    assert data["diffs"] == []
    assert data["wav"] == {}


def test_result_as_dict_with_diff():
    result = JanitorResult(diffs=[TagDiff(file="/a", old_tags={}, new_tags={})])
    data = result.as_dict()
    assert data["diffs"][0]["file"] == "/a"


# ------------------------------------------------------------ beets config


def test_default_beets_config_no_move():
    config = default_beets_config()
    assert config["import"]["move"] is False
    assert config["import"]["copy"] is False
    assert "bardo" in config["plugins"]
    assert config["pluginpath"]


def test_ensure_beets_config_creates(tmp_path):
    s = make_settings(tmp_path, acoustid_key="KEY")
    target = ensure_beets_config(s)
    assert (
        target == s.beetsdir_path
        if hasattr(s, "beetsdir_path")
        else Path(s.beetsdir) / "config.yaml"
    )
    assert target.exists()
    import yaml

    data = yaml.safe_load(target.read_text())
    assert data["acoustid"]["apikey"] == "KEY"
    assert data["subsonic"]["url"] == s.subsonic_url
    assert data["directory"] == str(tmp_path / "music")


def test_ensure_beets_config_does_not_overwrite(tmp_path):
    s = make_settings(tmp_path)
    target = Path(s.beetsdir) / "config.yaml"
    target.write_text("custom: true\n")
    assert ensure_beets_config(s) == target
    assert target.read_text() == "custom: true\n"


def test_ensure_beets_config_explicit_path(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text("x: 1")
    s = make_settings(tmp_path, beets_config=str(custom))
    assert ensure_beets_config(s) == custom.resolve()


def test_ensure_beets_config_missing_dir(tmp_path):
    s = make_settings(tmp_path, beetsdir=str(tmp_path / "noexiste"))
    assert ensure_beets_config(s) is None


# ------------------------------------------------------------ run_beets


def test_run_beets_no_binary(tmp_path, monkeypatch):
    s = make_settings(tmp_path, beets_bin="")
    monkeypatch.setattr("app.janitor.runner.shutil.which", lambda n: None)
    with pytest.raises(JanitorError):
        run_beets(tmp_path / "music", settings=s)


def test_run_beets_import_command(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        captured["cwd"] = kwargs.get("cwd")
        import subprocess

        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("app.janitor.runner.subprocess.run", fake_run)
    s = make_settings(tmp_path, beets_bin="/usr/bin/beet")
    out = run_beets(
        s.music_dir, settings=s, pretend=True, timid=True, logpath=Path("/tmp/log.jsonl")
    )
    assert out["returncode"] == 0
    assert out["cmd"] == ["/usr/bin/beet", "import", "-p", "-t", s.music_dir]
    assert captured["env"]["BEETSDIR"] == s.beetsdir
    assert captured["env"]["BARDO_BEETS_LOG"] == "/tmp/log.jsonl"
    assert captured["cwd"] == s.beetsdir


def test_run_beets_config_mode(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        import subprocess

        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr("app.janitor.runner.subprocess.run", fake_run)
    s = make_settings(tmp_path, beets_bin="/usr/bin/beet")
    run_beets(s.music_dir, settings=s, mode="config")
    assert captured["cmd"] == ["/usr/bin/beet", "config", "-p"]


def test_run_beets_extra_args_and_truncation(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        import subprocess

        return subprocess.CompletedProcess(cmd, 0, stdout="x" * 30000, stderr="y" * 30000)

    monkeypatch.setattr("app.janitor.runner.subprocess.run", fake_run)
    s = make_settings(tmp_path, beets_bin="/usr/bin/beet")
    out = run_beets(s.music_dir, settings=s, extra_args=["-q"])
    assert len(out["stdout"]) == 20000
    assert len(out["stderr"]) == 20000


def test_run_beets_no_cwd_when_no_beetsdir(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        import subprocess

        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("app.janitor.runner.subprocess.run", fake_run)
    s = make_settings(tmp_path, beetsdir="", beets_bin="/usr/bin/beet")
    run_beets(tmp_path, settings=s)
    assert captured["cwd"] is None


# ------------------------------------------------------------ rescan


def test_rescan_with_injected_client(tmp_path):
    s = make_settings(tmp_path)
    result = rescan_navidrome(s, client=FakeSubsonic())
    assert result == {"ok": True, "status": {"scanning": True}}


def test_rescan_failure(tmp_path):
    s = make_settings(tmp_path)
    result = rescan_navidrome(s, client=FakeSubsonic(fail=True))
    assert result["ok"] is False
    assert "scan failed" in result["error"]


def test_rescan_creates_and_closes_client(tmp_path, monkeypatch):
    s = make_settings(tmp_path)
    closed = {"v": False}

    class C(FakeSubsonic):
        def close(self):
            closed["v"] = True

    monkeypatch.setattr("app.subsonic.SubsonicClient", lambda st: C())
    result = rescan_navidrome(s)
    assert result["ok"] is True
    assert closed["v"] is True


# ------------------------------------------------------------ run_janitor


def test_run_janitor_disabled(tmp_path):
    s = make_settings(tmp_path, janitor_enabled=False)
    with pytest.raises(JanitorError):
        run_janitor(settings=s, pretend=False, do_rescan=False)


def test_run_janitor_pretend_when_disabled(tmp_path, monkeypatch):
    s = make_settings(tmp_path, janitor_enabled=False)
    monkeypatch.setattr(
        "app.janitor.runner.run_beets",
        lambda *a, **k: {"cmd": ["beet"], "returncode": 0, "stdout": "", "stderr": ""},
    )
    result = run_janitor(settings=s, pretend=True, do_rescan=False)
    assert result.status == "ok"


def test_run_janitor_without_music_dir(tmp_path):
    s = make_settings(tmp_path, music_dir="")
    s.music_dir = ""
    with pytest.raises(JanitorError):
        run_janitor(settings=s, pretend=True, do_rescan=False)


def test_run_janitor_full_flow(tmp_path, monkeypatch):
    s = make_settings(tmp_path)
    (Path(s.music_dir) / "a.wav").write_bytes(b"RIFF")

    def fake_run(cmd, **kwargs):
        import subprocess

        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("app.janitor.runner.subprocess.run", fake_run)
    monkeypatch.setattr(
        "app.janitor.runner.run_beets",
        lambda *a, **k: {
            "cmd": ["beet"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
    )
    conn = connect(tmp_path / "data" / "bardo.db")
    init_db(conn)
    events = []
    result = run_janitor(
        settings=s,
        conn=conn,
        pretend=True,
        do_rescan=False,
        progress=lambda stage, payload: events.append(stage),
    )
    assert result.status == "ok"
    assert result.wav["count"] == 1
    assert result.wav["converted"] == 0
    assert [e for e in events if e in ("start", "wav", "beets", "done")]
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (result.run_id,)).fetchone()
    assert run["status"] == "ok"
    assert run["kind"] == "pretend"
    conn.close()


def test_run_janitor_skip_wav(tmp_path, monkeypatch):
    s = make_settings(tmp_path)
    monkeypatch.setattr(
        "app.janitor.runner.run_beets",
        lambda *a, **k: {
            "cmd": ["beet"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
    )
    result = run_janitor(settings=s, pretend=True, skip_wav=True, do_rescan=False)
    assert result.wav == {"skipped": True}


def test_run_janitor_beets_failure_marks_error(tmp_path, monkeypatch):
    s = make_settings(tmp_path)
    monkeypatch.setattr(
        "app.janitor.runner.run_beets",
        lambda *a, **k: {
            "cmd": ["beet"],
            "returncode": 1,
            "stdout": "",
            "stderr": "fatal: algo\nsegunda linea",
        },
    )
    conn = connect(tmp_path / "data" / "bardo.db")
    init_db(conn)
    result = run_janitor(settings=s, pretend=False, skip_wav=True, do_rescan=False, conn=conn)
    assert result.status == "error"
    assert "segunda linea" in result.errors[-1]
    run = conn.execute("SELECT status FROM runs WHERE id = ?", (result.run_id,)).fetchone()
    assert run["status"] == "error"
    conn.close()


def test_run_janitor_beets_failure_empty_stderr(tmp_path, monkeypatch):
    s = make_settings(tmp_path)
    monkeypatch.setattr(
        "app.janitor.runner.run_beets",
        lambda *a, **k: {
            "cmd": ["beet"],
            "returncode": 2,
            "stdout": "",
            "stderr": "",
        },
    )
    result = run_janitor(settings=s, pretend=False, skip_wav=True, do_rescan=False)
    assert result.status == "error"
    assert "rc=2" in result.errors[-1]


def test_run_janitor_beets_exception(tmp_path, monkeypatch):
    s = make_settings(tmp_path)

    def boom(*a, **k):
        raise JanitorError("beet no encontrado")

    monkeypatch.setattr("app.janitor.runner.run_beets", boom)
    result = run_janitor(settings=s, pretend=True, skip_wav=True, do_rescan=False)
    assert result.status == "error"
    assert "beet no encontrado" in result.errors[0]


def test_run_janitor_wav_scan_error(tmp_path, monkeypatch):
    s = make_settings(tmp_path)
    monkeypatch.setattr(
        "app.janitor.runner.scan_wavs",
        lambda music: (_ for _ in ()).throw(OSError("disco roto")),
    )
    monkeypatch.setattr(
        "app.janitor.runner.run_beets",
        lambda *a, **k: {
            "cmd": ["beet"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
    )
    result = run_janitor(settings=s, pretend=True, do_rescan=False)
    assert any("disco roto" in e for e in result.errors)


def test_run_janitor_reads_jsonl_and_ingests(tmp_path, monkeypatch):
    s = make_settings(tmp_path)

    def fake_beets(*args, **kwargs):
        logpath = kwargs["logpath"]
        logpath.parent.mkdir(parents=True, exist_ok=True)
        diff = TagDiff(
            file="/music/a.flac",
            old_tags={"artist": "?"},
            new_tags={"artist": "Wind Rose"},
            match_score=0.9,
            status="APPLY",
        )
        report_mod.write_jsonl(type("R", (), {"items": [diff]})(), logpath)
        return {"cmd": ["beet"], "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("app.janitor.runner.run_beets", fake_beets)
    conn = connect(tmp_path / "data" / "bardo.db")
    init_db(conn)
    result = run_janitor(settings=s, pretend=False, skip_wav=True, do_rescan=False, conn=conn)
    assert len(result.diffs) == 1
    rows = conn.execute("SELECT COUNT(*) AS n FROM import_log").fetchone()["n"]
    assert rows == 1
    stats = json.loads(
        conn.execute("SELECT stats FROM runs WHERE id = ?", (result.run_id,)).fetchone()["stats"]
    )
    assert stats["diffs"] == 1
    conn.close()


def test_run_janitor_rescan_success_and_failure(tmp_path, monkeypatch):
    s = make_settings(tmp_path)
    monkeypatch.setattr(
        "app.janitor.runner.run_beets",
        lambda *a, **k: {
            "cmd": ["beet"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
    )
    result = run_janitor(
        settings=s,
        pretend=False,
        skip_wav=True,
        do_rescan=True,
        client=FakeSubsonic(),
    )
    assert result.rescan["ok"] is True

    result2 = run_janitor(
        settings=s,
        pretend=False,
        skip_wav=True,
        do_rescan=True,
        client=FakeSubsonic(fail=True),
    )
    assert any("rescan" in e for e in result2.errors)


def test_run_janitor_pretend_skips_rescan(tmp_path, monkeypatch):
    s = make_settings(tmp_path)
    monkeypatch.setattr(
        "app.janitor.runner.run_beets",
        lambda *a, **k: {
            "cmd": ["beet"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
    )
    called = {"n": 0}

    def fake_rescan(*a, **k):
        called["n"] += 1
        return {"ok": True}

    monkeypatch.setattr("app.janitor.runner.rescan_navidrome", fake_rescan)
    run_janitor(settings=s, pretend=True, skip_wav=True, do_rescan=True)
    assert called["n"] == 0


def test_run_janitor_no_run_id_without_conn(tmp_path, monkeypatch):
    s = make_settings(tmp_path)
    monkeypatch.setattr(
        "app.janitor.runner.run_beets",
        lambda *a, **k: {
            "cmd": ["beet"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
    )
    result = run_janitor(settings=s, pretend=True, skip_wav=True, do_rescan=False)
    assert result.run_id is None
    assert result.beets["logpath"].endswith("beets_manual.jsonl")


def test_run_janitor_report_read_error(tmp_path, monkeypatch):
    s = make_settings(tmp_path)

    def fake_beets(*args, **kwargs):
        kwargs["logpath"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["logpath"].write_text("{linea corrupta}\n")
        return {"cmd": ["beet"], "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("app.janitor.runner.run_beets", fake_beets)
    monkeypatch.setattr(
        "app.janitor.runner.report_mod.read_jsonl",
        lambda p: (_ for _ in ()).throw(ValueError("jsonl roto")),
    )
    result = run_janitor(settings=s, pretend=False, skip_wav=True, do_rescan=False)
    assert any("jsonl roto" in e for e in result.errors)


def test_bool_opt_helper(tmp_path):
    from app.janitor.runner import _bool_opt

    config = {"a": {"b": True}}
    assert _bool_opt(config, "a.b", False) is True
    assert _bool_opt(config, "a.c", "def") == "def"
    assert _bool_opt(config, "x.y.z", 1) == 1
    assert _bool_opt({}, "a", 2) == 2


def test_run_janitor_converts_wav_with_settings_bin(tmp_path, monkeypatch):
    from app.db import connect, init_db
    from app.janitor.runner import run_janitor

    music = tmp_path / "music"
    music.mkdir()
    (music / "a.wav").write_bytes(b"RIFF")
    beetsdir = tmp_path / "config"
    beetsdir.mkdir()
    script = tmp_path / "myffmpeg"
    script.write_text(
        '#!/bin/sh\nout=\'\'\nfor a in "$@"; do out="$a"; done\nprintf \'fLaC\' > "$out"\n'
    )
    script.chmod(0o755)
    s = Settings(
        data_dir=str(tmp_path / "data"),
        music_dir=str(music),
        beetsdir=str(beetsdir),
        janitor_enabled=True,
        ffmpeg_bin=str(script),
    )
    s.ensure_dirs()
    monkeypatch.setattr(
        "app.janitor.runner.run_beets",
        lambda *a, **k: {
            "cmd": ["beet"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
    )
    conn = connect(tmp_path / "data" / "bardo.db")
    init_db(conn)
    result = run_janitor(settings=s, conn=conn, do_rescan=False)
    assert result.wav["converted"] == 1
    conn.close()


def test_run_janitor_rescan_progress(tmp_path, monkeypatch):
    from app.janitor.runner import run_janitor

    music = tmp_path / "music"
    music.mkdir()
    beetsdir = tmp_path / "config"
    beetsdir.mkdir()
    s = Settings(
        data_dir=str(tmp_path / "data"),
        music_dir=str(music),
        beetsdir=str(beetsdir),
        janitor_enabled=True,
    )
    s.ensure_dirs()
    monkeypatch.setattr(
        "app.janitor.runner.run_beets",
        lambda *a, **k: {
            "cmd": ["beet"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
    )
    events = []
    run_janitor(
        settings=s,
        client=FakeSubsonic(),
        do_rescan=True,
        progress=lambda stage, payload: events.append(stage),
    )
    assert "rescan" in events


def test_run_janitor_no_ingest_without_run_id(tmp_path, monkeypatch):
    from app.janitor.runner import run_janitor

    music = tmp_path / "music"
    music.mkdir()
    beetsdir = tmp_path / "config"
    beetsdir.mkdir()
    s = Settings(
        data_dir=str(tmp_path / "data"),
        music_dir=str(music),
        beetsdir=str(beetsdir),
        janitor_enabled=True,
    )
    s.ensure_dirs()
    from app.janitor.report import TagDiff, write_jsonl

    def fake_beets(*args, **kwargs):
        logpath = kwargs["logpath"]
        logpath.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(
            type("R", (), {"items": [TagDiff(file="/a", old_tags={}, new_tags={})]})(),
            logpath,
        )
        return {"cmd": ["beet"], "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("app.janitor.runner.run_beets", fake_beets)
    result = run_janitor(settings=s, conn=None, pretend=False, skip_wav=True, do_rescan=False)
    assert len(result.diffs) == 1
    assert result.run_id is None
