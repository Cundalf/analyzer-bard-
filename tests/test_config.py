from __future__ import annotations

import pytest

from app.config import (
    BOOL_FIELDS,
    INT_FIELDS,
    OVERRIDABLE,
    Settings,
    coerce_override,
    get_settings,
    runtime_settings,
)


def test_paths_expand_user(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    s = Settings(data_dir="~/musicdata")
    assert s.data_path == (tmp_path / "musicdata").resolve()
    assert s.db_path == s.data_path / "bardo.db"
    assert s.runs_path == s.data_path / "runs"


def test_wav_backup_path_default_and_custom(tmp_path):
    s = Settings(data_dir=str(tmp_path / "data"))
    assert s.wav_backup_path == s.data_path / "wav_originals"
    custom = Settings(data_dir=str(tmp_path / "data"), wav_backup_dir=str(tmp_path / "bk"))
    assert custom.wav_backup_path == (tmp_path / "bk").resolve()


def test_ensure_dirs_creates_both(tmp_path):
    s = Settings(data_dir=str(tmp_path / "data"))
    s.ensure_dirs()
    assert s.data_path.is_dir()
    assert s.runs_path.is_dir()


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("SUBSONIC_URL", "http://nav.example:4533")
    monkeypatch.setenv("JANITOR_ENABLED", "true")
    monkeypatch.setenv("EMBED_DIM", "1024")
    monkeypatch.setenv("PLAYLIST_DEFAULT_SIZE", "12")
    s = Settings()
    assert s.subsonic_url == "http://nav.example:4533"
    assert s.janitor_enabled is True
    assert s.embed_dim == 1024
    assert s.playlist_default_size == 12


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("", False),
        ("basura", False),
        (1, True),
        (0, False),
    ],
)
def test_coerce_bool_field(value, expected):
    assert coerce_override("enable_lastfm", value) is expected


@pytest.mark.parametrize("field", sorted(INT_FIELDS))
def test_coerce_int_field(field):
    assert coerce_override(field, "42") == 42
    assert coerce_override(field, 7) == 7


def test_coerce_int_invalid_raises():
    with pytest.raises(ValueError):
        coerce_override("embed_dim", "no-numero")


def test_coerce_string_passthrough():
    assert coerce_override("subsonic_url", "http://x") == "http://x"
    assert coerce_override("ollama_chat_model", 123) == 123


def test_runtime_settings_without_conn():
    base = get_settings()
    assert runtime_settings(None) is base


def test_runtime_settings_applies_and_ignores(conn):
    from app.db import set_setting

    set_setting(conn, "subsonic_url", "http://override:4533")
    set_setting(conn, "janitor_enabled", "true")
    set_setting(conn, "embed_dim", "512")
    set_setting(conn, "clave_no_soportada", "x")
    set_setting(conn, "subsonic_user", "")
    s = runtime_settings(conn)
    assert s.subsonic_url == "http://override:4533"
    assert s.janitor_enabled is True
    assert s.embed_dim == 512
    assert s.subsonic_user == get_settings().subsonic_user
    assert not hasattr(s, "clave_no_soportada")


def test_runtime_settings_no_overrides_returns_base(conn):
    assert runtime_settings(conn) is get_settings()


def test_overridable_fields_exist():
    base = Settings()
    for field in OVERRIDABLE:
        assert hasattr(base, field), field
    for field in BOOL_FIELDS | INT_FIELDS:
        assert field in OVERRIDABLE, field
