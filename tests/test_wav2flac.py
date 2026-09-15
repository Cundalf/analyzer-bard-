from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.config import Settings
from app.janitor.wav2flac import (
    WavFinding,
    WavScanResult,
    convert_all,
    convert_wav_to_flac,
    ffmpeg_available,
    find_ffmpeg,
    scan_wavs,
)


# ------------------------------------------------------------ scan

def test_scan_finds_wav_and_wave(tmp_path: Path):
    (tmp_path / "a.wav").write_bytes(b"RIFF")
    (tmp_path / "b.WAVE").write_bytes(b"RIFF12")
    (tmp_path / "c.flac").write_bytes(b"fLaC")
    (tmp_path / "d.txt").write_text("x")
    result = scan_wavs(tmp_path)
    assert result.count == 2
    assert result.total_size == 10
    assert {f.path.name for f in result.findings} == {"a.wav", "b.WAVE"}


def test_scan_recursive_and_sorted(tmp_path: Path):
    (tmp_path / "z.wav").write_bytes(b"1")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "a.wav").write_bytes(b"22")
    result = scan_wavs(tmp_path)
    names = [f.path.name for f in result.findings]
    assert names == ["a.wav", "z.wav"]


def test_scan_empty_dir(tmp_path: Path):
    result = scan_wavs(tmp_path)
    assert result.count == 0
    assert result.total_size == 0
    assert result.as_dict()["findings"] == []


def test_scan_missing_dir():
    with pytest.raises(FileNotFoundError):
        scan_wavs("/ruta/inexistente/xyz")


def test_scan_as_dict(tmp_path: Path):
    (tmp_path / "a.wav").write_bytes(b"RIFF")
    data = scan_wavs(tmp_path).as_dict()
    assert data["count"] == 1
    assert data["music_dir"] == str(tmp_path.resolve())
    assert data["findings"][0]["converted"] is False


def test_finding_as_dict_without_flac():
    finding = WavFinding(path=Path("/m/a.wav"), size=10)
    assert finding.as_dict()["flac_path"] is None


def test_wav_scan_result_defaults():
    result = WavScanResult(music_dir=Path("/m"))
    assert result.count == 0 and result.total_size == 0


# ------------------------------------------------------------ ffmpeg detection

def test_find_ffmpeg_explicit(tmp_path: Path):
    fake = tmp_path / "myffmpeg"
    fake.write_text("#!/bin/sh\n")
    assert find_ffmpeg(str(fake)) == str(fake)


def test_find_ffmpeg_by_name_in_path(tmp_path: Path, monkeypatch):
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_ffmpeg() == str(fake)


def test_find_ffmpeg_none(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(
        "app.janitor.wav2flac.shutil.which", lambda name: None
    )

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "imageio_ffmpeg":
            raise ImportError("no imageio")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert find_ffmpeg() is None
    assert ffmpeg_available() is False


def test_find_ffmpeg_prefers_explicit(monkeypatch, tmp_path):
    explicit = tmp_path / "e"
    explicit.write_text("x")
    monkeypatch.setattr("app.janitor.wav2flac.shutil.which", lambda n: "/usr/bin/ffmpeg")
    assert find_ffmpeg(str(explicit)) == str(explicit)


def test_find_ffmpeg_unknown_name_falls_back(monkeypatch):
    monkeypatch.setattr("app.janitor.wav2flac.shutil.which", lambda n: None)
    monkeypatch.setattr(
        "app.janitor.wav2flac.shutil.which", lambda n: None
    )
    import sys
    import types

    fake = types.ModuleType("imageio_ffmpeg")
    fake.get_ffmpeg_exe = lambda: "/fake/ffmpeg"
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake)
    assert find_ffmpeg("no-existe") == "/fake/ffmpeg"


# ------------------------------------------------------------ convert

def fake_ffmpeg(tmp_path: Path, fail: bool = False, empty_output: bool = False):
    script = tmp_path / "ffmpeg-fake"
    behavior = "exit 1" if fail else ":"
    content = f"""#!/bin/sh
out=""
for arg in "$@"; do out="$arg"; done
{build_output(empty_output)}
{behavior}
"""
    script.write_text(content)
    script.chmod(0o755)
    return str(script)


def build_output(empty: bool) -> str:
    if empty:
        return ': > "$out"'
    return 'printf "fLaC" > "$out"'


def test_convert_missing_file(tmp_path: Path):
    finding = convert_wav_to_flac(tmp_path / "nope.wav", ffmpeg_bin="/bin/true")
    assert finding.error == "file not found"
    assert finding.size == 0


def test_convert_no_ffmpeg(tmp_path: Path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setattr("app.janitor.wav2flac.find_ffmpeg", lambda b=None: None)
    finding = convert_wav_to_flac(wav)
    assert finding.error == "ffmpeg not installed"
    assert finding.converted is False


def test_convert_skips_existing_flac(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    wav.with_suffix(".flac").write_bytes(b"fLaC")
    finding = convert_wav_to_flac(wav, ffmpeg_bin="/bin/true")
    assert finding.error == "flac already exists (skipped)"
    assert finding.flac_path == wav.with_suffix(".flac")


def test_convert_success_with_backup(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFDATA")
    binary = fake_ffmpeg(tmp_path)
    backup = tmp_path / "bk"
    finding = convert_wav_to_flac(
        wav, backup_dir=backup, ffmpeg_bin=binary
    )
    assert finding.converted is True
    assert finding.backed_up is True
    assert finding.deleted is False
    assert wav.exists()
    assert (backup / "a.wav").read_bytes() == b"RIFFDATA"


def test_convert_success_delete_original(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    binary = fake_ffmpeg(tmp_path)
    finding = convert_wav_to_flac(wav, delete_original=True, ffmpeg_bin=binary)
    assert finding.converted is True
    assert finding.deleted is True
    assert not wav.exists()
    assert wav.with_suffix(".flac").exists()


def test_convert_backup_name_collision(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"DATA")
    backup = tmp_path / "bk"
    backup.mkdir()
    (backup / "a.wav").write_bytes(b"OLD")
    binary = fake_ffmpeg(tmp_path)
    finding = convert_wav_to_flac(wav, backup_dir=backup, ffmpeg_bin=binary)
    assert finding.backed_up is True
    files = sorted(p.name for p in backup.iterdir())
    assert len(files) == 2
    assert "a.wav" in files
    other = next(f for f in files if f != "a.wav")
    assert other.startswith("a.")
    assert other.endswith(".wav")


def test_convert_overwrite_existing_flac(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    flac = wav.with_suffix(".flac")
    flac.write_bytes(b"OLD")
    binary = fake_ffmpeg(tmp_path)
    finding = convert_wav_to_flac(wav, overwrite=True, ffmpeg_bin=binary)
    assert finding.converted is True


def test_convert_ffmpeg_failure(tmp_path: Path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error raro\n")

    monkeypatch.setattr("app.janitor.wav2flac.subprocess.run", fake_run)
    finding = convert_wav_to_flac(wav, ffmpeg_bin="/bin/true")
    assert finding.converted is False
    assert finding.error == "error raro"


def test_convert_ffmpeg_failure_empty_stderr(tmp_path: Path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="  ")

    monkeypatch.setattr("app.janitor.wav2flac.subprocess.run", fake_run)
    finding = convert_wav_to_flac(wav, ffmpeg_bin="/bin/true")
    assert finding.error == "ffmpeg failed"


def test_convert_failure_removes_empty_flac(tmp_path: Path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    flac = wav.with_suffix(".flac")

    def fake_run(cmd, **kwargs):
        flac.write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom\n")

    monkeypatch.setattr("app.janitor.wav2flac.subprocess.run", fake_run)
    convert_wav_to_flac(wav, ffmpeg_bin="/bin/true")
    assert not flac.exists()


def test_convert_failure_keeps_nonempty_flac(tmp_path: Path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    flac = wav.with_suffix(".flac")

    def fake_run(cmd, **kwargs):
        flac.write_bytes(b"partial")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom\n")

    monkeypatch.setattr("app.janitor.wav2flac.subprocess.run", fake_run)
    convert_wav_to_flac(wav, ffmpeg_bin="/bin/true")
    assert flac.exists()


# ------------------------------------------------------------ convert_all

def test_convert_all_uses_settings_backup(tmp_path: Path):
    music = tmp_path / "music"
    music.mkdir()
    (music / "a.wav").write_bytes(b"RIFF")
    binary = fake_ffmpeg(tmp_path)
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        music_dir=str(music),
        ffmpeg_bin=binary,
    )
    scan = scan_wavs(music)
    events = []
    convert_all(scan, settings=settings, progress=lambda i, t, f: events.append((i, t)))
    assert scan.findings[0].converted
    assert events == [(1, 1)]
    assert settings.wav_backup_path.exists()


def test_convert_all_delete_original_from_settings(tmp_path: Path):
    music = tmp_path / "music"
    music.mkdir()
    wav = music / "a.wav"
    wav.write_bytes(b"RIFF")
    binary = fake_ffmpeg(tmp_path)
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        music_dir=str(music),
        wav_delete_originals=True,
        ffmpeg_bin=binary,
    )
    scan = scan_wavs(music)
    convert_all(scan, settings=settings)
    assert not wav.exists()
    assert scan.findings[0].deleted


def test_convert_all_empty(tmp_path: Path):
    scan = WavScanResult(music_dir=tmp_path)
    result = convert_all(scan, settings=Settings(data_dir=str(tmp_path / "d")))
    assert result.count == 0
