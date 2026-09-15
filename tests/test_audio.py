from __future__ import annotations

import io
import math
import struct
import wave
from pathlib import Path

import pytest

from app.enrich import audio as audio_mod
from app.enrich.audio import (
    AudioFeatures,
    analyze_file,
    apply_audio_features,
    librosa_available,
    resolve_track_path,
)
from tests.conftest import seed_library


def write_wav(path: Path, seconds: float = 1.0, freq: float = 440.0, amp: float = 0.8):
    rate = audio_mod.SAMPLE_RATE
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        data = bytearray()
        for i in range(frames):
            value = int(amp * 32767 * math.sin(2 * math.pi * freq * i / rate))
            data.extend(struct.pack("<h", value))
        wav.writeframes(bytes(data))
    return path


# ---------------------------------------------------------------- rutas

def test_resolve_track_path_absolute_inside(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    track = music / "a.flac"
    track.write_bytes(b"x")
    assert resolve_track_path(str(track), str(music)) == track.resolve()


def test_resolve_track_path_relative(tmp_path):
    music = tmp_path / "music"
    (music / "sub").mkdir(parents=True)
    track = music / "sub" / "a.flac"
    track.write_bytes(b"x")
    assert resolve_track_path("sub/a.flac", str(music)) == track.resolve()


def test_resolve_track_path_leading_slash(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    track = music / "a.flac"
    track.write_bytes(b"x")
    assert resolve_track_path("/a.flac", str(music)) == track.resolve()


def test_resolve_track_path_escapes_root(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    outside = tmp_path / "outside.flac"
    outside.write_bytes(b"x")
    assert resolve_track_path("../outside.flac", str(music)) is None


def test_resolve_track_path_missing_inputs(tmp_path):
    assert resolve_track_path(None, str(tmp_path)) is None
    assert resolve_track_path("a.flac", None) is None
    assert resolve_track_path("", "") is None


def test_resolve_track_path_not_a_file(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    assert resolve_track_path("nope.flac", str(music)) is None


def test_resolve_track_path_directory(tmp_path):
    music = tmp_path / "music"
    (music / "dir").mkdir(parents=True)
    assert resolve_track_path("dir", str(music)) is None


# ---------------------------------------------------------------- análisis

def test_analyze_missing_file(tmp_path):
    assert analyze_file(tmp_path / "nope.flac") is None


def test_analyze_success(tmp_path):
    wav = write_wav(tmp_path / "tone.wav")
    features = analyze_file(wav)
    assert features is not None
    assert 0.5 < features.energy <= 1.0
    assert features.brightness >= 0
    assert features.analyzed_seconds > 0


def test_analyze_silence_energy_zero(tmp_path):
    wav = write_wav(tmp_path / "silence.wav", amp=0.0)
    features = analyze_file(wav)
    assert features is not None
    assert features.energy == 0.0


def test_analyze_low_amplitude(tmp_path):
    wav = write_wav(tmp_path / "quiet.wav", amp=0.001)
    features = analyze_file(wav)
    assert features is not None
    assert features.energy < 0.5


def test_analyze_no_ffmpeg(tmp_path, monkeypatch):
    wav = write_wav(tmp_path / "tone.wav")
    monkeypatch.setattr("app.enrich.audio._find_ffmpeg", lambda b: None)
    assert analyze_file(wav) is None


def test_analyze_decode_failure(tmp_path, monkeypatch):
    import subprocess

    wav = write_wav(tmp_path / "tone.wav")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"boom")

    monkeypatch.setattr("app.enrich.audio._find_ffmpeg", lambda b: "/bin/false")
    monkeypatch.setattr("app.enrich.audio._decode_pcm", lambda *a, **k: None)
    features = analyze_file(wav)
    assert features is None


def test_decode_pcm_timeout(tmp_path, monkeypatch):
    import subprocess as sp

    wav = write_wav(tmp_path / "tone.wav")
    monkeypatch.setattr("app.janitor.wav2flac.find_ffmpeg", lambda b=None: "/bin/ffmpeg")

    def fake_run(cmd, **kwargs):
        raise sp.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(sp, "run", fake_run)
    assert audio_mod._decode_pcm(wav, ffmpeg_bin=None, seconds=5) is None


def test_samples_from_invalid_wav():
    assert audio_mod._samples_from_wav(b"no es wav") == []
    assert audio_mod._samples_from_wav(b"") == []


def test_samples_stereo_downmix():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(struct.pack("<hhhh", 100, -100, 200, -200))
    samples = audio_mod._samples_from_wav(buffer.getvalue())
    assert samples == [100, 200]


def test_rms_empty():
    assert audio_mod._rms([]) == 0.0


def test_zero_crossing_rate_short():
    assert audio_mod._zero_crossing_rate([1]) == 0.0
    assert audio_mod._zero_crossing_rate([]) == 0.0


def test_normalize_energy_bounds():
    assert audio_mod._normalize_energy(0.0) == 0.0
    assert audio_mod._normalize_energy(40000) == 1.0
    assert 0.0 <= audio_mod._normalize_energy(100) <= 1.0


def test_dynamic_range_short_signal():
    assert audio_mod._dynamic_range([1, 2, 3]) == 0.0


def test_dynamic_range_with_variation(tmp_path):
    rate = audio_mod.SAMPLE_RATE
    with wave.open(str(tmp_path / "dyn.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        data = bytearray()
        for i in range(rate):
            amp = 0.8 if i < rate // 2 else 0.05
            value = int(amp * 32767 * math.sin(2 * math.pi * 440 * i / rate))
            data.extend(struct.pack("<h", value))
        wav.writeframes(bytes(data))
    features = analyze_file(tmp_path / "dyn.wav")
    assert features is not None
    assert features.dynamic_range > 0.1


# ---------------------------------------------------------------- facets

def test_features_as_facets():
    features = AudioFeatures(
        energy=0.8, brightness=0.3, dynamic_range=0.4, bpm=120.0, key="Am"
    )
    facets = features.as_facets()
    assert facets["energy"] == 0.8
    assert facets["energy_source"] == "audio"
    assert facets["bpm"] == 120.0
    assert facets["key"] == "Am"


def test_features_as_facets_without_rhythm():
    features = AudioFeatures(energy=0.5, brightness=0.1, dynamic_range=0.2)
    facets = features.as_facets()
    assert "bpm" not in facets
    assert "key" not in facets


def test_apply_audio_features_overrides_llm_energy():
    facets = {"energy": 0.1, "moods": ["tranquilo"]}
    features = AudioFeatures(energy=0.95, brightness=0.5, dynamic_range=0.6)
    merged = apply_audio_features(facets, features)
    assert merged["energy"] == 0.95
    assert merged["energy_source"] == "audio"
    assert merged["moods"] == ["tranquilo"]


def test_apply_audio_features_no_energy_key():
    features = AudioFeatures(energy=0.0, brightness=0.0, dynamic_range=0.0)
    merged = apply_audio_features({}, features)
    assert merged["energy"] == 0.0


def test_librosa_available_false(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "librosa":
            raise ImportError("no librosa")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert librosa_available() is False


def test_analyze_rhythm_without_librosa(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"librosa", "numpy"}:
            raise ImportError("no librosa")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    wav = write_wav(tmp_path / "tone.wav")
    features = analyze_file(wav, with_rhythm=True)
    assert features is not None
    assert features.bpm is None


def test_analyze_rhythm_handles_error(tmp_path, monkeypatch):
    import types

    fake = types.ModuleType("librosa")

    def boom(**kwargs):
        raise RuntimeError("librosa rota")

    fake.beat = types.SimpleNamespace(beat_track=boom)
    monkeypatch.setitem(__import__("sys").modules, "librosa", fake)
    result = audio_mod._analyze_rhythm(_minimal_wav())
    assert result == (None, None)


def _minimal_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(struct.pack("<hhhh", 0, 1000, -1000, 0) * 10)
    return buffer.getvalue()


# ---------------------------------------------------------------- pipeline

class NoLastFm:
    def fetch_for(self, *a, **k):
        return []

    def close(self):
        pass


@pytest.mark.anyio
async def test_pipeline_audio_analysis(conn, settings, tmp_path):
    from app.enrich.artist import Ficha, get_ficha, save_ficha
    from app.enrich.pipeline import enrich_library
    from tests.conftest import FakeOllama

    seed_library(conn)
    music = tmp_path / "music"
    music.mkdir()
    wav = write_wav(music / "t1.flac")
    conn.execute("UPDATE tracks SET path = 't1.flac' WHERE navidrome_id = 't1'")
    conn.commit()
    save_ficha(conn, Ficha("track", "t1", {"moods": []}, "d", 0.9, "inherited", "h"))
    settings.analyze_audio = True
    settings.music_dir = str(music)
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=NoLastFm(),
        artists=False,
        albums=False,
        tracks=True,
        force=True,
    )
    assert result["audio"] >= 1
    ficha = get_ficha(conn, "track", "t1")
    assert ficha["facets"]["energy_source"] == "audio"
    assert wav.exists()


@pytest.mark.anyio
async def test_pipeline_audio_disabled(conn, settings, tmp_path):
    from app.enrich.artist import Ficha, save_ficha
    from app.enrich.pipeline import enrich_library
    from tests.conftest import FakeOllama

    seed_library(conn)
    save_ficha(conn, Ficha("track", "t1", {"moods": []}, "d", 0.9, "inherited", "h"))
    settings.analyze_audio = False
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=NoLastFm(),
        artists=False,
        albums=False,
        tracks=True,
        force=True,
    )
    assert result["audio"] == 0


@pytest.mark.anyio
async def test_pipeline_audio_without_music_dir(conn, settings):
    from app.enrich.artist import Ficha, save_ficha
    from app.enrich.pipeline import enrich_library
    from tests.conftest import FakeOllama

    seed_library(conn)
    save_ficha(conn, Ficha("track", "t1", {"moods": []}, "d", 0.9, "inherited", "h"))
    settings.analyze_audio = True
    settings.music_dir = ""
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=NoLastFm(),
        artists=False,
        albums=False,
        tracks=True,
        force=True,
    )
    assert result["audio"] == 0


@pytest.mark.anyio
async def test_pipeline_audio_missing_path(conn, settings, tmp_path):
    from app.enrich.artist import Ficha, save_ficha
    from app.enrich.pipeline import enrich_library
    from tests.conftest import FakeOllama

    seed_library(conn)
    save_ficha(conn, Ficha("track", "t1", {"moods": []}, "d", 0.9, "inherited", "h"))
    settings.analyze_audio = True
    settings.music_dir = str(tmp_path / "vacio")
    (tmp_path / "vacio").mkdir()
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=NoLastFm(),
        artists=False,
        albums=False,
        tracks=True,
        force=True,
    )
    assert result["audio"] == 0


@pytest.mark.anyio
async def test_pipeline_audio_no_ficha(conn, settings, tmp_path):
    from app.enrich.pipeline import enrich_library
    from tests.conftest import FakeOllama

    seed_library(conn)
    music = tmp_path / "music"
    music.mkdir()
    write_wav(music / "t1.flac")
    conn.execute("UPDATE tracks SET path = 't1.flac' WHERE navidrome_id = 't1'")
    conn.commit()
    settings.analyze_audio = True
    settings.music_dir = str(music)
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=NoLastFm(),
        artists=False,
        albums=False,
        tracks=False,
        force=True,
    )
    assert result["audio"] == 0
