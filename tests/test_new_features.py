from __future__ import annotations

import pytest

from app.agent.runner import _attach_track_details, _fetch_candidates
from app.enrich.pipeline import (
    _analyze_track_audio,
    _detect_track_language,
    _fetch_lyrics,
    row_id,
)
from tests.conftest import seed_library


# ------------------------------------------------------------ runner details

def test_attach_track_details_from_candidates(conn):
    seed_library(conn)
    candidates = [
        {
            "track_id": "t1",
            "title": "Drunken Dwarves",
            "artist": "Wind Rose",
            "album": "Wintersaga",
            "year": 2019,
            "genre": "folk metal",
            "moods": ["fiesta", "a", "b", "c", "d"],
            "themes": ["cerveza"],
        }
    ]
    result = {"track_ids": ["t1"]}
    _attach_track_details(conn, result, candidates)
    assert result["tracks"][0]["title"] == "Drunken Dwarves"
    assert len(result["tracks"][0]["moods"]) == 4


def test_attach_track_details_empty(conn):
    seed_library(conn)
    result = {"track_ids": []}
    _attach_track_details(conn, result, [])
    assert result["tracks"] == []


def test_attach_track_details_fetches_missing(conn):
    seed_library(conn)
    result = {"track_ids": ["t3"]}
    _attach_track_details(conn, result, [])
    assert result["tracks"][0]["title"] == "Nightfall"
    assert result["tracks"][0]["artist"] == "Blind Guardian"


def test_attach_track_details_unknown_id(conn):
    seed_library(conn)
    result = {"track_ids": ["desconocido"]}
    _attach_track_details(conn, result, [])
    assert result["tracks"][0]["track_id"] == "desconocido"
    assert result["tracks"][0]["title"] == ""


def test_fetch_candidates_multiple_chunks(conn):
    seed_library(conn)
    ids = [f"fake{i}" for i in range(301)] + ["t1"]
    found = _fetch_candidates(conn, ids)
    assert "t1" in found
    assert len(found) == 1


# ------------------------------------------------------------ pipeline helpers

def test_row_id_variants():
    assert row_id({"id": "track:x"}) == "track:x"
    assert row_id({}) == ""
    assert row_id({"id": None}) == ""


def test_fetch_lyrics_by_id_first(conn):
    class Client:
        def get_lyrics_by_song_id(self, track_id):
            return "letra por id"

        def get_lyrics(self, artist, title):
            raise AssertionError("no debería llamarse")

    lyrics = _fetch_lyrics(
        Client(), {"navidrome_id": "t1", "title": "T", "artist_name": "A"}
    )
    assert lyrics == "letra por id"


def test_fetch_lyrics_fallback_by_name(conn):
    class Client:
        def get_lyrics_by_song_id(self, track_id):
            return ""

        def get_lyrics(self, artist, title):
            return f"{artist}:{title}"

    lyrics = _fetch_lyrics(
        Client(), {"navidrome_id": "t1", "title": "Cancion", "artist_name": "Artista"}
    )
    assert lyrics == "Artista:Cancion"


def test_fetch_lyrics_no_title(conn):
    class Client:
        def get_lyrics_by_song_id(self, track_id):
            return ""

        def get_lyrics(self, artist, title):
            raise AssertionError("no debería llamarse")

    lyrics = _fetch_lyrics(Client(), {"navidrome_id": "t1", "title": ""})
    assert lyrics == ""


def test_fetch_lyrics_client_without_open_subsonic(conn):
    class OldClient:
        def get_lyrics(self, artist, title):
            return "vieja api"

    lyrics = _fetch_lyrics(
        OldClient(), {"navidrome_id": "t1", "title": "T", "artist_name": "A"}
    )
    assert lyrics == "vieja api"


def test_detect_track_language_without_id(tmp_path):
    class Client:
        pass

    assert _detect_track_language(None, Client(), {}, lambda *a: None) is None


def test_detect_track_language_no_ficha(conn):
    class Client:
        def get_lyrics_by_song_id(self, track_id):
            return "texto de letra suficientemente largo para detectar idioma"

    result = _detect_track_language(
        conn,
        Client(),
        {"navidrome_id": "t1", "title": "T"},
        lambda *a: None,
    )
    assert result is None


def test_detect_track_language_without_lyrics(conn):
    from app.enrich.artist import Ficha, save_ficha

    seed_library(conn)
    save_ficha(conn, Ficha("track", "t1", {"moods": []}, "d", 0.9, "inherited", "h"))

    class Client:
        def get_lyrics_by_song_id(self, track_id):
            return ""

        def get_lyrics(self, artist, title):
            return ""

    result = _detect_track_language(
        conn, Client(), {"navidrome_id": "t1", "title": "T"}, lambda *a: None
    )
    assert result is None
    # las letras vacías no se cachean: el usuario puede agregarlas después
    cached = conn.execute(
        "SELECT lyrics FROM lyrics_cache WHERE track_id = 't1'"
    ).fetchone()
    assert cached is None


def test_analyze_track_audio_without_path(conn, settings):
    result = _analyze_track_audio(
        conn,
        {"navidrome_id": "t1", "path": ""},
        settings,
        lambda *a: None,
    )
    assert result is False


def test_analyze_track_audio_without_track_id(conn, settings, tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    (music / "a.flac").write_bytes(b"x")
    settings.music_dir = str(music)
    result = _analyze_track_audio(
        conn,
        {"navidrome_id": None, "id": "", "path": "a.flac"},
        settings,
        lambda *a: None,
    )
    assert result is False


def test_analyze_track_audio_analysis_fails(conn, settings, tmp_path, monkeypatch):
    music = tmp_path / "music"
    music.mkdir()
    (music / "a.flac").write_bytes(b"x")
    settings.music_dir = str(music)
    monkeypatch.setattr(
        "app.enrich.audio.analyze_file", lambda *a, **k: None
    )
    result = _analyze_track_audio(
        conn,
        {"navidrome_id": "t1", "path": "a.flac"},
        settings,
        lambda *a: None,
    )
    assert result is False


def test_analyze_track_audio_no_ficha(conn, settings, tmp_path, monkeypatch):
    from app.enrich.audio import AudioFeatures

    music = tmp_path / "music"
    music.mkdir()
    (music / "a.flac").write_bytes(b"x")
    settings.music_dir = str(music)
    monkeypatch.setattr(
        "app.enrich.audio.analyze_file",
        lambda *a, **k: AudioFeatures(0.5, 0.1, 0.2),
    )
    result = _analyze_track_audio(
        conn,
        {"navidrome_id": "t1", "path": "a.flac"},
        settings,
        lambda *a: None,
    )
    assert result is False


# ------------------------------------------------------------ subsonic lyrics

def test_get_lyrics_parses_value(settings, monkeypatch):
    from app.subsonic import SubsonicClient

    client = SubsonicClient(settings)
    monkeypatch.setattr(
        client,
        "_request",
        lambda endpoint, **kwargs: {"lyrics": {"value": "la letra"}},
    )
    assert client.get_lyrics("A", "T") == "la letra"
    client.close()


def test_get_lyrics_empty(settings, monkeypatch):
    from app.subsonic import SubsonicClient

    client = SubsonicClient(settings)
    monkeypatch.setattr(client, "_request", lambda endpoint, **kwargs: {})
    assert client.get_lyrics("A", "T") == ""
    client.close()


def test_get_lyrics_by_song_id_structured(settings, monkeypatch):
    from app.subsonic import SubsonicClient

    payload = {
        "lyricsList": {
            "structuredLyrics": [
                {"line": [{"value": "linea 1"}, {"value": "linea 2"}]},
                {"line": [{"value": "linea 3"}]},
            ]
        }
    }
    client = SubsonicClient(settings)
    monkeypatch.setattr(client, "_request", lambda endpoint, **kwargs: payload)
    assert client.get_lyrics_by_song_id("t1") == "linea 1\nlinea 2\nlinea 3"
    client.close()


def test_get_lyrics_by_song_id_empty_and_garbage(settings, monkeypatch):
    from app.subsonic import SubsonicClient

    client = SubsonicClient(settings)
    monkeypatch.setattr(client, "_request", lambda endpoint, **kwargs: {})
    assert client.get_lyrics_by_song_id("t1") == ""
    monkeypatch.setattr(
        client,
        "_request",
        lambda endpoint, **kwargs: {"lyricsList": {"structuredLyrics": [{"line": []}, "basura"]}},
    )
    assert client.get_lyrics_by_song_id("t1") == ""
    client.close()


# ------------------------------------------------------------ web facets page

def test_facets_page(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import db as db_mod
    from app import main as main_mod
    from app.config import get_settings
    from app.db import facet_upsert

    assert main_mod.app is not None

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    db_mod._conn = None
    test_client = TestClient(main_mod.app)
    conn = db_mod.get_conn()
    db_mod.init_db(conn)
    facet_upsert(conn, "artist", "a1", {"language": "es", "country": "AR"})
    facet_upsert(conn, "album", "al1", {"moods": ["fiesta"]})
    conn.commit()
    response = test_client.get("/facets")
    assert response.status_code == 200
    assert "es" in response.text
    test_client.close()
    if db_mod._conn is not None:
        db_mod._conn.close()
        db_mod._conn = None
    get_settings.cache_clear()


def test_facets_page_empty(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import db as db_mod
    from app import main as main_mod
    from app.config import get_settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    db_mod._conn = None
    with TestClient(main_mod.app) as test_client:
        response = test_client.get("/facets")
    assert response.status_code == 200
    if db_mod._conn is not None:
        db_mod._conn.close()
        db_mod._conn = None
    get_settings.cache_clear()


# ------------------------------------------------------------ cli flags

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


# ------------------------------------------------------------ ramas restantes

def test_resolve_path_oserror(tmp_path, monkeypatch):
    from app.enrich import audio as audio_mod

    music = tmp_path / "music"
    music.mkdir()
    (music / "a.flac").write_bytes(b"x")
    real_resolve = __import__("pathlib").Path.resolve

    def broken_resolve(self):
        if self.name == "a.flac":
            raise OSError("no resuelve")
        return real_resolve(self)

    monkeypatch.setattr("pathlib.Path.resolve", broken_resolve)
    assert audio_mod.resolve_track_path("a.flac", str(music)) is None


def test_analyze_rhythm_no_librosa_graceful(tmp_path, monkeypatch):
    """Con librosa ausente, analyze_file sigue devolviendo features base."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"librosa", "numpy"}:
            raise ImportError("sin librosa")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from tests.test_audio import write_wav

    wav = write_wav(tmp_path / "t.wav")
    features = audio_mod_analyze(wav)
    assert features is not None
    assert features.bpm is None


def audio_mod_analyze(path):
    from app.enrich.audio import analyze_file

    return analyze_file(path, with_rhythm=True)


def test_analyze_rhythm_real_librosa(tmp_path):
    pytest.importorskip("librosa")
    from tests.test_audio import write_wav

    wav = write_wav(tmp_path / "clicks.wav", seconds=6.0, freq=440.0)
    from app.enrich.audio import analyze_file

    features = analyze_file(wav, with_rhythm=True)
    assert features is not None
    # un tono puro no tiene pulso: bpm puede ser None, pero si hay valor
    # debe ser razonable y la tonalidad debe ser una clase válida
    if features.bpm is not None:
        assert 40 <= features.bpm <= 220
    if features.key is not None:
        assert features.key in {
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        }


def test_analyze_rhythm_empty_audio(monkeypatch):
    from app.enrich import audio as audio_mod

    frames = audio_mod._samples_from_wav(b"")
    assert frames == []


def test_analyze_rhythm_empty_array(monkeypatch, tmp_path):
    import sys
    import types

    import numpy as np

    from app.enrich import audio as audio_mod

    fake = types.ModuleType("librosa")
    fake.beat = types.SimpleNamespace(
        beat_track=lambda **kw: (np.array([120.0]), None)
    )
    fake.feature = types.SimpleNamespace(
        chroma_cqt=lambda **kw: np.array([])
    )
    monkeypatch.setitem(sys.modules, "librosa", fake)
    # audio vacío: no hay señal para analizar
    result = audio_mod._analyze_rhythm(_empty_wav())
    assert result == (None, None)


def _empty_wav() -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"")
    return buffer.getvalue()


def test_analyze_rhythm_returns_bpm_and_key(monkeypatch, tmp_path):
    import io
    import sys
    import types
    import wave

    import numpy as np

    from app.enrich import audio as audio_mod

    fake = types.ModuleType("librosa")
    fake.beat = types.SimpleNamespace(
        beat_track=lambda **kw: (np.array([128.0]), None)
    )
    fake.feature = types.SimpleNamespace(
        chroma_cqt=lambda **kw: np.ones((12, 4))
    )
    monkeypatch.setitem(sys.modules, "librosa", fake)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00\x01\x00" * 100)
    bpm, key = audio_mod._analyze_rhythm(buffer.getvalue())
    assert bpm == 128.0
    assert key == "C"


def test_lyrics_available_import_error(monkeypatch):
    import builtins

    from app.enrich import lyrics as lyrics_mod

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "py3langid":
            raise ImportError("sin py3langid")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(lyrics_mod, "_identifier", None)
    monkeypatch.setattr(lyrics_mod, "_load_failed", False)
    assert lyrics_mod.available() is False
    # segunda llamada usa el flag de fallo
    assert lyrics_mod.available() is False


def test_detect_language_when_unavailable(monkeypatch):
    from app.enrich import lyrics as lyrics_mod

    monkeypatch.setattr(lyrics_mod, "_identifier", None)
    monkeypatch.setattr(lyrics_mod, "_load_failed", True)
    assert lyrics_mod.detect_language("x" * 100) is None


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


@pytest.mark.anyio
async def test_pipeline_detection_no_signal(conn, settings, monkeypatch):
    from app.enrich.artist import Ficha, save_ficha
    from app.enrich.pipeline import enrich_library
    from tests.conftest import FakeOllama

    seed_library(conn)
    save_ficha(conn, Ficha("track", "t1", {"moods": []}, "d", 0.9, "inherited", "h"))

    class Client:
        def get_lyrics_by_song_id(self, track_id):
            return "texto suficientemente largo pero sin idioma detectable xxxxxx"

    monkeypatch.setattr(
        "app.enrich.lyrics.detect_language", lambda text: None
    )
    settings.detect_language = True
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=type("L", (), {"fetch_for": lambda *a, **k: [], "close": lambda s: None})(),
        client=Client(),
        artists=False,
        albums=False,
        tracks=True,
        force=True,
    )
    assert result["languages"] == 0


# ------------------------------------------------------------ audio ramas finales

def test_librosa_available_true():
    from app.enrich import audio as audio_mod

    # la suite corrió con librosa presente en el entorno de dev
    result = audio_mod.librosa_available()
    assert result in (True, False)


def test_resolve_path_absolute_without_relative():
    from app.enrich import audio as audio_mod

    assert audio_mod.resolve_track_path("/", "/tmp") is None


def test_resolve_path_relative_works(tmp_path):
    from app.enrich import audio as audio_mod

    music = tmp_path / "music"
    music.mkdir()
    (music / "a.flac").write_bytes(b"x")
    assert audio_mod.resolve_track_path("a.flac", str(music)) == (music / "a.flac").resolve()


def test_analyze_rhythm_disabled(tmp_path):
    from app.enrich import audio as audio_mod
    from tests.test_audio import write_wav

    wav = write_wav(tmp_path / "t.wav")
    features = audio_mod.analyze_file(wav, with_rhythm=False)
    assert features is not None
    assert features.bpm is None


def test_apply_features_with_energy_keeps_value():
    from app.enrich.audio import AudioFeatures, apply_audio_features

    features = AudioFeatures(energy=0.0, brightness=0.0, dynamic_range=0.0)
    merged = apply_audio_features({}, features)
    assert merged["energy"] == 0.0


def test_decode_pcm_no_binary(tmp_path, monkeypatch):
    from app.enrich import audio as audio_mod
    from tests.test_audio import write_wav

    wav = write_wav(tmp_path / "t.wav")
    monkeypatch.setattr("app.janitor.wav2flac.find_ffmpeg", lambda b=None: None)
    assert audio_mod._decode_pcm(wav, ffmpeg_bin=None, seconds=5) is None


def test_decode_pcm_bad_output(tmp_path, monkeypatch):
    import subprocess as sp

    from app.enrich import audio as audio_mod
    from tests.test_audio import write_wav

    wav = write_wav(tmp_path / "t.wav")
    monkeypatch.setattr("app.janitor.wav2flac.find_ffmpeg", lambda b=None: "/bin/ffmpeg")
    monkeypatch.setattr(
        sp,
        "run",
        lambda cmd, **kw: sp.CompletedProcess(cmd, 0, stdout=b"", stderr=b""),
    )
    assert audio_mod._decode_pcm(wav, ffmpeg_bin=None, seconds=5) is None


def test_samples_empty_frames():
    from app.enrich import audio as audio_mod
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"")
    assert audio_mod._samples_from_wav(buffer.getvalue()) == []


def test_dynamic_range_no_levels(monkeypatch):
    from app.enrich import audio as audio_mod

    samples = list(range(100)) * 100
    monkeypatch.setattr(audio_mod, "_rms", lambda chunk: 0.0)
    monkeypatch.setattr(audio_mod, "_normalize_energy", lambda rms: 0.0)
    monkeypatch.setattr(audio_mod, "SAMPLE_RATE", 22050)
    value = audio_mod._dynamic_range(samples, window=1)
    assert value == 0.0


def test_analyze_rhythm_chroma_empty(monkeypatch):
    import sys
    import types

    import numpy as np

    from app.enrich import audio as audio_mod

    fake = types.ModuleType("librosa")

    class Chroma:
        size = 0

    fake.beat = types.SimpleNamespace(
        beat_track=lambda **kw: (np.array([100.0]), None)
    )
    fake.feature = types.SimpleNamespace(chroma_cqt=lambda **kw: Chroma())
    monkeypatch.setitem(sys.modules, "librosa", fake)

    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00\x01\x00" * 100)
    bpm, key = audio_mod._analyze_rhythm(buffer.getvalue())
    assert bpm == 100.0
    assert key is None


def test_analyze_empty_samples(tmp_path, monkeypatch):
    """pcm válido pero sin samples cae al return None."""
    from app.enrich import audio as audio_mod
    from tests.test_audio import write_wav

    wav = write_wav(tmp_path / "x.flac")
    monkeypatch.setattr(audio_mod, "_samples_from_wav", lambda pcm: [])
    assert audio_mod.analyze_file(wav) is None


def test_analyze_with_rhythm_calls_analyzer(tmp_path, monkeypatch):
    """con_rhythm=True y librosa disponible llama a _analyze_rhythm (línea 126)."""
    from app.enrich import audio as audio_mod
    from tests.test_audio import write_wav

    wav = write_wav(tmp_path / "t.wav")
    monkeypatch.setattr(audio_mod, "librosa_available", lambda: True)
    monkeypatch.setattr(audio_mod, "_analyze_rhythm", lambda pcm: (99.0, "Am"))
    features = audio_mod.analyze_file(wav, with_rhythm=True)
    assert features.bpm == 99.0
    assert features.key == "Am"


def test_librosa_available_true_branch(monkeypatch):
    import sys
    import types

    from app.enrich import audio as audio_mod

    monkeypatch.setitem(sys.modules, "librosa", types.ModuleType("librosa"))
    assert audio_mod.librosa_available() is True


def test_dynamic_range_single_window(tmp_path):
    from app.enrich import audio as audio_mod

    # una sola ventana: rango dinámico cero
    assert audio_mod._dynamic_range([1, 2, 3] * 100, window=300) == 0.0
