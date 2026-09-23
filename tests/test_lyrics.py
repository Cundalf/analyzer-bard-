from __future__ import annotations

import pytest
from tests.conftest import seed_library

from app.enrich import lyrics as lyrics_mod
from app.enrich.pipeline import (
    _detect_track_language,
    _fetch_lyrics,
    row_id,
)

SPANISH_LYRICS = """\
[Verse 1]
Persiana americana
la noche es un rumor
y en el silencio de la calle
te vuelvo a ver pasar

[Chorus]
No quiero verte más
no quiero verte más
"""

ENGLISH_LYRICS = """\
[Verse 1]
I walk alone through the empty streets
and every shadow calls your name
the city lights are fading out
and nothing feels the same

[Chorus]
And I will carry on
carry on
"""


class FakeClient:
    def __init__(self, by_id=None, by_name=None):
        self.by_id = by_id or {}
        self.by_name = by_name or {}
        self.calls: list[tuple] = []
        self.closed = False

    def get_lyrics_by_song_id(self, track_id):
        self.calls.append(("id", track_id))
        return self.by_id.get(track_id, "")

    def get_lyrics(self, artist, title):
        self.calls.append(("name", artist, title))
        return self.by_name.get((artist, title), "")

    def close(self):
        self.closed = True


# ------------------------------------------------------------ detección


def test_available_without_dependency(monkeypatch):
    monkeypatch.setattr(lyrics_mod, "_identifier", None)
    monkeypatch.setattr(lyrics_mod, "_load_failed", True)
    assert lyrics_mod.available() is False


def test_detect_language_none_and_short():
    assert lyrics_mod.detect_language("") is None
    assert lyrics_mod.detect_language("corto") is None
    assert lyrics_mod.detect_language(None) is None


def test_clean_lyrics_removes_section_markers():
    clean = lyrics_mod._clean_lyrics(SPANISH_LYRICS)
    assert "[Verse 1]" not in clean
    assert "[Chorus]" not in clean
    assert "Persiana americana" in clean


def test_clean_lyrics_removes_timestamps_and_html():
    text = "[00:12.34] <i>Hola mundo</i>\n[00:15.00] segunda linea"
    clean = lyrics_mod._clean_lyrics(text)
    assert "00:12" not in clean
    assert "<i>" not in clean
    assert "Hola mundo" in clean


def test_detect_language_real_spanish():
    if not lyrics_mod.available():
        pytest.skip("py3langid no instalado")
    result = lyrics_mod.detect_language(SPANISH_LYRICS)
    assert result is not None
    assert result[0] in {"es", "pt", "gl", "ca"}


def test_detect_language_real_english():
    if not lyrics_mod.available():
        pytest.skip("py3langid no instalado")
    result = lyrics_mod.detect_language(ENGLISH_LYRICS)
    assert result is not None
    assert result[0] == "en"


def test_detect_language_handles_identifier_error(monkeypatch):
    class Boom:
        @staticmethod
        def classify(text):
            raise RuntimeError("modelo roto")

    monkeypatch.setattr(lyrics_mod, "_identifier", Boom)
    monkeypatch.setattr(lyrics_mod, "_load_failed", False)
    assert lyrics_mod.detect_language(SPANISH_LYRICS) is None


def test_detect_language_und_and_zxx(monkeypatch):
    class Und:
        @staticmethod
        def classify(text):
            return ("und", 0.9)

    monkeypatch.setattr(lyrics_mod, "_identifier", Und)
    monkeypatch.setattr(lyrics_mod, "_load_failed", False)
    assert lyrics_mod.detect_language(SPANISH_LYRICS) is None


def test_detect_language_low_confidence(monkeypatch):
    class Low:
        @staticmethod
        def classify(text):
            return ("es", 0.1)

    monkeypatch.setattr(lyrics_mod, "_identifier", Low)
    monkeypatch.setattr(lyrics_mod, "_load_failed", False)
    assert lyrics_mod.detect_language(SPANISH_LYRICS) is None


def test_detect_language_normalizes_confidence(monkeypatch):
    class High:
        @staticmethod
        def classify(text):
            return ("es", 5.0)

    monkeypatch.setattr(lyrics_mod, "_identifier", High)
    monkeypatch.setattr(lyrics_mod, "_load_failed", False)
    code, confidence = lyrics_mod.detect_language(SPANISH_LYRICS)
    assert code == "es"
    assert confidence == 1.0


def test_detect_language_non_numeric_score(monkeypatch):
    class Weird:
        @staticmethod
        def classify(text):
            return ("es", "no-num")

    monkeypatch.setattr(lyrics_mod, "_identifier", Weird)
    monkeypatch.setattr(lyrics_mod, "_load_failed", False)
    code, confidence = lyrics_mod.detect_language(SPANISH_LYRICS)
    assert code == "es"
    assert confidence == 0.5


def test_detect_language_log_prob_score(monkeypatch):
    class LogProb:
        @staticmethod
        def classify(text):
            return ("en", -68.5)

    monkeypatch.setattr(lyrics_mod, "_identifier", LogProb)
    monkeypatch.setattr(lyrics_mod, "_load_failed", False)
    code, confidence = lyrics_mod.detect_language(ENGLISH_LYRICS)
    assert code == "en"
    assert confidence == 0.5


# ------------------------------------------------------------ integración pipeline


class LyricsLastFm:
    def fetch_for(self, *a, **k):
        return []

    def close(self):
        pass


@pytest.mark.anyio
async def test_pipeline_detects_language_from_lyrics(conn, settings):
    from tests.conftest import FakeOllama

    from app.enrich.artist import Ficha, get_ficha, save_ficha
    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    save_ficha(conn, Ficha("track", "t1", {"moods": []}, "d", 0.9, "inherited", "h"))
    client = FakeClient(by_id={"t1": SPANISH_LYRICS})
    settings.detect_language = True
    await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=LyricsLastFm(),
        client=client,
        artists=False,
        albums=False,
        tracks=True,
        force=True,
    )
    ficha = get_ficha(conn, "track", "t1")
    detected = ficha["facets"].get("language")
    if detected is None:
        pytest.skip("py3langid no instalado")
    assert detected in {"es", "pt", "gl", "ca"}
    assert ficha["facets"]["language_source"] == "lyrics"


@pytest.mark.anyio
async def test_pipeline_language_disabled(conn, settings):
    from tests.conftest import FakeOllama

    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    settings.detect_language = False
    client = FakeClient(by_id={"t1": SPANISH_LYRICS})
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=LyricsLastFm(),
        client=client,
        artists=False,
        albums=False,
        tracks=True,
        force=True,
    )
    assert result["languages"] == 0
    assert client.calls == []


@pytest.mark.anyio
async def test_pipeline_lyrics_cached_once(conn, settings):
    from tests.conftest import FakeOllama

    from app.enrich.artist import Ficha, save_ficha
    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    save_ficha(conn, Ficha("track", "t1", {"moods": []}, "d", 0.9, "inherited", "h"))
    client = FakeClient(by_id={"t1": SPANISH_LYRICS})
    settings.detect_language = True
    for _ in range(2):
        await enrich_library(
            conn,
            settings=settings,
            ollama=FakeOllama(),
            lastfm=LyricsLastFm(),
            client=client,
            artists=False,
            albums=False,
            tracks=True,
            force=True,
        )
    id_calls = [c for c in client.calls if c[0] == "id"]
    # 4 tracks, una llamada cada uno en la primera corrida; la segunda usa
    # la caché sólo para los que sí tenían letra (t1)
    assert len(id_calls) == 7  # 4 + 3 sin letra que se reintentan
    cached = conn.execute("SELECT COUNT(*) AS n FROM lyrics_cache").fetchone()["n"]
    assert cached == 1


@pytest.mark.anyio
async def test_pipeline_falls_back_to_artist_title(conn, settings):
    from tests.conftest import FakeOllama

    from app.enrich.artist import Ficha, save_ficha
    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    save_ficha(conn, Ficha("track", "t1", {"moods": []}, "d", 0.9, "inherited", "h"))
    client = FakeClient(by_id={}, by_name={("Wind Rose", "Drunken Dwarves"): ENGLISH_LYRICS})
    settings.detect_language = True
    await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=LyricsLastFm(),
        client=client,
        artists=False,
        albums=False,
        tracks=True,
        force=True,
    )
    fallback = [c for c in client.calls if c[0] == "name"]
    assert fallback


@pytest.mark.anyio
async def test_pipeline_no_lyrics_no_detection(conn, settings):
    from tests.conftest import FakeOllama

    from app.enrich.artist import Ficha, save_ficha
    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    save_ficha(conn, Ficha("track", "t1", {"moods": []}, "d", 0.9, "inherited", "h"))
    client = FakeClient()
    settings.detect_language = True
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=LyricsLastFm(),
        client=client,
        artists=False,
        albums=False,
        tracks=True,
        force=True,
    )
    assert result["languages"] == 0


@pytest.mark.anyio
async def test_pipeline_lyrics_client_error_tolerated(conn, settings):
    from tests.conftest import FakeOllama

    from app.enrich.artist import Ficha, save_ficha
    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    save_ficha(conn, Ficha("track", "t1", {"moods": []}, "d", 0.9, "inherited", "h"))

    class Broken(FakeClient):
        def get_lyrics_by_song_id(self, track_id):
            raise RuntimeError("navidrome caído")

    settings.detect_language = True
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=LyricsLastFm(),
        client=Broken(),
        artists=False,
        albums=False,
        tracks=True,
        force=True,
    )
    assert result["languages"] == 0


@pytest.mark.anyio
async def test_pipeline_no_ficha_for_track(conn, settings):
    from tests.conftest import FakeOllama

    from app.enrich.pipeline import enrich_library

    seed_library(conn)
    client = FakeClient(by_id={"t1": SPANISH_LYRICS})
    settings.detect_language = True
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=LyricsLastFm(),
        client=client,
        artists=False,
        albums=False,
        tracks=False,
        force=True,
    )
    assert result["languages"] == 0


@pytest.mark.anyio
async def test_pipeline_detection_without_navidrome_id(conn, settings):
    from tests.conftest import FakeOllama

    from app.enrich.pipeline import enrich_library

    conn.execute(
        "INSERT INTO tracks(id, navidrome_id, title, artist_id) "
        "VALUES ('track:local', NULL, 'Local', NULL)"
    )
    conn.commit()
    settings.detect_language = True
    client = FakeClient(by_id={"track:local": SPANISH_LYRICS})
    result = await enrich_library(
        conn,
        settings=settings,
        ollama=FakeOllama(),
        lastfm=LyricsLastFm(),
        client=client,
        artists=False,
        albums=False,
        tracks=False,
        force=True,
    )
    assert result["languages"] == 0


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

    lyrics = _fetch_lyrics(Client(), {"navidrome_id": "t1", "title": "T", "artist_name": "A"})
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

    lyrics = _fetch_lyrics(OldClient(), {"navidrome_id": "t1", "title": "T", "artist_name": "A"})
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
    cached = conn.execute("SELECT lyrics FROM lyrics_cache WHERE track_id = 't1'").fetchone()
    assert cached is None


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
