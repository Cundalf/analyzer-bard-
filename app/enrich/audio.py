"""Análisis acústico real de archivos (opcional, requiere disco).

Datos objetivos por canción:
- energy: RMS normalizado (dBFS → 0..1), siempre disponible con ffmpeg.
- brightness: tasa de cruces por cero, proxy de brillo espectral.
- dynamic_range: diferencia entre RMS alto y bajo.
- bpm y key: sólo si `librosa` está instalado (extra "audio").

Es un dato duro: la energía medida pisa la estimación del LLM.
Todo degrada a no-op si falta ffmpeg, el archivo o MUSIC_DIR.
"""

from __future__ import annotations

import array
import io
import logging
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("bardo.enrich.audio")

SAMPLE_RATE = 22050
ANALYSIS_SECONDS = 30
FULL_SCALE_DB = 90.0
SILENCE_DB = -60.0
BPM_MIN = 40.0
BPM_MAX = 220.0


@dataclass
class AudioFeatures:
    energy: float
    brightness: float
    dynamic_range: float
    bpm: float | None = None
    key: str | None = None
    analyzed_seconds: float = 0.0

    def as_facets(self) -> dict[str, Any]:
        facets: dict[str, Any] = {
            "energy": round(self.energy, 3),
            "energy_source": "audio",
            "brightness": round(self.brightness, 3),
            "dynamic_range": round(self.dynamic_range, 3),
        }
        if self.bpm is not None:
            facets["bpm"] = round(self.bpm, 1)
        if self.key is not None:
            facets["key"] = self.key
        return facets


def librosa_available() -> bool:
    try:
        import librosa  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_track_path(track_path: str | None, music_dir: str | None) -> Path | None:
    """Resuelve la ruta local de un track contra MUSIC_DIR, sin escaparla.

    Navidrome guarda rutas absolutas vistas *dentro* de su contenedor
    (ej. `/music/Artist/Album/a.flac`). Se prueban en orden:
    1. la ruta tal cual (si existe en este host);
    2. la ruta relativa a MUSIC_DIR, quitando el prefijo;
    y se rechaza cualquier resultado fuera de MUSIC_DIR.
    """
    if not track_path or not music_dir:
        return None
    root = Path(music_dir).expanduser().resolve()
    raw = str(track_path).strip()
    candidates: list[Path] = []
    direct = Path(raw).expanduser()
    if direct.is_absolute():
        if direct.is_file():
            candidates.append(direct)
        relative = raw.lstrip("/\\")
        if relative:
            candidates.append(root / relative)
    else:
        candidates.append(root / raw.lstrip("/\\"))
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        if resolved != root and root not in resolved.parents:
            continue
        return resolved
    return None


def analyze_file(
    path: str | Path,
    *,
    ffmpeg_bin: str | None = None,
    seconds: int = ANALYSIS_SECONDS,
    with_rhythm: bool = True,
) -> AudioFeatures | None:
    """Analiza un archivo de audio. Devuelve None si no se puede."""
    file_path = Path(path)
    if not file_path.is_file():
        return None
    pcm = _decode_pcm(file_path, ffmpeg_bin=ffmpeg_bin, seconds=seconds)
    if not pcm:
        return None
    samples = _samples_from_wav(pcm)
    if not samples:
        return None
    rms = _rms(samples)
    energy = _normalize_energy(rms)
    brightness = _zero_crossing_rate(samples)
    dynamic_range = _dynamic_range(samples)
    bpm: float | None = None
    key: str | None = None
    if with_rhythm and librosa_available():
        bpm, key = _analyze_rhythm(pcm)
    return AudioFeatures(
        energy=energy,
        brightness=brightness,
        dynamic_range=dynamic_range,
        bpm=bpm,
        key=key,
        analyzed_seconds=len(samples) / SAMPLE_RATE,
    )


def apply_audio_features(facets: dict[str, Any], features: AudioFeatures) -> dict[str, Any]:
    """Pisa energía/moods objetivos sobre una ficha (dato duro)."""
    from app.enrich.merge import merge_hard_facets

    merged = merge_hard_facets(facets)
    merged.update(features.as_facets())
    return merged


# ---------------------------------------------------------------- decode


def _find_ffmpeg(binary: str | None) -> str | None:
    from app.janitor.wav2flac import find_ffmpeg

    return find_ffmpeg(binary)


def _decode_pcm(path: Path, *, ffmpeg_bin: str | None, seconds: int) -> bytes | None:
    import subprocess

    binary = _find_ffmpeg(ffmpeg_bin)
    if not binary:
        return None
    cmd = [
        binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-t",
        str(max(1, int(seconds))),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "wav",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("audio decode failed for %s: %s", path, exc)
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return proc.stdout


def _samples_from_wav(pcm: bytes) -> list[int]:
    try:
        with wave.open(io.BytesIO(pcm), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            channels = wav.getnchannels()
    except (wave.Error, EOFError):
        return []
    if not frames:
        return []
    ints = array.array("h")
    ints.frombytes(frames[: len(frames) - (len(frames) % 2)])
    if channels > 1:
        ints = array.array("h", ints[::channels])
    return list(ints)


def _rms(samples: list[int]) -> float:
    if not samples:
        return 0.0
    total = sum(float(s) * float(s) for s in samples)
    return math.sqrt(total / len(samples))


def _normalize_energy(rms: float) -> float:
    if rms <= 0:
        return 0.0
    db = 20.0 * math.log10(rms / 32768.0)
    if db <= SILENCE_DB:
        return 0.0
    if db >= 0:
        return 1.0
    span = -SILENCE_DB
    return round(min(1.0, max(0.0, (db - SILENCE_DB) / span)), 3)


def _zero_crossing_rate(samples: list[int]) -> float:
    if len(samples) < 2:
        return 0.0
    crossings = 0
    previous = samples[0]
    for sample in samples[1:]:
        if (sample >= 0) != (previous >= 0):
            crossings += 1
        previous = sample
    return round(min(1.0, crossings / (len(samples) - 1)), 3)


def _dynamic_range(samples: list[int], window: int = 2205) -> float:
    """Diferencia (0..1) entre el RMS más alto y el más bajo por ventanas."""
    if len(samples) < window * 2:
        return 0.0
    levels: list[float] = []
    for start in range(0, len(samples) - window + 1, window):
        chunk = samples[start : start + window]
        levels.append(_normalize_energy(_rms(chunk)))
    return round(max(levels) - min(levels), 3)


# ---------------------------------------------------------------- rhythm

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _analyze_rhythm(pcm: bytes) -> tuple[float | None, str | None]:
    """BPM y tonalidad con librosa (si está). Nunca rompe el análisis."""
    try:
        import librosa
        import numpy as np

        with wave.open(io.BytesIO(pcm), "rb") as wav:
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return None, None
        tempo, _ = librosa.beat.beat_track(y=audio, sr=rate)
        bpm = float(np.atleast_1d(tempo)[0])
        # 0 o valores absurdos indican que no hay pulso detectable
        if not (BPM_MIN <= bpm <= BPM_MAX):
            bpm = None
        chroma = librosa.feature.chroma_cqt(y=audio, sr=rate)
        if chroma.size == 0:
            return bpm, None
        index = int(np.argmax(chroma.mean(axis=1)))
        return bpm, _PITCH_CLASSES[index % 12]
    except Exception as exc:  # pragma: no cover - defensivo
        log.warning("rhythm analysis failed: %s", exc)
        return None, None
