from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings, get_settings

log = logging.getLogger("bardo.janitor.wav2flac")

WAV_SUFFIXES = {".wav", ".wave"}


@dataclass
class WavFinding:
    path: Path
    size: int
    converted: bool = False
    flac_path: Path | None = None
    backed_up: bool = False
    deleted: bool = False
    error: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "file": str(self.path),
            "size": self.size,
            "converted": self.converted,
            "flac_path": str(self.flac_path) if self.flac_path else None,
            "backed_up": self.backed_up,
            "deleted": self.deleted,
            "error": self.error,
        }


@dataclass
class WavScanResult:
    music_dir: Path
    findings: list[WavFinding] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.findings)

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.findings)

    def as_dict(self) -> dict[str, object]:
        return {
            "music_dir": str(self.music_dir),
            "count": self.count,
            "total_size": self.total_size,
            "findings": [f.as_dict() for f in self.findings],
        }


def scan_wavs(music_dir: str | Path) -> WavScanResult:
    root = Path(music_dir).expanduser().resolve()
    result = WavScanResult(music_dir=root)
    if not root.exists():
        raise FileNotFoundError(f"music dir not found: {root}")
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in WAV_SUFFIXES:
                path = Path(dirpath) / name
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                result.findings.append(WavFinding(path=path, size=size))
    result.findings.sort(key=lambda f: str(f.path))
    return result


def find_ffmpeg(binary: str | None = None) -> str | None:
    """Localiza ffmpeg: setting explícito, PATH o imageio-ffmpeg (fallback)."""
    if binary:
        candidate = Path(binary).expanduser()
        if candidate.exists():
            return str(candidate)
        found = shutil.which(binary)
        if found:
            return found
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ffmpeg_available(binary: str | None = None) -> bool:
    return find_ffmpeg(binary) is not None


def convert_wav_to_flac(
    wav: Path,
    *,
    backup_dir: Path | None = None,
    delete_original: bool = False,
    overwrite: bool = False,
    ffmpeg_bin: str | None = None,
) -> WavFinding:
    finding = WavFinding(path=wav, size=wav.stat().st_size if wav.exists() else 0)
    if not wav.exists():
        finding.error = "file not found"
        return finding
    flac = wav.with_suffix(".flac")
    finding.flac_path = flac
    if flac.exists() and not overwrite:
        finding.error = "flac already exists (skipped)"
        return finding
    binary = find_ffmpeg(ffmpeg_bin)
    if not binary:
        finding.error = "ffmpeg not installed"
        return finding

    cmd = [
        binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(wav),
        "-c:a",
        "flac",
        "-compression_level",
        "8",
        str(flac),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        finding.error = stderr.splitlines()[-1] if stderr else "ffmpeg failed"
        if flac.exists() and flac.stat().st_size == 0:
            flac.unlink(missing_ok=True)
        return finding

    finding.converted = True
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        target = backup_dir / wav.name
        if target.exists():
            target = backup_dir / f"{wav.stem}.{os.getpid()}{wav.suffix}"
        shutil.copy2(wav, target)
        finding.backed_up = True
    if delete_original:
        wav.unlink()
        finding.deleted = True
    return finding


def convert_all(
    result: WavScanResult,
    *,
    settings: Settings | None = None,
    delete_original: bool | None = None,
    backup_dir: Path | None = None,
    overwrite: bool = False,
    ffmpeg_bin: str | None = None,
    progress: callable | None = None,
) -> WavScanResult:
    settings = settings or get_settings()
    if delete_original is None:
        delete_original = settings.wav_delete_originals
    if backup_dir is None:
        backup_dir = settings.wav_backup_path
    if ffmpeg_bin is None:
        ffmpeg_bin = settings.ffmpeg_bin or None
    converted: list[WavFinding] = []
    for i, finding in enumerate(result.findings, 1):
        updated = convert_wav_to_flac(
            finding.path,
            backup_dir=None if delete_original else backup_dir,
            delete_original=delete_original,
            overwrite=overwrite,
            ffmpeg_bin=ffmpeg_bin,
        )
        converted.append(updated)
        if progress:
            progress(i, result.count, updated)
    result.findings = converted
    return result
