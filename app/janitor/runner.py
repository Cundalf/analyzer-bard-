from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.config import Settings, get_settings
from app.janitor import report as report_mod
from app.janitor.wav2flac import convert_all, scan_wavs

log = logging.getLogger("bardo.janitor.runner")


class JanitorError(Exception):
    pass


@dataclass
class JanitorResult:
    run_id: int | None = None
    wav: dict[str, Any] = field(default_factory=dict)
    beets: dict[str, Any] = field(default_factory=dict)
    diffs: list[report_mod.TagDiff] = field(default_factory=list)
    rescan: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "wav": self.wav,
            "beets": self.beets,
            "diffs": [d.as_dict() for d in self.diffs],
            "rescan": self.rescan,
            "status": self.status,
            "errors": self.errors,
        }


def default_beets_config() -> dict[str, Any]:
    return {
        "directory": os.environ.get("MUSIC_DIR", "/music"),
        "library": os.environ.get("BEETS_LIBRARY", "/config/library.db"),
        "import": {
            "write": True,
            "move": False,
            "copy": False,
            "autotag": True,
            "detail": True,
            "quiet": True,
            "resume": False,
            "incremental": False,
        },
        "plugins": [
            "chroma",
            "fromfilename",
            "fetchart",
            "mbsync",
            "subsonicupdate",
            "lyrics",
            "bardo",
        ],
        "pluginpath": [str(Path(__file__).resolve().parent)],
        "bardo": {
            "logpath": os.environ.get(
                "BARDO_BEETS_LOG", "/data/runs/beets.jsonl"
            )
        },
        "subsonic": {
            "url": os.environ.get("SUBSONIC_URL", "http://localhost:4533"),
            "user": os.environ.get("SUBSONIC_USER", "admin"),
            "pass": os.environ.get("SUBSONIC_PASS", ""),
            "auth": "token",
        },
    }


def ensure_beets_config(settings: Settings | None = None) -> Path | None:
    """Crea config.yaml en BEETSDIR si no existe, desde los settings.

    Permite que el Janitor funcione out-of-the-box en el container sin editar
    YAML a mano. Si el usuario ya tiene un config.yaml, no se toca.
    """
    settings = settings or get_settings()
    if settings.beets_config:
        return Path(settings.beets_config).expanduser().resolve()
    beetsdir = Path(settings.beetsdir or "/config").expanduser()
    if not beetsdir.exists():
        return None
    target = beetsdir / "config.yaml"
    if target.exists():
        return target

    import yaml

    config = default_beets_config()
    config["directory"] = settings.music_dir or config["directory"]
    config["library"] = str(beetsdir / "library.db")
    config["bardo"]["logpath"] = str(settings.runs_path / "beets.jsonl")
    if settings.acoustid_key:
        config["acoustid"] = {"apikey": settings.acoustid_key}
    config["subsonic"] = {
        "url": settings.subsonic_url,
        "user": settings.subsonic_user,
        "pass": settings.subsonic_pass,
        "auth": "token",
    }
    target.write_text(yaml.safe_dump(config, sort_keys=False))
    log.info("wrote default beets config: %s", target)
    return target


def _bool_opt(config: dict[str, Any], dotpath: str, default: Any) -> Any:
    node: Any = config
    for part in dotpath.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def run_beets(
    music_path: str | Path,
    *,
    settings: Settings | None = None,
    mode: str = "import",
    pretend: bool = False,
    timid: bool = False,
    logpath: Path | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Ejecuta beets como subproceso, devolviendo código y logs.

    mode: "import" | "config". En "config" sólo imprime la config efectiva.
    """
    settings = settings or get_settings()
    beets_bin = shutil.which("beet") or settings.beets_bin
    if not beets_bin:
        raise JanitorError(
            "beet no encontrado. Instalá beets (extra 'janitor') o usá Docker."
        )

    env = dict(os.environ)
    env["BEETSDIR"] = settings.beetsdir
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    )
    if logpath:
        env["BARDO_BEETS_LOG"] = str(logpath)
    env.setdefault("MUSIC_DIR", settings.music_dir or "/music")
    env.setdefault("SUBSONIC_URL", settings.subsonic_url)
    env.setdefault("SUBSONIC_USER", settings.subsonic_user)
    env.setdefault("SUBSONIC_PASS", settings.subsonic_pass)

    if mode == "config":
        cmd = [beets_bin, "config", "-p"]
    else:
        cmd = [beets_bin, "import"]
        if pretend:
            cmd.append("-p")
        if timid:
            cmd.append("-t")
        cmd.append(str(music_path))
    if extra_args:
        cmd.extend(extra_args)

    log.info("running: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(settings.beetsdir) if settings.beetsdir else None,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-20000:],
        "stderr": proc.stderr[-20000:],
    }


def rescan_navidrome(
    settings: Settings | None = None,
    *,
    full: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    if client is None:
        from app.subsonic import SubsonicClient

        client = SubsonicClient(settings)
        close = True
    else:
        close = False
    try:
        status = client.start_scan(full=full)
        return {"ok": True, "status": status}
    except Exception as exc:
        log.warning("rescan failed: %s", exc)
        return {"ok": False, "error": str(exc)}
    finally:
        if close:
            client.close()


def run_janitor(
    *,
    settings: Settings | None = None,
    conn: Any | None = None,
    client: Any | None = None,
    music_path: str | Path | None = None,
    pretend: bool = False,
    timid: bool = False,
    skip_wav: bool = False,
    do_rescan: bool = True,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> JanitorResult:
    """Orquesta el pipeline del Módulo A.

    1. WAV → FLAC
    2. beets import (pretend/timid/real)
    3. rescan Navidrome
    4. log JSONL + ingest a import_log
    """
    settings = settings or get_settings()
    if not settings.janitor_enabled and not pretend:
        raise JanitorError(
            "Janitor deshabilitado (JANITOR_ENABLED=false). "
            "Usá pretend o habilitá Janitor."
        )
    if not settings.music_dir:
        raise JanitorError("MUSIC_DIR vacío: el Janitor necesita acceso a disco")

    settings.ensure_dirs()
    ensure_beets_config(settings)
    music = Path(music_path or settings.music_dir)
    result = JanitorResult()

    run_id = None
    if conn is not None:
        from app.db import start_run

        run_id = start_run(conn, "janitor", "pretend" if pretend else "import")
        result.run_id = run_id

    if progress:
        progress("start", {"run_id": run_id, "music": str(music)})

    if not skip_wav:
        try:
            scan = scan_wavs(music)
            summary = scan.as_dict()
            if pretend:
                result.wav = {**summary, "converted": 0}
            else:
                convert_all(
                    scan, settings=settings, ffmpeg_bin=settings.ffmpeg_bin or None
                )
                result.wav = scan.as_dict()
                result.wav["converted"] = sum(
                    1 for f in scan.findings if f.converted
                )
            if progress:
                progress("wav", result.wav)
        except Exception as exc:
            result.errors.append(f"wav: {exc}")
            log.exception("wav stage failed")
    else:
        result.wav = {"skipped": True}

    logpath = settings.runs_path / f"beets_{run_id or 'manual'}.jsonl"
    try:
        beets_out = run_beets(
            music,
            settings=settings,
            pretend=pretend,
            timid=timid,
            logpath=logpath,
        )
        result.beets = {
            "returncode": beets_out["returncode"],
            "logpath": str(logpath),
        }
        if beets_out["returncode"] != 0:
            result.status = "error"
            result.errors.append(
                f"beets rc={beets_out['returncode']}: "
                f"{beets_out['stderr'].strip().splitlines()[-1] if beets_out['stderr'].strip() else ''}"
            )
        if progress:
            progress("beets", result.beets)
    except Exception as exc:
        result.status = "error"
        result.errors.append(f"beets: {exc}")
        log.exception("beets stage failed")

    if logpath.exists():
        try:
            result.diffs = report_mod.read_jsonl(logpath)
            if conn is not None and run_id is not None:
                report_mod.ingest_into_db(conn, run_id, result.diffs)
        except Exception as exc:
            result.errors.append(f"report: {exc}")
            log.exception("report ingest failed")

    if do_rescan and not pretend:
        result.rescan = rescan_navidrome(settings, client=client)
        if not result.rescan.get("ok"):
            result.errors.append(f"rescan: {result.rescan.get('error')}")
        if progress:
            progress("rescan", result.rescan)

    if conn is not None and run_id is not None:
        from app.db import finish_run

        stats = {
            "wav": result.wav,
            "beets": result.beets,
            "diffs": len(result.diffs),
            "errors": result.errors,
        }
        finish_run(
            conn,
            run_id,
            status=result.status if result.status == "ok" else "error",
            stats=stats,
        )

    if progress:
        progress("done", result.as_dict())
    return result
