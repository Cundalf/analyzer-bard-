from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import __version__
from app.agent.runner import generate_playlist
from app.config import OVERRIDABLE, runtime_settings
from app.db import get_conn, init_db, set_setting
from app.enrich.canonicalize import Canonicalizer
from app.janitor.report import library_health
from app.web.i18n import detect_lang, translations

log = logging.getLogger("bardo.web")

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))

app = FastAPI(title="Bardo", version=__version__)
app.mount(
    "/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static"
)

_jobs: dict[str, dict[str, Any]] = {}


def _ctx(request: Request, **extra: Any) -> dict[str, Any]:
    lang = detect_lang(request)
    return {
        "t": translations(lang),
        "lang": lang,
        "version": __version__,
        **extra,
    }


def conn() -> Any:
    c = get_conn()
    init_db(c)
    return c


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    c = conn()
    health = library_health(c)
    settings = runtime_settings(c)
    ollama_ok = False
    subsonic_ok = False
    try:
        from app.ollama import OllamaClient

        async def check() -> bool:
            client = OllamaClient(settings)
            try:
                return await client.ping()
            finally:
                await client.close()

        ollama_ok = asyncio.run(check())
    except Exception:
        ollama_ok = False
    try:
        from app.subsonic import SubsonicClient

        with SubsonicClient(settings) as client:
            client.ping()
        subsonic_ok = True
    except Exception:
        subsonic_ok = False

    playlists = c.execute(
        "SELECT id, module, kind, status, started_at, finished_at, stats "
        "FROM runs ORDER BY id DESC LIMIT 10"
    ).fetchall()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        _ctx(
            request,
            health=health,
            runs=[dict(r) for r in playlists],
            ollama_ok=ollama_ok,
            subsonic_ok=subsonic_ok,
            settings=settings,
        ),
    )


@app.get("/library", response_class=HTMLResponse)
def library(request: Request) -> HTMLResponse:
    c = conn()
    health = library_health(c)
    runs = [
        dict(r)
        for r in c.execute(
            "SELECT * FROM runs WHERE module = 'janitor' ORDER BY id DESC LIMIT 20"
        ).fetchall()
    ]
    settings = runtime_settings(c)
    return templates.TemplateResponse(
        request, "library.html", _ctx(request, health=health, runs=runs, settings=settings)
    )


@app.post("/library/fix", response_class=HTMLResponse)
def library_fix(
    request: Request, background: BackgroundTasks, pretend: bool = Form(False)
) -> HTMLResponse:
    from app.janitor.runner import run_janitor

    settings = runtime_settings(conn())

    def job() -> None:
        try:
            run_janitor(settings=settings, conn=get_conn(), pretend=pretend)
        except Exception as exc:
            log.exception("janitor job failed: %s", exc)

    background.add_task(job)
    mode = "pretend (simulación)" if pretend else "import real"
    return templates.TemplateResponse(
        request,
        "partials/job_started.html",
        _ctx(request, job=f"Janitor {mode} iniciado en background."),
    )


@app.get("/library/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: int) -> HTMLResponse:
    c = conn()
    run = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if not run:
        return HTMLResponse("<p>Run no encontrado</p>", status_code=404)
    entries = [
        dict(r)
        for r in c.execute(
            "SELECT * FROM import_log WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
    ]
    for entry in entries:
        try:
            entry["old"] = json.loads(entry["old_tags"] or "{}")
            entry["new"] = json.loads(entry["new_tags"] or "{}")
            entry["changed"] = {
                k: {"old": entry["old"].get(k), "new": entry["new"].get(k)}
                for k in set(entry["old"]) | set(entry["new"])
                if entry["old"].get(k) != entry["new"].get(k)
            }
        except json.JSONDecodeError:
            entry["changed"] = {}
    stats = {}
    try:
        stats = json.loads(run["stats"] or "{}")
    except json.JSONDecodeError:
        pass
    return templates.TemplateResponse(
        request,
        "run_detail.html",
        _ctx(request, run=dict(run), entries=entries, stats=stats),
    )


@app.get("/playlists", response_class=HTMLResponse)
def playlists(request: Request) -> HTMLResponse:
    c = conn()
    settings = runtime_settings(c)
    existing: list[dict[str, Any]] = []
    try:
        from app.subsonic import SubsonicClient

        with SubsonicClient(settings) as client:
            existing = [
                {"id": p.id, "name": p.name, "song_count": p.song_count}
                for p in client.get_playlists()
            ]
    except Exception as exc:
        log.warning("could not list playlists: %s", exc)
    return templates.TemplateResponse(
        request, "playlists.html", _ctx(request, settings=settings, existing=existing)
    )


@app.post("/playlists/generate", response_class=HTMLResponse)
async def playlists_generate(
    request: Request,
    prompt: str = Form(...),
    use_agent: bool = Form(False),
) -> HTMLResponse:
    c = conn()
    settings = runtime_settings(c)
    from app.subsonic import SubsonicClient

    try:
        client = SubsonicClient(settings)
    except Exception as exc:
        log.exception("subsonic client failed")
        return templates.TemplateResponse(
            request, "partials/playlist_error.html", _ctx(request, error=str(exc))
        )
    try:
        result = await generate_playlist(
            c,
            prompt,
            settings=settings,
            client=client,
            save=False,
            use_agent=use_agent,
        )
    except Exception as exc:
        log.exception("generate failed")
        return templates.TemplateResponse(
            request, "partials/playlist_error.html", _ctx(request, error=str(exc))
        )
    finally:
        client.close()
    return templates.TemplateResponse(
        request, "partials/playlist_preview.html", _ctx(request, result=result, prompt=prompt)
    )


@app.post("/playlists/save", response_class=HTMLResponse)
def playlists_save(
    request: Request,
    name: str = Form(...),
    track_ids: str = Form(...),
) -> HTMLResponse:
    c = conn()
    settings = runtime_settings(c)
    ids = [t.strip() for t in track_ids.split(",") if t.strip()]
    if not ids:
        return templates.TemplateResponse(
            request, "partials/playlist_error.html", _ctx(request, error="sin track_ids")
        )
    try:
        from app.subsonic import SubsonicClient

        with SubsonicClient(settings) as client:
            playlist = client.create_playlist(name, ids)
        return templates.TemplateResponse(
            request,
            "partials/playlist_saved.html",
            _ctx(
                request,
                playlist={
                    "id": playlist.id,
                    "name": playlist.name,
                    "song_count": playlist.song_count or len(ids),
                },
                settings=settings,
            ),
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request, "partials/playlist_error.html", _ctx(request, error=str(exc))
        )


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> HTMLResponse:
    c = conn()
    stored = {
        r["key"]: r["value"]
        for r in c.execute("SELECT key, value FROM settings").fetchall()
    }
    settings = runtime_settings(c)
    return templates.TemplateResponse(
        request,
        "settings.html",
        _ctx(
            request,
            settings=settings,
            stored=stored,
            overridable=OVERRIDABLE,
            models=_list_models(settings),
        ),
    )


def _list_models(settings: Any) -> list[str]:
    try:
        from app.ollama import OllamaClient

        async def run() -> list[str]:
            client = OllamaClient(settings)
            try:
                return await client.list_models()
            finally:
                await client.close()

        return asyncio.run(run())
    except Exception:
        return []


@app.post("/settings")
def settings_save(request: Request) -> RedirectResponse:
    return RedirectResponse("/settings", status_code=303)


@app.post("/settings/save")
async def settings_save_form(request: Request) -> Any:
    c = conn()
    form = await request.form()
    for key in OVERRIDABLE:
        if key in form:
            set_setting(c, key, str(form[key]))
    return RedirectResponse("/settings", status_code=303)


@app.get("/api/health")
def api_health() -> JSONResponse:
    c = conn()
    settings = runtime_settings(c)
    return JSONResponse(
        {
            "version": __version__,
            "library": library_health(c),
            "subsonic_ok": _check_subsonic(settings),
            "janitor_enabled": settings.janitor_enabled,
        }
    )


def _check_subsonic(settings: Any) -> bool:
    try:
        from app.subsonic import SubsonicClient

        with SubsonicClient(settings) as client:
            client.ping()
        return True
    except Exception:
        return False


@app.get("/api/synonyms")
def api_synonyms() -> JSONResponse:
    c = conn()
    canon = Canonicalizer.from_db(c)
    return JSONResponse({"synonyms": canon.synonyms})


@app.post("/api/sync")
def api_sync() -> JSONResponse:
    from app.subsonic import SubsonicClient, sync_library

    c = conn()
    settings = runtime_settings(c)
    with SubsonicClient(settings) as client:
        stats = sync_library(c, client)
    return JSONResponse(stats)
