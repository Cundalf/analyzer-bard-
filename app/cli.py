from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from app.config import get_settings
from app.db import get_conn, init_db


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def cmd_init(args: argparse.Namespace) -> int:
    settings = get_settings()
    settings.ensure_dirs()
    conn = get_conn()
    init_db(conn)
    print(f"DB lista en {settings.db_path}")
    return 0


def cmd_ping(args: argparse.Namespace) -> int:
    from app.subsonic import SubsonicClient

    settings = get_settings()
    with SubsonicClient(settings) as client:
        body = client.ping()
    print(json.dumps(body, indent=2, ensure_ascii=False))
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    from app.subsonic import SubsonicClient, sync_library

    settings = get_settings()
    conn = get_conn()
    init_db(conn)
    with SubsonicClient(settings) as client:
        stats = sync_library(conn, client, with_tracks=not args.no_tracks)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    from app.janitor.report import library_health

    conn = get_conn()
    init_db(conn)
    print(json.dumps(library_health(conn), indent=2, ensure_ascii=False))
    return 0


def cmd_janitor(args: argparse.Namespace) -> int:
    from app.janitor.runner import run_janitor

    conn = get_conn()
    init_db(conn)

    def progress(stage: str, payload: dict) -> None:
        print(f"[{stage}] {json.dumps(payload, ensure_ascii=False)[:300]}")

    result = run_janitor(
        conn=conn,
        music_path=args.music,
        pretend=args.pretend,
        timid=args.timid,
        skip_wav=args.skip_wav,
        do_rescan=not args.no_rescan,
        progress=progress,
    )
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False)[:5000])
    return 0 if result.status == "ok" else 1


def cmd_enrich(args: argparse.Namespace) -> int:
    from app.enrich.pipeline import enrich_library
    from app.subsonic import SubsonicClient

    settings = get_settings()
    conn = get_conn()
    init_db(conn)
    if args.audio:
        settings = settings.model_copy(update={"analyze_audio": True})
    client = None
    if settings.detect_language and not args.no_lyrics:
        try:
            client = SubsonicClient(settings)
        except Exception as exc:
            print(f"aviso: sin detección de idioma ({exc})")
    try:
        result = asyncio.run(
            enrich_library(
                conn,
                settings=settings,
                client=client,
                artists=not args.no_artists,
                albums=not args.no_albums,
                tracks=not args.no_tracks,
                limit=args.limit,
                force=args.force,
                progress=lambda s, p: print(f"[{s}] {json.dumps(p, ensure_ascii=False)[:200]}"),
            )
        )
    finally:
        if client is not None:
            client.close()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from app.index.build import build_index

    conn = get_conn()
    init_db(conn)
    result = asyncio.run(
        build_index(
            conn,
            force=args.force,
            limit=args.limit,
            progress=lambda s, p: print(f"[{s}] {json.dumps(p, ensure_ascii=False)[:200]}"),
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_facets(args: argparse.Namespace) -> int:
    from app.db import facet_list, facet_rebuild_all

    conn = get_conn()
    init_db(conn)
    if args.rebuild:
        count = facet_rebuild_all(conn)
        print(json.dumps({"rebuilt": count}, indent=2))
        return 0
    summary = {
        facet: facet_list(conn, args.entity_type, facet)
        for facet in ("languages", "countries", "genres", "decades", "moods")
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_playlist(args: argparse.Namespace) -> int:
    from app.agent.runner import generate_playlist
    from app.subsonic import SubsonicClient

    conn = get_conn()
    init_db(conn)
    settings = get_settings()
    result = asyncio.run(
        generate_playlist(
            conn,
            args.prompt,
            save=args.save,
            use_agent=not args.no_agent,
            client=None if args.save else SubsonicClient(settings),
            progress=lambda s, p: print(f"[{s}] {json.dumps(p, ensure_ascii=False)[:300]}"),
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=args.host or settings.web_host,
        port=args.port or settings.web_port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bardo", description="Bardo CLI")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="crea DB y directorios")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("ping", help="verifica conexión Subsonic")
    p.set_defaults(func=cmd_ping)

    p = sub.add_parser("sync", help="espeja la biblioteca de Navidrome a SQLite")
    p.add_argument("--no-tracks", action="store_true")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("health", help="salud de la biblioteca")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("janitor", help="corre el Módulo A (wav2flac + beets)")
    p.add_argument("music", nargs="?", default=None)
    p.add_argument("-p", "--pretend", action="store_true")
    p.add_argument("-t", "--timid", action="store_true")
    p.add_argument("--skip-wav", action="store_true")
    p.add_argument("--no-rescan", action="store_true")
    p.set_defaults(func=cmd_janitor)

    p = sub.add_parser("enrich", help="genera fichas (artista/álbum/canción)")
    p.add_argument("--no-artists", action="store_true")
    p.add_argument("--no-albums", action="store_true")
    p.add_argument("--no-tracks", action="store_true")
    p.add_argument("--no-lyrics", action="store_true", help="sin detección de idioma")
    p.add_argument("--audio", action="store_true", help="analiza audio (requiere MUSIC_DIR)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_enrich)

    p = sub.add_parser("index", help="construye embeddings + FTS")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("facets", help="inspecciona/rebuild el índice de facetas")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--entity-type", default="track", choices=["track", "album", "artist"])
    p.set_defaults(func=cmd_facets)

    p = sub.add_parser("playlist", help="genera una playlist desde un prompt")
    p.add_argument("prompt")
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-agent", action="store_true")
    p.set_defaults(func=cmd_playlist)

    p = sub.add_parser("serve", help="levanta la UI web")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
