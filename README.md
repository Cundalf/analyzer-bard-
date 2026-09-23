<div align="center">

# 🎧 Bardo

**Fix your library's metadata and build thematic playlists with AI — 100% local, free, portable.**

[English](README.md) · [Español](README.es.md)

[![CI](https://github.com/Cundalf/analyzer-bard-/actions/workflows/ci.yml/badge.svg)](https://github.com/Cundalf/analyzer-bard-/actions/workflows/ci.yml)
![status](https://img.shields.io/badge/status-alpha-orange)
![python](https://img.shields.io/badge/python-3.12%2B-blue)
![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![license](https://img.shields.io/badge/license-MIT-green)
![local](https://img.shields.io/badge/AI-Ollama-purple)

</div>

---

## What is Bardo?

Bardo is a self-hosted web server that sits next to your
[Navidrome](https://www.navidrome.org/) (or any **Subsonic-compatible**) music
server and does two things:

1. **Module A — "Janitor":** fixes library metadata (`[Unknown artist]`,
   `[Unknown album]`, wrong years) using [beets](https://beets.io/) +
   acoustic fingerprinting (Chromaprint → AcoustID → MusicBrainz).
   **Requires disk access** to the same folder Navidrome serves.
2. **Module B — "AI Playlists":** builds thematic playlists from a natural
   language prompt ("tavern celebration", "Lord of the Rings") using an
   **agentic RAG** pipeline. **Pure Subsonic API — no disk needed.**

Everything runs in one container. **Ollama** powers all AI (chat + embeddings),
**sqlite-vec** stores vectors, and the UI is **FastAPI + HTMX + Pico CSS**.
**$0 in external services.**

## Why not SUB/WAVE?

[SUB/WAVE](https://www.getsubwave.com) (and similar tools) mix by **acoustic
similarity**. They cannot select songs by **world knowledge** or accept free
prompts like "fantasy tavern music". Bardo is complementary: Janitor fixes the
metadata that makes acoustic mixing good, and AI Playlists curates by meaning.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  BARDO (FastAPI + Jinja2 + HTMX + Pico CSS)              │
│  one container, no build step                            │
│  ┌──────────────┐          ┌───────────────────────┐     │
│  │ Module A     │          │ Module B              │     │
│  │ Janitor      │          │ AI Playlists          │     │
│  │ (beets)      │          │ (agentic RAG)         │     │
│  └──────┬───────┘          └────┬─────────────┬────┘     │
└─────────┼───────────────────────┼─────────────┼──────────┘
          │                       │             │
     [DISK]                  [Subsonic]     [Ollama]
     bind mount = the        Navidrome/     chat + embed
     same folder Navidrome   gonic/LMS      (gemma4 + nomic)
     serves                       │
          │                  [sqlite-vec]─────┘
          ▼                       │
   beets writes tags ── rescan ──► Navidrome ──► SUB/WAVE
   (subsonicupdate)   (startScan)   (API)      (reads the lib)
```

**Chain of truth:** `files → beets (fix tags) → Navidrome (rescan) → Subsonic
API → Bardo Module B`.

**Golden rule:** Module B inherits Module A's quality. If tags are
`[Unknown]`, the enrichment is garbage → playlists are garbage. **Run A before B.**

## Quickstart (Docker)

Requirements:

- Docker + Docker Compose
- [Ollama](https://ollama.com) running and reachable (local or in another host)
- A Navidrome (or Subsonic-compatible) server
- A free [AcoustID](https://acoustid.org/new-application) API key (for the Janitor)

```bash
git clone https://github.com/Cundalf/analyzer-bard-.git
cd bardo
cp .env.example .env
# edit .env: SUBSONIC_*, OLLAMA_*, ACOUSTID_KEY, MUSIC_PATH...
docker compose up -d --build
```

Open <http://localhost:8080>.

```bash
# Pull the models Bardo needs (once)
ollama pull nomic-embed-text:v1.5
ollama pull gemma4:cloud     # or any tool-calling chat model, e.g. gemma4:12b
```

> **Ollama on Linux:** the default systemd install only listens on
> `127.0.0.1`, so the container can't reach it on a bridge network.
> Use the host-network override:
>
> ```bash
> docker compose -f docker-compose.yml -f docker-compose.host.yml up -d
> ```
>
> Alternatively, expose Ollama on the LAN (e.g. `OLLAMA_HOST=0.0.0.0` in the
> systemd unit) and set `BARDO_OLLAMA_URL=http://<your-ip>:11434` in `.env`.

## Quickstart (local, no Docker)

```bash
# Python 3.12+
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[janitor,lyrics]"   # extras: ffmpeg + fpcalc on PATH; add "audio" for BPM/key
cp .env.example .env

bardo init          # creates the SQLite DB
bardo ping          # checks the Subsonic connection
bardo sync          # mirrors artists/albums/tracks from Navidrome
bardo serve         # UI at http://localhost:8080
```

## CLI

| Command | What it does |
|---|---|
| `bardo init` | Creates the DB and data directories |
| `bardo ping` | Verifies the Subsonic connection |
| `bardo sync` | Mirrors Navidrome's library into SQLite |
| `bardo health` | Library health report (missing years, unknown artist…) |
| `bardo janitor [-p] [-t] [--skip-wav] [path]` | Module A: WAV→FLAC + beets import + rescan |
| `bardo enrich [--limit N] [--force]` | Module B: builds artist/album/track "fichas" |
| `bardo index [--force]` | Builds embeddings into `sqlite-vec` + FTS |
| `bardo facets [--rebuild]` | Inspects the normalized facet index (languages, countries, decades…) |
| `bardo playlist "prompt" [--save] [--no-agent]` | Generates a playlist from a prompt |
| `bardo serve [--host] [--port] [--reload]` | Runs the web UI |

### Safe Janitor flow

1. **Backup** your library (or test on a copy of ~10 tracks).
2. **Dry run:** `bardo janitor -p /music` (pretend — writes nothing).
3. **Interactive:** `bardo janitor -t /music` (timid — asks on every match).
4. Real import + automatic Navidrome rescan.

The run log is a JSONL per file (`old_tags`, `new_tags`, `match_score`,
`album_id`, `status`) that the UI shows as a before/after diff.

## How the AI playlists work

```
prompt ──► [1] expansion (LLM → canonical terms + associations)
        ──► [2] hybrid recall (sqlite-vec + FTS5 + tags) → ~50 candidates
        ──► [3] rerank/selection with LLM (only over the candidates)
        ──► [4] validate real track_ids against Navidrome (anti-hallucination)
        ──► [5] create_playlist → shows up in Navidrome (and SUB/WAVE)
```

- **Fichas (enrichment cards):** one per artist/album/track, with a controlled
  facet vocabulary (`genres`, `moods`, `references`, …) plus a prose
  `description` — the description is the best embedding signal.
- **Hard filters:** language (ISO 639-1), country (ISO 3166-1), decade, genre
  (include/exclude), energy range, instrumental/ballad/concept-album. "Music in
  Spanish", "Argentinian bands", "80s", "calm music" resolve to structured
  filters, not guesswork.
- **Hard data beats the LLM:** Navidrome genre tags, Last.fm crowd tags and
  offline language detection (py3langid, from lyrics) override the model when
  they disagree.
- **Nothing valid is ever lost:** re-enrichment merges by field (lists union,
  scalars only replaced by valid values). Generic placeholders ("Unknown",
  "Various Artists", "Track 01", "N/A") are never treated as data and never
  overwrite a real value.
- **No LLM on junk:** `[Unknown Artist]` / `[Unknown Album]` are marked
  `needs_janitor` instead of being sent to the model (which would hallucinate);
  the dashboard counts them so Module A fixes them.
- **Optional real audio analysis:** loudness/energy, brightness, dynamic range
  always via ffmpeg; BPM and musical key with the `audio` extra (librosa).
  Disabled by default (`ANALYZE_AUDIO=true`, needs `MUSIC_DIR`).
- **Full inheritance:** tracks inherit language, country, genre,
  instrumentation, vocal style and energy from album → artist.
- **Artist/album** are enriched with the LLM. **Tracks** inherit from their
  album/artist, optionally enriched by lyrics/tags (Last.fm) — 5000 tracks
  are never sent to the LLM one by one.
- **Canonicalization:** synonyms are normalized via the `synonyms` table
  (`party/joda/fiesta → fiesta`).
- **Incremental:** every ficha stores a `content_hash`; unchanged entities are
  not re-enriched.
- **Hard rule:** max ~6 tool calls per request. Deterministic pre-filter →
  short agent.

## Configuration

All settings live in `.env` (see `.env.example`). Key ones:

| Variable | Default | Notes |
|---|---|---|
| `SUBSONIC_URL` | `http://localhost:4533` | Navidrome base URL |
| `OLLAMA_CHAT_MODEL` | `gemma4:cloud` | must support tool calling |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text:v1.5` | 768 dims |
| `EMBED_DIM` | `768` | must match the embedding model |
| `JANITOR_ENABLED` | `false` | enables Module A (needs disk) |
| `MUSIC_DIR` | — | same folder Navidrome serves |
| `ENABLE_LASTFM` | `false` | crowd tags via `LASTFM_API_KEY` |
| `ENRICH_TRACKS_LLM` | `false` | per-track LLM enrichment (expensive) |
| `DETECT_LANGUAGE` | `true` | offline lyrics language detection (`lyrics` extra) |
| `ANALYZE_AUDIO` | `false` | real audio features; needs `MUSIC_DIR` (`audio` extra) |
| `WEB_SEARCH_ENABLED` | `false` | surgical factual validation only |

Settings can also be overridden at runtime from **Settings** in the UI
(stored in SQLite, they take precedence over `.env`).

## Testing

```bash
make install       # venv + all extras + dev tools
make test          # 937 tests
make cover         # 100% line + branch coverage (enforced)
make lint          # ruff check + format check
make check         # lint + tests, what CI runs
```

The suite enforces **100% line and branch coverage** (`fail_under = 100` in
`pyproject.toml`) and is linted with [ruff](https://docs.astral.sh/ruff/).
Every feature and error path has tests: Subsonic client (all endpoints,
retries, malformed payloads, lyrics), Ollama (JSON recovery, retries, tool
calling), retrieval (facet filters, fallbacks, anti-hallucination validation),
enrichment (LLM failures, inheritance, staleness, generic filtering,
non-destructive merge), lyrics language detection, audio analysis, Janitor
(WAV→FLAC, beets runner, before/after log), web UI and CLI.

CI runs lint, the test matrix on Python 3.12/3.13 with all extras, and a Docker
build with a healthcheck smoke test (see `.github/workflows/ci.yml`).

## Project layout

```
analyzer-bard-/
├── app/
│   ├── main.py            # FastAPI routes + UI
│   ├── config.py          # env + DB overrides
│   ├── db.py              # SQLite + sqlite-vec + FTS5 + facet index
│   ├── subsonic.py        # Subsonic client (read/write/scan/lyrics)
│   ├── ollama.py          # chat, embeddings, tool calling
│   ├── cli.py             # `bardo` command
│   ├── janitor/           # Module A: wav2flac, beets runner, report, plugin
│   ├── enrich/            # prompts, cards, vocab, generic filter, lyrics, audio
│   ├── index/             # embeddings → vec_fichas
│   ├── agent/             # tools, retrieval, agent loop, playlist runner
│   └── web/               # templates, static, i18n
├── beets-config/          # beets config example (container generates the rest)
├── docs/SPEC.md           # founding design document
├── tests/                 # 937 tests, 100% line + branch coverage
├── .github/workflows/     # CI: lint, tests (3.12/3.13), docker smoke test
├── Dockerfile
└── docker-compose.yml
```

## FAQ

**Does Bardo stream music?** No. Navidrome does that. Bardo only manages
metadata and playlists (via the Subsonic API).

**Does it modify my files?** Only Module A, only through beets, and never
moving/renaming files (`move: no`) so Navidrome IDs stay stable. Use the dry
run first.

**Can I use it with gonic/LMS/Airsonic?** Any Subsonic-compatible server that
implements `createPlaylist` and `startScan` should work.

**Do I need a GPU?** No. Ollama on CPU works; a GPU makes enrichment much
faster. Cloud models via Ollama are also supported, e.g. `gemma4:cloud`.

**Is my data sent anywhere?** Only what you enable: AcoustID/MusicBrainz
(fingerprints, free), Last.fm (optional tags), and your Ollama host. No
telemetry.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the quality bar (100% coverage,
ruff) and the project principles. Notable changes are tracked in
[CHANGELOG.md](CHANGELOG.md), and the founding design lives in
[docs/SPEC.md](docs/SPEC.md).

## License

MIT. See [LICENSE](LICENSE).
