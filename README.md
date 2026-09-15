<div align="center">

# 🎧 Bardo

**Fix your library's metadata and build thematic playlists with AI — 100% local, free, portable.**

[English](README.md) · [Español](README.es.md)

![status](https://img.shields.io/badge/status-alpha-orange)
![python](https://img.shields.io/badge/python-3.12%2B-blue)
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
git clone https://github.com/cundalf/bardo.git
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
pip install -e ".[janitor]"   # Janitor extras need ffmpeg + fpcalc on PATH
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
  language detection override the model when they disagree.
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
| `WEB_SEARCH_ENABLED` | `false` | surgical factual validation only |

Settings can also be overridden at runtime from **Settings** in the UI
(stored in SQLite, they take precedence over `.env`).

## Testing

```bash
pip install -e ".[janitor,dev]"
make test          # 564 tests
make cover         # 100% line + branch coverage required
make lint          # pyflakes
```

The suite enforces **100% line and branch coverage** (`fail_under = 100` in
`pyproject.toml`). Every feature and error path has tests: Subsonic client
(all endpoints, retries, malformed payloads), Ollama (JSON recovery, retries,
tool calling), retrieval (filters, fallbacks, anti-hallucination validation),
enrichment (LLM failures, inheritance, staleness), Janitor (WAV→FLAC, beets
runner, before/after log), web UI and CLI.

## Project layout

```
bardo/
├── app/
│   ├── main.py            # FastAPI routes + UI
│   ├── config.py          # env + DB overrides
│   ├── db.py              # SQLite + sqlite-vec + FTS5
│   ├── subsonic.py        # Subsonic client (read/write/scan)
│   ├── ollama.py          # chat, embeddings, tool calling
│   ├── cli.py             # `bardo` command
│   ├── janitor/           # Module A: wav2flac, beets runner, report
│   ├── enrich/            # prompts, artist/album/track cards, Last.fm
│   ├── index/             # embeddings → vec_fichas
│   ├── agent/             # tools, retrieval, ReAct loop
│   └── web/               # templates + static
├── beets-config/          # beets config for the container
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

## License

MIT. See [LICENSE](LICENSE).
