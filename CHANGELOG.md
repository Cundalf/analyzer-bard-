# Changelog

All notable changes to Bardo are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Language detection from lyrics** (offline): `py3langid` (139 languages,
  BSD-3). Cleans section markers/timestamps/HTML, filters low confidence and
  `und`/`zxx`, degrades to a no-op without the dependency. Lyrics are fetched
  via OpenSubsonic `getLyrics`/`getLyricsBySongId` and cached (empty lyrics are
  not cached, so they can be added later).
- **Real audio analysis** (optional, needs `MUSIC_DIR`): loudness/energy,
  brightness and dynamic range always via ffmpeg; BPM and musical key with the
  `audio` extra (librosa), guarded to 40..220 BPM. Measured energy overrides
  the LLM estimate. Resolves Navidrome absolute paths (`/music/...`) against
  `MUSIC_DIR` without allowing root escapes.
- **Facet index and hard filters**: language (ISO 639-1), country (ISO 3166-1),
  decade, genre include/exclude, energy range, instrumental, ballad and
  concept-album. Filtering matches at track, album or artist level with query
  inheritance.
- **Controlled vocabulary** (`vocab.py`): normalizes languages, countries,
  decades (`80s`, `1985`, `late 90s`) and energy (`alta`, 8/10, 0.8) so "música
  en español" becomes a structured filter instead of guesswork.
- **Generic value filtering** (`generic.py`): placeholders like `Unknown`,
  `Various Artists`, `N/A`, `Track 01` are never treated as data. Exact
  normalized matching avoids false positives (`Unknown Mortal Orchestra` is a
  real band).
- **Non-destructive merge**: re-enrichment merges by field — lists union,
  scalars only replaced by valid values, measured fields (`language_source:
  lyrics`, `energy_source: audio`) are protected from the LLM. Descriptions and
  confidences are never erased by an empty run.
- **`needs_janitor` marking**: generic artists/albums are not sent to the LLM
  (it would hallucinate); they are flagged and counted in the dashboard so
  Module A fixes them.
- **`/facets` page**: browse the controlled vocabulary per entity with counts.
- **Facet badges** in the playlist preview (genre/mood/theme) with a track
  table, filling missing details from the mirror.
- **CLI**: `bardo facets [--rebuild]`, `bardo enrich --audio`, `--no-lyrics`.
- **Hard data beats the LLM**: Navidrome genre tags, Last.fm crowd tags and
  lyrics language detection take precedence; the tag comes first when merging
  valid genres.
- **Tests**: 937 tests with 100% line and branch coverage, enforced via
  `fail_under = 100`.
- **CI**: lint (ruff), tests on Python 3.12/3.13 with all extras and a Docker
  build + healthcheck smoke test.
- **Docs**: `docs/SPEC.md` (founding design), README (EN/ES), CONTRIBUTING.

### Changed

- beets 2.14 compatibility: the Janitor plugin listens on
  `import_task_created` (the event that actually fires on AS-IS imports) and
  reads distances via `Match.distance`.
- `subsonicupdate` uses the `subsonic:` config section with `auth: token`
  (beets 2.14 renamed it).
- The Janitor auto-generates `beets-config/config.yaml` from the environment
  when missing, so the container works out of the box.
- Docker image installs the package non-editable and ships the `lyrics` extra.

### Fixed

- Incremental sync counted tracks per track instead of per album, re-downloading
  the whole library on every run.
- Albums without `artistId` produced a phantom `artist:` id.
- Guest artists on tracks did not create their artist row (broken reference).
- `validate_track_ids` did not deduplicate, allowing repeated tracks.
- Year/genre filters were dropped in retrieval fallbacks and ignored by vector
  search.
- `canonical_facets` ignored `references`, `for_fans_of` and plural
  `languages`/`countries` from the LLM.
- `expand_prompt` crashed on non-numeric `size`, `None` term lists and read
  `size=0` as the default.
- `extract_json` broke on braces inside strings (rewritten as a balanced
  scanner tolerant of markdown fences and trailing commas).
- Last.fm cache crashed on corrupt payloads and never refreshed them.
- `wav2flac` crashed on empty stderr; `convert_all` ignored
  `settings.ffmpeg_bin`.
- `filter_valid` crashed on non-iterables and iterated strings character by
  character.
- `apply_hard_genre_to_artist` treated a stray string as a character list.
- `inherited` track confidence was hardcoded to 0.5 and ignored the artist's
  `lyrical_themes`.
- librosa returns `bpm=0.0` for unpulsed signals, which was stored as valid.
- Web Janitor checkbox defaulted to dry-run, preventing a real import.
- CLI logging did not apply when handlers already existed (`basicConfig` now
  uses `force=True`).

## [0.1.0] - 2026-09-14

### Added

- Initial implementation of the full spec.
- **Module A — Janitor**: WAV→FLAC with safe backup, beets runner, own plugin
  with before/after JSONL diff ingested into SQLite, Navidrome rescan, dry-run
  and timid flows, web diff UI.
- **Module B — AI Playlists**: full Subsonic client, incremental library
  mirror, per-entity enrichment cards (artist/album/track) with a local LLM,
  synonym canonicalization, optional Last.fm, track inheritance, sqlite-vec +
  FTS5 hybrid recall with RRF, tool drill-down and a function-calling agent
  capped at 6 calls, anti-hallucination track id validation.
- **Web/CLI**: FastAPI + Jinja2 + HTMX + Pico CSS, bilingual ES/EN, no build
  step; `bardo` CLI (init, ping, sync, health, janitor, enrich, index,
  playlist, serve).
- **Deploy**: Docker image and compose with an optional host-network override
  for Ollama listening on localhost.

[Unreleased]: https://github.com/Cundalf/analyzer-bard-/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Cundalf/analyzer-bard-/releases/tag/v0.1.0
