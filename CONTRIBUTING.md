# Contributing to Bardo

Thanks for wanting to help. Bardo is a small, opinionated project: a
self-hosted music library janitor and AI playlist builder for Navidrome /
Subsonic servers. This guide covers the practical bits.

## Development setup

Requirements: Python 3.12+ and (optionally) Docker for the container path.

```bash
git clone https://github.com/Cundalf/analyzer-bard-.git
cd analyzer-bard-
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[janitor,lyrics,audio,dev]"
```

System packages used by the extras:

- `ffmpeg` — WAV→FLAC conversion and audio decoding.
- `libchromaprint-tools` (`fpcalc`) — acoustic fingerprinting in the Janitor.

Run the checks:

```bash
make test     # pytest
make cover    # pytest --cov (requires 100% line + branch coverage)
make lint     # ruff check + format check
make serve    # web UI on http://localhost:8080
```

## Project layout

```
app/
├── agent/     # retrieval, tools, function-calling loop, playlist runner
├── enrich/    # prompts, per-entity cards, canonicalization, vocab, merge
├── index/     # embeddings into sqlite-vec + FTS
├── janitor/   # Module A: wav2flac, beets runner, before/after report
├── web/       # Jinja templates, static assets, i18n
├── db.py      # SQLite schema, facet index, vector helpers
├── subsonic.py, ollama.py, config.py, main.py, cli.py
tests/         # mirrors the modules; conftest has shared fakes
docs/SPEC.md   # founding design document
```

## Quality bar

- **100% line and branch coverage is enforced** (`fail_under = 100` in
  `pyproject.toml`). A change that adds untested branches will fail CI.
- `ruff check` and `ruff format --check` must pass.
- Tests use fakes (`FakeOllama`, `FakeSubsonic`) — never hit the network, a real
  Navidrome or a real Ollama in tests.
- Prefer small, single-purpose modules. Existing patterns:
  - `app/enrich/generic.py` for value validation
  - `app/enrich/merge.py` for non-destructive merging
  - `app/enrich/vocab.py` for controlled vocabulary

## Principles worth respecting

1. **Module B inherits Module A's quality.** If tags are junk, cards are junk.
2. **Hard data beats the LLM.** Measured facts (Navidrome tags, Last.fm tags,
   lyrics language, audio features) win over model guesses.
3. **Nothing valid is ever lost.** Merge by field; never erase a real value
   with an empty or generic one.
4. **No LLM on junk.** Generic entities are marked and left for the Janitor
   instead of being sent to a model that would hallucinate.
5. **Anti-hallucination is non-negotiable.** Every track id is validated against
   the real library mirror before a playlist is created.
6. **Stay within the spec's non-goals** (`docs/SPEC.md` §14): Bardo does not
   stream music, does not replace Navidrome, does not write custom tags, and is
   single-user.

## Adding a feature

1. Add tests first where practical; they pin the expected behavior.
2. Keep the public surface small: new settings go in `app/config.py` with an
   entry in `.env.example` and the compose file.
3. If it touches enrichment, route it through `merge_hard_facets` /
   `merge_ficha_facets` so precedence rules hold.
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Run `make cover` and `make lint` before opening a PR.

## Commit style

Short imperative subject, body explaining the why when it is not obvious:

```
Fix incremental sync counting tracks per album

The old query grouped by track navidrome_id, so every album looked
missing on each run and the whole library was re-downloaded.
```

## Reporting issues

Include: what you expected, what happened, the relevant logs (with secrets
redacted), your Navidrome/Subsonic server and version, your Ollama models, and
whether the Janitor (disk access) is enabled.

**Never paste API keys, passwords or `.env` contents.**

## License

By contributing you agree your work is released under the MIT license
(`LICENSE`).
