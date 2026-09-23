<div align="center">

# 🎧 Bardo

**Arreglá la metadata de tu biblioteca y armá playlists temáticas con IA — 100% local, gratis y portable.**

[English](README.md) · [Español](README.es.md)

![status](https://img.shields.io/badge/estado-alpha-orange)
![python](https://img.shields.io/badge/python-3.12%2B-blue)
![license](https://img.shields.io/badge/licencia-MIT-green)
![local](https://img.shields.io/badge/IA-Ollama-purple)

</div>

---

## ¿Qué es Bardo?

Bardo es un servidor web self-hosted que convive con tu
[Navidrome](https://www.navidrome.org/) (o cualquier servidor
**Subsonic-compatible**) y hace dos cosas:

1. **Módulo A — "Janitor":** corrige la metadata de la biblioteca
   (`[Unknown artist]`, `[Unknown album]`, años mal) usando [beets](https://beets.io/)
   + fingerprinting acústico (Chromaprint → AcoustID → MusicBrainz).
   **Requiere acceso a disco** a la misma carpeta que sirve Navidrome.
2. **Módulo B — "Playlists IA":** arma playlists temáticas por prompt en
   lenguaje natural ("celebración de taberna", "Señor de los Anillos") con un
   pipeline de **agentic RAG**. **Pura Subsonic API — sin disco.**

Todo corre en un contenedor. **Ollama** para toda la IA (chat + embeddings),
**sqlite-vec** para vectores, y la UI es **FastAPI + HTMX + Pico CSS**.
**$0 de servicios externos.**

## ¿Por qué no alcanza SUB/WAVE?

[SUB/WAVE](https://www.getsubwave.com) (y herramientas similares) mezclan por
**similitud sonora**. No pueden seleccionar temas por **conocimiento del mundo**
ni aceptar prompts libres como "música de taberna fantástica". Bardo es
complementario: el Janitor arregla la metadata que hace que el mixing suene
bien, y las Playlists IA curan por significado.

## Arquitectura

```
┌──────────────────────────────────────────────────────────┐
│  BARDO (FastAPI + Jinja2 + HTMX + Pico CSS)              │
│  un contenedor, sin build step                           │
│  ┌──────────────┐          ┌───────────────────────┐     │
│  │ Módulo A     │          │ Módulo B              │     │
│  │ Janitor      │          │ Playlists IA          │     │
│  │ (beets)      │          │ (agentic RAG)         │     │
│  └──────┬───────┘          └────┬─────────────┬────┘     │
└─────────┼───────────────────────┼─────────────┼──────────┘
          │                       │             │
     [DISCO]                  [Subsonic]     [Ollama]
     bind mount = la         Navidrome/     chat + embed
     misma carpeta que       gonic/LMS      (gemma4 + nomic)
     sirve Navidrome              │
          │                  [sqlite-vec]─────┘
          ▼                       │
   beets escribe tags ── rescan ──► Navidrome ──► SUB/WAVE
   (subsonicupdate)   (startScan)   (API)      (lee la lib)
```

**Cadena de la verdad:** `archivos → beets (arregla tags) → Navidrome (rescan)
→ Subsonic API → Bardo Módulo B`.

**Regla de oro:** el Módulo B hereda la calidad del Módulo A. Si los tags son
`[Unknown]`, las fichas salen basura → playlists basura. **Correr A antes que B.**

## Quickstart (Docker)

Requisitos:

- Docker + Docker Compose
- [Ollama](https://ollama.com) corriendo y accesible (local o en otro host)
- Un Navidrome (o servidor Subsonic-compatible)
- Una API key gratis de [AcoustID](https://acoustid.org/new-application) (para el Janitor)

```bash
git clone https://github.com/cundalf/bardo.git
cd bardo
cp .env.example .env
# editá .env: SUBSONIC_*, OLLAMA_*, ACOUSTID_KEY, MUSIC_PATH...
docker compose up -d --build
```

Abrí <http://localhost:8080>.

```bash
# Bajá los modelos que Bardo necesita (una vez)
ollama pull nomic-embed-text:v1.5
ollama pull gemma4:cloud     # o cualquier modelo con tool calling, ej. gemma4:12b
```

> **Ollama en Linux:** la instalación systemd por defecto escucha sólo en
> `127.0.0.1`, así que el contenedor no lo alcanza por red bridge.
> Usá el override de red host:
>
> ```bash
> docker compose -f docker-compose.yml -f docker-compose.host.yml up -d
> ```
>
> Alternativa: exponé Ollama en la LAN (`OLLAMA_HOST=0.0.0.0` en la unit de
> systemd) y poné `BARDO_OLLAMA_URL=http://<tu-ip>:11434` en `.env`.

## Quickstart (local, sin Docker)

```bash
# Python 3.12+
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[janitor,lyrics]"   # extras: ffmpeg + fpcalc en PATH; agregá "audio" para BPM/key
cp .env.example .env

bardo init          # crea la DB SQLite
bardo ping          # verifica la conexión Subsonic
bardo sync          # espeja artistas/álbumes/tracks de Navidrome
bardo serve         # UI en http://localhost:8080
```

## CLI

| Comando | Qué hace |
|---|---|
| `bardo init` | Crea la DB y los directorios de datos |
| `bardo ping` | Verifica la conexión Subsonic |
| `bardo sync` | Espeja la biblioteca de Navidrome a SQLite |
| `bardo health` | Reporte de salud (sin año, artista desconocido…) |
| `bardo janitor [-p] [-t] [--skip-wav] [ruta]` | Módulo A: WAV→FLAC + beets import + rescan |
| `bardo enrich [--limit N] [--force]` | Módulo B: genera fichas de artista/álbum/canción |
| `bardo index [--force]` | Construye embeddings en `sqlite-vec` + FTS |
| `bardo facets [--rebuild]` | Inspecciona el índice de facetas normalizadas (idiomas, países, décadas…) |
| `bardo playlist "prompt" [--save] [--no-agent]` | Genera una playlist desde un prompt |
| `bardo serve [--host] [--port] [--reload]` | Levanta la UI web |

### Flujo seguro del Janitor

1. **Backup** de la biblioteca (o probá sobre una copia de ~10 temas).
2. **Preview:** `bardo janitor -p /music` (pretend — no escribe nada).
3. **Interactivo:** `bardo janitor -t /music` (timid — pregunta en cada match).
4. Import real + rescan automático de Navidrome.

El log de corrida es un JSONL por archivo (`old_tags`, `new_tags`,
`match_score`, `album_id`, `status`) que la UI muestra como diff antes/después.

## Cómo funcionan las playlists IA

```
prompt ──► [1] expansión (LLM → términos canónicos + asociaciones)
        ──► [2] recall híbrido (sqlite-vec + FTS5 + tags) → ~50 candidatos
        ──► [3] rerank/selección con LLM (sólo sobre los candidatos)
        ──► [4] validar track_ids reales contra Navidrome (anti-alucinación)
        ──► [5] create_playlist → aparece en Navidrome (y en SUB/WAVE)
```

- **Fichas:** una por artista/álbum/canción, con vocabulario controlado por
  facetas (`genres`, `moods`, `references`, …) más una `description` en prosa
  — la descripción es la mejor señal para el embedding.
- **Filtros duros:** idioma (ISO 639-1), país (ISO 3166-1), década, género
  (incluir/excluir), rango de energía, instrumental/balada/álbum conceptual.
  "Música en español", "bandas argentinas", "de los 80", "música tranquila" se
  resuelven a filtros estructurados, no a adivinanzas.
- **Dato duro pisa al LLM:** los tags de género de Navidrome, los crowd tags de
  Last.fm y la detección de idioma offline (py3langid, desde la letra) tienen
  prioridad cuando contradicen al modelo.
- **Nada válido se pierde:** el re-enriquecimiento fusiona por campo (las listas
  se unen, los escalares sólo se reemplazan por valores válidos). Los
  placeholders genéricos ("Unknown", "Various Artists", "Track 01", "N/A") no
  son datos y nunca pisan un valor real.
- **Sin LLM para basura:** `[Unknown Artist]` / `[Unknown Album]` se marcan
  `needs_janitor` en vez de mandarse al modelo (que alucinaría); el dashboard
  los cuenta para que el Módulo A los arregle.
- **Análisis acústico real opcional:** loudness/energía, brillo y rango dinámico
  siempre vía ffmpeg; BPM y tonalidad con el extra `audio` (librosa).
  Deshabilitado por defecto (`ANALYZE_AUDIO=true`, requiere `MUSIC_DIR`).
- **Herencia completa:** las canciones heredan idioma, país, género,
  instrumentación, estilo vocal y energía desde álbum → artista.
- **Artista/álbum** se enriquecen con el LLM. Las **canciones** heredan de su
  álbum/artista, opcionalmente enriquecidas por letras/tags (Last.fm) — jamás
  se mandan 5000 canciones al LLM una por una.
- **Canonicalización:** los sinónimos se normalizan vía la tabla `synonyms`
  (`party/joda/fiesta → fiesta`).
- **Incremental:** cada ficha guarda un `content_hash`; las entidades que no
  cambiaron no se re-enriquecen.
- **Regla dura:** máximo ~6 tool calls por pedido. Pre-filtro determinístico →
  agente corto.

## Configuración

Todo se configura en `.env` (ver `.env.example`). Las clave:

| Variable | Default | Notas |
|---|---|---|
| `SUBSONIC_URL` | `http://localhost:4533` | URL base de Navidrome |
| `OLLAMA_CHAT_MODEL` | `gemma4:cloud` | debe soportar tool calling |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text:v1.5` | 768 dims |
| `EMBED_DIM` | `768` | debe coincidir con el modelo de embeddings |
| `JANITOR_ENABLED` | `false` | habilita el Módulo A (necesita disco) |
| `MUSIC_DIR` | — | misma carpeta que sirve Navidrome |
| `ENABLE_LASTFM` | `false` | crowd tags vía `LASTFM_API_KEY` |
| `ENRICH_TRACKS_LLM` | `false` | enriquecimiento por canción con LLM (caro) |
| `DETECT_LANGUAGE` | `true` | detección de idioma por letra offline (extra `lyrics`) |
| `ANALYZE_AUDIO` | `false` | features de audio reales; requiere `MUSIC_DIR` (extra `audio`) |
| `WEB_SEARCH_ENABLED` | `false` | validación factual quirúrgica |

También podés pisar settings en runtime desde **Configuración** en la UI
(se guardan en SQLite y tienen prioridad sobre `.env`).

## Tests

```bash
pip install -e ".[janitor,dev]"
make test          # 564 tests
make cover         # exige 100% de líneas y ramas
make lint          # pyflakes
```

La suite exige **100% de cobertura de líneas y ramas** (`fail_under = 100` en
`pyproject.toml`). Cada feature y camino de error tiene tests: cliente Subsonic
(todos los endpoints, reintentos, payloads malformados), Ollama (recuperación
de JSON, reintentos, tool calling), retrieval (filtros, fallbacks, validación
anti-alucinación), enriquecimiento (fallos del LLM, herencia, staleness),
Janitor (WAV→FLAC, runner de beets, log antes/después), UI web y CLI.

## Estructura del repo

```
bardo/
├── app/
│   ├── main.py            # rutas FastAPI + UI
│   ├── config.py          # env + overrides de DB
│   ├── db.py              # SQLite + sqlite-vec + FTS5
│   ├── subsonic.py        # cliente Subsonic (lectura/escritura/scan)
│   ├── ollama.py          # chat, embeddings, tool calling
│   ├── cli.py             # comando `bardo`
│   ├── janitor/           # Módulo A: wav2flac, runner beets, report, plugin
│   ├── enrich/            # prompts, fichas artista/álbum/canción, Last.fm
│   ├── index/             # embeddings → vec_fichas
│   ├── agent/             # tools, retrieval, loop ReAct
│   └── web/               # templates + static
├── beets-config/          # config de beets (ejemplo; el container la autogenera)
├── Dockerfile
└── docker-compose.yml
```

## FAQ

**¿Bardo reproduce música?** No. Eso lo hace Navidrome. Bardo sólo gestiona
metadata y playlists (vía Subsonic API).

**¿Modifica mis archivos?** Sólo el Módulo A, sólo a través de beets, y nunca
mueve/renombra archivos (`move: no`) para que los IDs de Navidrome queden
estables. Usá primero la simulación (pretend).

**¿Sirve con gonic/LMS/Airsonic?** Cualquier servidor Subsonic-compatible que
implemente `createPlaylist` y `startScan` debería funcionar.

**¿Necesito GPU?** No. Ollama en CPU funciona; una GPU acelera mucho el
enriquecimiento. También se pueden usar modelos cloud vía Ollama, ej. `gemma4:cloud`.

**¿Mis datos se envían a algún lado?** Sólo lo que habilites: AcoustID/MusicBrainz
(fingerprints, gratis), Last.fm (tags opcionales) y tu host de Ollama. Sin telemetría.

## Licencia

MIT. Ver [LICENSE](LICENSE).
