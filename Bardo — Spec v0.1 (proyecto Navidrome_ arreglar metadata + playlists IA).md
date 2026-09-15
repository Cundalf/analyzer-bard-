# 🎧 Bardo — Spec v0.1

> **Codename tentativo: "Bardo"** (cambiable). Documento maestro para pasar a una IA y arrancar el vibe-coding.
> **Leer de arriba a abajo. No inventar features que no estén acá. Respetar la sección 14 (No-alcance).**

---

## 0. TL;DR

**Bardo** es un servidor web self-hosted **open source** que trabaja sobre una biblioteca musical servida por **Navidrome** (o cualquier servidor **Subsonic-compatible**). Hace **dos cosas**:

1. **Módulo A — "Janitor" (arreglar la biblioteca):** corrige metadata (`[Unknown artist]`, `[Unknown album]`, años mal) usando **beets + fingerprinting acústico**. **Requiere acceso a disco.**
2. **Módulo B — "Playlists IA":** arma playlists temáticas por **prompt en lenguaje natural** ("celebración de taberna", "Señor de los Anillos"). **Pura Subsonic API, sin disco.**

Un solo contenedor. **Ollama** para toda la IA (chat + embeddings). **sqlite-vec** para vectores. **FastAPI + HTMX + Pico CSS** para la UI. $0 de servicios externos.

---

## 1. Visión y problema

**Problema:** el usuario tiene una biblioteca con muchos temas como `[Unknown artist]`/`[Unknown album]` y años incorrectos (rips de YouTube + archivos viejos). Y quiere playlists **temáticas por prompt**, no solo por similitud sonora.

**Por qué no basta SUB/WAVE:** SUB/WAVE mezcla por **similitud sonora** (features de audio). No hace **selección temática por conocimiento del mundo** ni acepta prompts libres. Son complementarios.

**Objetivo:** arreglar la metadata **y** construir un armador de playlists temáticas, todo **local, gratis, portable y lindo**.

---

## 2. Glosario y definiciones (NO confundir)

| Término | Definición |
|---|---|
| **Fingerprinting acústico** | Huella del audio → consulta **AcoustID** → identidad en **MusicBrainz**. Devuelve **quién/cuál es** (artista, título, álbum, año). Lo hace **beets/Chromaprint**. |
| **Análisis acústico** | Mide features del sonido (**BPM, tonalidad, energía, embedding**). Sirve para **similitud**, NO identifica. Lo hace **SUB/WAVE**. |
| **Metadata factual** | artista, título, álbum, año. Vive **en el archivo**. La escribe **beets**. |
| **Metadata derivada / "ficha"** | temas, moods, referencias (`Tolkien`, `taberna`). La genera el **LLM**. Vive en la **DB propia de Bardo** (NO en el archivo). |
| **Ficha** | Tarjeta de enriquecimiento por entidad. Hay fichas de **artista**, **álbum** y **canción**. Estructura por **facetas** + una `description` en prosa. |
| **Recall (narrowing)** | Reducir 5000 temas → ~50 candidatos. Es la capa de **búsqueda** (retrieval / RAG). |
| **Drill-down (expansión)** | Dado un artista → sus álbumes → sus temas. Son los **tools**. |
| **Agentic RAG** | LLM orquestando tools, donde el **retrieval es un tool más** (el de recall). |
| **Canonicalización** | Normalizar sinónimos a un término único (`fiesta/joda/party → party`). |
| **RAG (recordar)** | **El RAG no sabe nada por sí solo.** El conocimiento vive en las **fichas**; el vector es solo el **buscador** sobre ese texto. |

---

## 3. Arquitectura general

```
        ┌──────────────────────────────────────────────────────┐
        │   BARDO  (FastAPI + Jinja2 + HTMX + Pico CSS)         │
        │   un contenedor, sin build step                       │
        │  ┌───────────────┐         ┌───────────────────────┐  │
        │  │ Módulo A      │         │ Módulo B              │  │
        │  │ Janitor       │         │ Playlists IA          │  │
        │  │ (beets)       │         │ (agentic RAG)         │  │
        │  └──────┬────────┘         └─────┬──────────┬──────┘  │
        └─────────┼────────────────────────┼──────────┼─────────┘
                  │                        │          │
            [DISCO]                  [Subsonic]   [Ollama]
         bind mount = la            Navidrome/   chat + embed
         misma carpeta que          gonic/LMS/   (nomic-embed +
         sirve Navidrome            Airsonic     gemma)
                  │                        │          │
                  │                   [sqlite-vec]────┘
                  ▼                        │
        beets escribe tags ── rescan ──► Navidrome ──► SUB/WAVE
        (subsonicupdate)     (startScan)   (API)      (lee la lib)
```

**Cadena de la verdad:**
`archivos → beets (arregla tags) → Navidrome (rescan) → Subsonic API → Bardo Módulo B`

**Regla de oro:** el Módulo B **hereda la calidad** del Módulo A. Si los tags son `[Unknown]`, las fichas salen basura → playlists basura. **Correr A antes que B.**

---

## 4. Stack técnico (DECISIONES CERRADAS)

| Capa | Elección | Por qué |
|---|---|---|
| Lenguaje | **Python 3.12** | Mismo lenguaje que beets; ecosistema IA |
| Web backend | **FastAPI** + Jinja2 | Liviano, server-rendered |
| Interactividad | **HTMX** (+ Alpine.js si hace falta) | Sin npm, sin build step, sin React |
| CSS | **Pico CSS** | Classless, ~10 KB, moderno, responsive. Alternativas: Water.css / Simple.css; Tailwind+DaisyUI solo si se acepta build step |
| Vectores | **sqlite-vec** (NO pgvector) | Un archivo, cero servidor, portable. Brute-force alcanza para miles de vectores |
| IA | **Ollama** (embeddings + chat) | Ya disponible; mismo endpoint |
| Embeddings | **nomic-embed-text:v1.5** (768 dims) | Liviano, CPU-friendly. ⚠️ es English-centric: si el matching en español falla, usar **bge-m3** (multilingual). Configurable |
| Chat | **gemma** (vía Ollama) | Modelo del ecosistema del usuario |
| Metadata | **beets** (Módulo A) | Fingerprint → AcoustID → MusicBrainz, escritura de tags |
| Datos | **SQLite** | Un archivo, backup trivial |
| Deploy | **Docker / docker-compose** | Un contenedor (+ el de beets) |

---

## 5. Módulo A — Janitor (beets)

**Requiere disco sí o sí** (la misma carpeta/bind mount que sirve Navidrome). Puede correr **headless/CLI**, incluso antes de que exista la app web.

### 5.1 Pipeline
1. Detectar archivos **WAV** → convertir a **FLAC** (`ffmpeg -i x.wav -c:a flac x.flac`). Lossless→lossless, sin pérdida. (El WAV casi no soporta tags → causa de muchos `[Unknown]`.)
2. Correr **beets import** (fingerprint → AcoustID → MusicBrainz → escribir tags + carátula).
3. Disparar **rescan de Navidrome** (plugin `subsonicupdate` o endpoint `startScan`).
4. Registrar **log JSONL** (antes/después por archivo).

### 5.2 Dockerfile de beets
```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libchromaprint-tools \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir beets pyacoustid requests
ENTRYPOINT ["beet"]
```
> `libchromaprint-tools` provee `fpcalc`. NO confiar en imágenes de terceros que a veces no lo traen.

### 5.3 config de beets (`beets-config/config.yaml`)
```yaml
directory: /music
library: /config/library.db
import:
  write: yes
  move: no        # CRÍTICO: no mover/renombrar → mantiene estables los IDs de Navidrome
  copy: no
  autotag: yes
  detail: yes
plugins: chroma fromfilename fetchart mbsync subsonicupdate lyrics
acoustid:
  apikey: TU_API_KEY_DE_ACOUSTID   # gratis: acoustid.org
subsonicupdate:
  url: http://navidrome:4533
  user: TU_USER
  password: TU_PASS
```
> `move: no` es **crítico**: si beets mueve archivos, Navidrome les cambia el ID y las fichas del Módulo B se desvinculan.

### 5.4 Flujo de uso seguro
1. Backup (o probar sobre una copia de ~10 temas).
2. Preview sin escribir: `beet import -p /music` (**pretend**).
3. Import interactivo: `beet import -t /music` (**timid**: pregunta incluso en matches fuertes).
4. Rescan Navidrome.

### 5.5 Log / antes-después
Guardar por archivo: `{ file, old_tags, new_tags, match_score, album_id, status }` en JSONL (`runs/*.jsonl`) e ingestarlo a la tabla `import_log`. La UI lo muestra como **diff**.

---

## 6. Módulo B — Playlists IA (agentic RAG)

**Pura Subsonic API. NO requiere disco** (salvo features de audio opcionales, ver 6.7).

### 6.1 Cliente Subsonic
Endpoints a usar (auth: `u`, `t` = md5(pass+salt), `s` = salt, `v=1.16.1`, `c=bardo`, `f=json`):
- Lectura: `ping`, `getArtists`, `getArtist`, `getAlbum`, `getAlbumList2`, `search3`, `getPlaylists`, `getPlaylist`, `getCoverArt`, `stream`
- Escritura: `createPlaylist`, `updatePlaylist`, `deletePlaylist`, `star`/`unstar`
- Mantenimiento: `startScan`
> La Subsonic API es **solo lectura para tags**. Para arreglar tags → Módulo A (beets, disco). No intentar escribir tags por la API.

### 6.2 Enriquecimiento (fichas) — **OFFLINE, una vez, incremental**
- **Por ARTISTA (~500)** con LLM → facetas amplias + `description`.
- **Por ÁLBUM (~1500)** con LLM → concepto (clave en discos conceptuales).
- **Por CANCIÓN (5000+): NO con LLM de una.** Por **herencia del álbum** + tags de **Last.fm** + **letras** + (opcional) **features de audio**.
- Fuentes: LLM (world knowledge) + **Last.fm API** (crowd tags) + **web search** (validación quirúrgica, no masiva).
- **Incremental:** guardar `content_hash` por entidad; si no cambió, no re-enriquecer.

### 6.3 Esquema de facetas (vocabulario controlado)
> NO "10 keywords sueltas". Sí **esquema por facetas** + **canonicalización** de sinónimos. Guardar **tags estructurados** (match exacto/híbrido) **+** una `description` en prosa (mejor señal para embeddings).

**Artista:**
```json
{
  "name": "", "genres": [], "subgenres": [], "country": "", "era": "",
  "language": "", "instrumentation": [], "vocal_style": "",
  "lyrical_themes": [], "moods": [], "references": [], "for_fans_of": [],
  "energy": 0.0, "description": "", "confidence": 0.0
}
```
**Álbum:**
```json
{
  "artist": "", "album": "", "year": 0, "is_concept_album": false,
  "concept": "", "themes": [], "moods": [], "references": [],
  "energy": 0.0, "description": "", "confidence": 0.0
}
```
**Canción:**
```json
{
  "title": "", "moods": [], "themes": [], "energy": 0.0,
  "is_ballad": false, "is_instrumental": false,
  "description": "", "confidence": 0.0
}
```

### 6.4 Índice vectorial
- Embeddear la **`description` + tags canónicos** de cada ficha. Una fila por entidad, con `entity_type`.
- `sqlite-vec` (brute-force; para miles de vectores es instantáneo). Considerar quantización binaria solo si >100k vectores.

### 6.5 Tools del agente (function calling)
| Tool | Capa | Motor |
|---|---|---|
| `search_candidates(query, filters, limit)` | **Recall** | Retrieval híbrido (vector sqlite-vec + full-text + tags) |
| `list_artists(query)` | Recall/lista | SQL |
| `list_albums(artist)` | Drill-down | Subsonic API |
| `list_tracks(album)` | Drill-down | Subsonic API |
| `web_search(query)` | Validación | Búsqueda web (solo para confirmar hipótesis fuertes) |
| `create_playlist(name, track_ids)` | Acción | Subsonic API (`createPlaylist`) |

### 6.6 Flujo de consulta
```
prompt ─► [1] expansión (LLM → términos canónicos + asociaciones)
       ─► [2] search_candidates (retrieval HÍBRIDO) → ~50 candidatos
       ─► [3] rerank/selección con LLM (sobre los candidatos, NO toda la lib)
       ─► [4] validar track_ids reales contra Navidrome (anti-alucinación)
       ─► [5] create_playlist → aparece en Navidrome → y en SUB/WAVE
```
Regla de eficiencia: **máximo ~6 tool calls**. Pre-filtro determinístico → agente corto.

### 6.7 Features de audio (OPCIONAL / futuro)
Si se quiere BPM/tonalidad en canciones, eso **también pide disco** o usar `stream`. **Hacerlo opcional y desacoplado**: si hay `MUSIC_DIR`, se calcula; si no, el Módulo B sigue funcionando solo con metadata + LLM + Last.fm.

---

## 7. PROMPTS (copiar tal cual)

### 7.A — Enriquecimiento de ARTISTA
```
SYSTEM:
Sos un curador musical y musicólogo con conocimiento enciclopédico de bandas de
todos los géneros. Describís a un ARTISTA para armar playlists temáticas.
Devolvés SIEMPRE JSON válido, sin markdown, sin texto extra.
Reglas:
- Géneros: usá el nombre canónico estándar (usualmente en inglés: "folk metal", "power metal").
- Temas/moods/referencias: en español neutro.
- No inventes. Si dudás, bajá "confidence".
- "description": 1 oración de 15-30 palabras, natural y rica en significado (se usa para embedding).

INPUT:
Artista: {{artist_name}}
Géneros que ya figuran en la biblioteca: {{genres_from_library}}
Álbumes en la biblioteca: {{albums_list}}

OUTPUT (JSON):
{ "name": "", "genres": [], "subgenres": [], "country": "", "era": "",
  "language": "", "instrumentation": [], "vocal_style": "", "lyrical_themes": [],
  "moods": [], "references": [], "for_fans_of": [], "energy": 0.0,
  "description": "", "confidence": 0.0 }
```

### 7.B — Enriquecimiento de ÁLBUM
```
SYSTEM:
Igual que antes, pero enfocado en el ÁLBUM. Prestá especial atención a discos
CONCEPTUALES (ej: "Nightfall in Middle-Earth" → Tolkien/Silmarillion).
Devolvés JSON válido, sin markdown.

INPUT:
Artista: {{artist}}
Álbum: {{album}}
Año: {{year}}
Géneros del artista: {{artist_genres}}

OUTPUT (JSON):
{ "artist": "", "album": "", "year": 0, "is_concept_album": false,
  "concept": "", "themes": [], "moods": [], "references": [],
  "energy": 0.0, "description": "", "confidence": 0.0 }
```

### 7.C — Facetas de CANCIÓN (ligero / herencia)
```
SYSTEM:
Describís una CANCIÓN a partir de datos ya existentes. No inventes si falta info.
Devolvés JSON válido, sin markdown.

INPUT:
Artista: {{artist}}
Álbum: {{album}}
Título: {{title}}
Concepto del álbum: {{album_concept}}
Tags del álbum: {{album_tags}}
Letra (si existe): {{lyrics}}

OUTPUT (JSON):
{ "title": "", "moods": [], "themes": [], "energy": 0.0,
  "is_ballad": false, "is_instrumental": false,
  "description": "", "confidence": 0.0 }
```

### 7.D — Expansión de consulta
```
SYSTEM:
Convertís un pedido en lenguaje natural a un objeto de búsqueda. Devolvés JSON, sin markdown.
Asociá términos difusos a un campo semántico ("taberna" → ["folk metal","celta","fiesta","drinking song"]).

INPUT:
"{{user_prompt}}"

OUTPUT (JSON):
{ "intent": "playlist", "canonical_terms": [], "expanded_terms": [],
  "moods": [], "reference": "", "filters": { "year_min": null, "year_max": null,
  "exclude_genres": [] }, "size": 30 }
```

### 7.E — Rerank / selección (anti-alucinación)
```
SYSTEM:
Te doy un pedido y una lista de CANDIDATOS REALES de la biblioteca (id, artista,
álbum, título, facetas). Elegí los que mejor encajan. Devolvés JSON con IDs EXACTOS
de la lista dada. PROHIBIDO inventar IDs o títulos.
INPUT:
Pedido: {{user_prompt}}
Candidatos: {{candidates_json}}
OUTPUT (JSON):
{ "playlist_name": "", "track_ids": [], "reasoning": "" }
```

### 7.F — System prompt del AGENTE (con tools)
```
SYSTEM:
Sos Bardo, arquitecto de playlists. Tenés tools. Objetivo: a partir del pedido del
usuario, armar una playlist con temas REALES de su biblioteca.
Reglas:
- Nunca inventes track IDs. Usá solo IDs devueltos por los tools.
- Flujo típico: search_candidates(pedido) → (si hace falta) list_albums/list_tracks →
  (opcional) web_search para validar un dato del mundo → create_playlist(ids).
- Máximo 6 tool calls. Sé eficiente. Si el pedido es ambiguo, asumí y aclaralo al final.
- Devolvé el nombre de la playlist y una justificación breve.
```

### 7.G — Validación web (opcional)
```
SYSTEM:
Verificás UN dato factual sobre música. Devolvés JSON {"answer": true|false, "evidence": ""}.
INPUT:
"¿El artista/álbum {{name}} está relacionado con el tema '{{theme}}'? Álbum: {{album}}"
```

---

## 8. Modelo de datos (SQLite)

```sql
-- Espejo de Navidrome
CREATE TABLE artists  (id TEXT PRIMARY KEY, navidrome_id TEXT UNIQUE, name TEXT);
CREATE TABLE albums   (id TEXT PRIMARY KEY, navidrome_id TEXT UNIQUE, artist_id TEXT, name TEXT, year INT);
CREATE TABLE tracks   (id TEXT PRIMARY KEY, navidrome_id TEXT UNIQUE, album_id TEXT, artist_id TEXT,
                       title TEXT, duration INT, path TEXT);

-- Fichas (artista | album | track)
CREATE TABLE fichas (
  entity_type TEXT, entity_id TEXT, facets TEXT /*JSON*/, description TEXT,
  confidence REAL, source TEXT /*llm|lastfm|inherited|web|audio*/,
  content_hash TEXT, updated_at TEXT,
  PRIMARY KEY (entity_type, entity_id)
);

-- Índice vectorial (sqlite-vec)
CREATE VIRTUAL TABLE vec_fichas USING vec0(
  entity_id TEXT, entity_type TEXT, embedding FLOAT[768]
);

-- Vocabulario controlado
CREATE TABLE synonyms (term TEXT PRIMARY KEY, canonical TEXT);

-- Logs / runs
CREATE TABLE runs (id INTEGER PRIMARY KEY, module TEXT, started_at TEXT, finished_at TEXT, stats TEXT);
CREATE TABLE import_log (id INTEGER PRIMARY KEY, run_id INT, file TEXT,
                         old_tags TEXT, new_tags TEXT, match_score REAL, status TEXT);
```

---

## 9. Endpoints web (internos)

- `GET  /` → dashboard (salud de biblioteca + acceso a playlists)
- `GET  /library` → tabla **antes/después** + salud (X sin artista, Y sin año…)
- `POST /library/fix` → dispara Janitor (beets) en background
- `GET  /library/runs/{id}` → log/diff de una corrida
- `GET  /playlists` → builder (prompt + preview)
- `POST /playlists/generate` `{prompt}` → devuelve preview (nombre + temas + razonamiento)
- `POST /playlists/save` `{name, track_ids}` → crea en Navidrome vía Subsonic
- `GET  /settings` → config (Ollama, Subsonic, Last.fm key, toggles)

---

## 10. UI (Pico CSS, server-rendered)

Dos pestañas:
1. **📚 Biblioteca**: salud (semáforo que baja), tabla antes/después, botón "Arreglar", log de corridas.
2. **🎵 Playlists IA**: caja de prompt, botón "Generar" → **preview** (temas + por qué + score) → "Guardar en Navidrome".

Principios: HTML semántico + Pico (sin clases), HTMX para las acciones, **sin build step**, responsive.

---

## 11. Docker / despliegue

`docker-compose.yml` (servicio de la app; el de beets es aparte/opcional):
```yaml
services:
  bardo:
    build: .
    environment:
      - SUBSONIC_URL=http://navidrome:4533
      - SUBSONIC_USER=${SUBSONIC_USER}
      - SUBSONIC_PASS=${SUBSONIC_PASS}
      - OLLAMA_URL=http://ollama:11434
      - OLLAMA_CHAT_MODEL=gemma
      - OLLAMA_EMBED_MODEL=nomic-embed-text:v1.5
      - LASTFM_API_KEY=${LASTFM_API_KEY}
      - MUSIC_DIR=/music           # OPCIONAL: solo si se habilitan features de audio
      - JANITOR_ENABLED=true       # habilita Módulo A (requiere el bind mount)
    volumes:
      - ./data:/data               # sqlite + logs
      - ${MUSIC_PATH:-./music}:/music   # MISMA carpeta que Navidrome (solo para Módulo A)
    ports:
      - "8080:8080"
    restart: unless-stopped

  # Opcional: servicio beets (usar solo si JANITOR_ENABLED)
  beets:
    build: ./beets
    environment: [ "BEETSDIR=/config" ]
    volumes:
      - ./music:/music
      - ./beets-config:/config
    user: "1000:1000"
```

---

## 12. Estructura del repo

```
bardo/
├── docker-compose.yml
├── Dockerfile
├── README.md
├── LICENSE                 # MIT
├── .env.example
├── app/
│   ├── main.py             # FastAPI + rutas
│   ├── config.py           # env + settings
│   ├── db.py               # sqlite + sqlite-vec (load extension)
│   ├── subsonic.py         # cliente Subsonic (read/write/scan)
│   ├── ollama.py           # chat + embeddings
│   ├── janitor/
│   │   ├── runner.py       # orquesta beets
│   │   ├── wav2flac.py     # convierte WAV → FLAC
│   │   └── report.py       # JSONL → diff antes/después
│   ├── enrich/
│   │   ├── prompts.py
│   │   ├── artist.py
│   │   ├── album.py
│   │   ├── track.py
│   │   └── canonicalize.py # sinónimos → vocabulario controlado
│   ├── index/
│   │   └── build.py        # embeddings → vec_fichas
│   ├── agent/
│   │   ├── tools.py        # definiciones + implementaciones
│   │   ├── loop.py         # bucle ReAct / function calling
│   │   └── prompts.py
│   └── web/
│       ├── templates/
│       └── static/
└── beets/
    ├── Dockerfile
    └── config.yaml
```

---

## 13. Roadmap por fases (construir en este orden)

- **Fase 0 — Andamio:** repo, Docker, config, cliente Subsonic (`ping` OK), conexión Ollama.
- **Fase 1 — MVP Janitor (CLI):** beets + WAV→FLAC + log JSONL. Ya escuchás música con tags arreglados.
- **Fase 2 — UI Biblioteca:** antes/después + salud + botón "Arreglar".
- **Fase 3 — MVP Playlists v1 (SIN vectores):** prompt → tags/full-text → rerank LLM → `createPlaylist`. Sin RAG todavía.
- **Fase 4 — Enriquecimiento + índice:** fichas de artista y álbum (LLM) + Last.fm → embeddings → `sqlite-vec`.
- **Fase 5 — Agente completo:** tools + drill-down + web search + fichas de canción (herencia).
- **Fase 6 — Pulido + launch OSS:** README, screenshots, demo, GIF.

> Regla: **cada fase debe funcionar sola**. No adelantar features de fases futuras.

---

## 14. NO-ALCANCE (anti over-engineering)

**Bardo NO es / NO hace:**
- ❌ Reproductor ni streaming propio (eso es Navidrome).
- ❌ Reemplazar a Navidrome ni escanear la biblioteca por su cuenta para metadata.
- ❌ Editar tags de archivos desde la app (eso es Módulo A con beets).
- ❌ Multiusuario / login complejo (es **single-user**; auth simple o ninguna).
- ❌ App móvil nativa.
- ❌ Entrenar o fine-tunear modelos.
- ❌ Soportar servidores no-Subsonic.
- ❌ Escribir tags custom en archivos (queda como **futuro/opcional**).
- ❌ Web search masivo (solo quirúrgico para validar).
- ❌ Enriquecer 5000 canciones con LLM de una (usar herencia/tags/audio).

---

## 15. Riesgos y trampas

| Riesgo | Mitigación |
|---|---|
| Fingerprint falla en lives/remixes/oscuras | Fallback: `fromfilename` + SongRec (Shazam CLI) + `beet modify` |
| Rate limit AcoustID/MusicBrainz | beets ya lo respeta; no paralelizar de más |
| LLM alucina temas | **Validar IDs** contra la biblioteca real antes de crear la playlist |
| `nomic-embed-text` es English-centric | Probar; si el español matchea mal → `bge-m3` (configurable) |
| Costo de enriquecer canciones | Enriquecer artista+álbum con LLM; canciones por herencia/tags/audio |
| IDs de Navidrome cambian | `move: no` en beets |
| Contexto del LLM se llena | Chunking por artista + retrieval; NUNCA toda la lib en contexto |
| Sinónimos inconsistentes | Canonicalizar con tabla `synonyms` |
| Idioma mezclado (es/en) | Definir idioma consistente; géneros en canónico (inglés) |

---

## 16. Decisiones abiertas (a definir en la app)

- Nombre final del proyecto (¿"Bardo"?).
- ¿Escribir fichas también como tags en el archivo? (portabilidad — futuro).
- Modelo de embeddings final (`nomic-embed-text` vs `bge-m3`).
- ¿Auth sí/no?
- ¿Soporte de múltiples bibliotecas/servidores?
- ¿Integración Last.fm en el MVP o en fase posterior?

---

## 17. Licencia y OSS

- **MIT.**
- README con: qué es, captura/GIF, `docker compose up`, requisitos (Ollama + Navidrome + API key AcoustID gratis), y compatibilidad (**cualquier Subsonic**).
- Pitch: *"Arreglá la metadata de tu biblioteca y armá playlists temáticas con IA, 100% local y gratis."*

---

_Fin del spec v0.1 — Bardo._
