from __future__ import annotations

ARTIST_SYSTEM = """\
Sos un curador musical y musicólogo con conocimiento enciclopédico de bandas de
todos los géneros. Describís a un ARTISTA para armar playlists temáticas.
Devolvés SIEMPRE JSON válido, sin markdown, sin texto extra.
Reglas:
- Géneros: usá el nombre canónico estándar (usualmente en inglés: "folk metal", "power metal").
- Temas/moods/referencias: en español neutro.
- No inventes. Si dudás, bajá "confidence".
- "description": 1 oración de 15-30 palabras, natural y rica en significado (se usa para embedding).
"""

ARTIST_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "genres": {"type": "array", "items": {"type": "string"}},
        "subgenres": {"type": "array", "items": {"type": "string"}},
        "country": {"type": "string"},
        "era": {"type": "string"},
        "language": {"type": "string"},
        "instrumentation": {"type": "array", "items": {"type": "string"}},
        "vocal_style": {"type": "string"},
        "lyrical_themes": {"type": "array", "items": {"type": "string"}},
        "moods": {"type": "array", "items": {"type": "string"}},
        "references": {"type": "array", "items": {"type": "string"}},
        "for_fans_of": {"type": "array", "items": {"type": "string"}},
        "energy": {"type": "number"},
        "description": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "name",
        "genres",
        "moods",
        "description",
        "confidence",
    ],
}


def artist_messages(
    artist_name: str,
    genres_from_library: list[str],
    albums_list: list[str],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ARTIST_SYSTEM},
        {
            "role": "user",
            "content": (
                "INPUT:\n"
                f"Artista: {artist_name}\n"
                "Géneros que ya figuran en la biblioteca: "
                f"{', '.join(genres_from_library) or '(ninguno)'}\n"
                f"Álbumes en la biblioteca: {', '.join(albums_list) or '(ninguno)'}\n\n"
                "OUTPUT (JSON):\n"
                '{ "name": "", "genres": [], "subgenres": [], "country": "",'
                ' "era": "", "language": "", "instrumentation": [],'
                ' "vocal_style": "", "lyrical_themes": [], "moods": [],'
                ' "references": [], "for_fans_of": [], "energy": 0.0,'
                ' "description": "", "confidence": 0.0 }'
            ),
        },
    ]


ALBUM_SYSTEM = """\
Igual que antes, pero enfocado en el ÁLBUM. Prestá especial atención a discos
CONCEPTUALES (ej: "Nightfall in Middle-Earth" → Tolkien/Silmarillion).
Devolvés JSON válido, sin markdown.
"""

ALBUM_SCHEMA = {
    "type": "object",
    "properties": {
        "artist": {"type": "string"},
        "album": {"type": "string"},
        "year": {"type": "integer"},
        "is_concept_album": {"type": "boolean"},
        "concept": {"type": "string"},
        "themes": {"type": "array", "items": {"type": "string"}},
        "moods": {"type": "array", "items": {"type": "string"}},
        "references": {"type": "array", "items": {"type": "string"}},
        "energy": {"type": "number"},
        "description": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["artist", "album", "themes", "moods", "description", "confidence"],
}


def album_messages(
    artist: str,
    album: str,
    year: int | None,
    artist_genres: list[str],
    track_titles: list[str] | None = None,
) -> list[dict[str, str]]:
    tracks_line = (
        f"Temas del álbum: {', '.join(track_titles[:40]) or '(sin datos)'}\n"
        if track_titles is not None
        else ""
    )
    return [
        {"role": "system", "content": ARTIST_SYSTEM + "\n" + ALBUM_SYSTEM},
        {
            "role": "user",
            "content": (
                "INPUT:\n"
                f"Artista: {artist or '(desconocido)'}\n"
                f"Álbum: {album}\n"
                f"Año: {year or '(desconocido)'}\n"
                f"Géneros del artista: {', '.join(artist_genres) or '(desconocidos)'}\n"
                f"{tracks_line}\n"
                "OUTPUT (JSON):\n"
                '{ "artist": "", "album": "", "year": 0,'
                ' "is_concept_album": false, "concept": "", "themes": [],'
                ' "moods": [], "references": [], "energy": 0.0,'
                ' "description": "", "confidence": 0.0 }'
            ),
        },
    ]


TRACK_SYSTEM = """\
Describís una CANCIÓN a partir de datos ya existentes. No inventes si falta info.
Devolvés JSON válido, sin markdown.
"""

TRACK_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "moods": {"type": "array", "items": {"type": "string"}},
        "themes": {"type": "array", "items": {"type": "string"}},
        "energy": {"type": "number"},
        "is_ballad": {"type": "boolean"},
        "is_instrumental": {"type": "boolean"},
        "description": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["title", "moods", "themes", "description", "confidence"],
}


def track_messages(
    artist: str,
    album: str,
    title: str,
    album_concept: str,
    album_tags: list[str],
    lyrics: str = "",
) -> list[dict[str, str]]:
    lyrics_block = (lyrics or "")[:4000]
    return [
        {"role": "system", "content": TRACK_SYSTEM},
        {
            "role": "user",
            "content": (
                "INPUT:\n"
                f"Artista: {artist}\n"
                f"Álbum: {album}\n"
                f"Título: {title}\n"
                f"Concepto del álbum: {album_concept or '(sin concepto)'}\n"
                f"Tags del álbum: {', '.join(album_tags) or '(sin tags)'}\n"
                f"Letra (si existe): {lyrics_block or '(sin letra)'}\n\n"
                "OUTPUT (JSON):\n"
                '{ "title": "", "moods": [], "themes": [], "energy": 0.0,'
                ' "is_ballad": false, "is_instrumental": false,'
                ' "description": "", "confidence": 0.0 }'
            ),
        },
    ]


EXPANSION_SYSTEM = """\
Convertís un pedido en lenguaje natural a un objeto de búsqueda. Devolvés JSON, sin markdown.
Asociá términos difusos a un campo semántico
("taberna" → ["folk metal","celta","fiesta","drinking song"]).

Reglas de filtros (IMPORTANTE, usalos para acotar el universo):
- "música en español" → filters.languages: ["es"]. "en inglés" → ["en"]. "en portugués" → ["pt"].
- "argentino", "de Argentina" → filters.countries: ["AR"]. "español" (de España) → ["ES"].
  OJO: "español" como idioma es languages, como país es countries. Si el pedido es ambiguo,
  poné el país SOLO si menciona gentilicio o país explícito.
- "de los 80" → filters.decades: [1980]. "noventas" → [1990].
- "tranquilo/relajado/calmo" → energy_max: 0.35. "enérgico/fiesta/bailable" → energy_min: 0.7.
- "instrumental" → instrumental: true. "baladas" → ballad: true.
- "álbum conceptual" → concept_album: true.
- "rock pero sin metal" → exclude_genres: ["metal"].
- "solo metal sinfónico" → include_genres: ["symphonic metal"].
- Usá códigos ISO (es, en, pt; AR, ES, MX) para languages/countries.
- Si no hay restricción clara, dejá la lista vacía o el campo en null.
"""

EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "canonical_terms": {"type": "array", "items": {"type": "string"}},
        "expanded_terms": {"type": "array", "items": {"type": "string"}},
        "moods": {"type": "array", "items": {"type": "string"}},
        "reference": {"type": "string"},
        "filters": {
            "type": "object",
            "properties": {
                "year_min": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "year_max": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "exclude_genres": {"type": "array", "items": {"type": "string"}},
                "include_genres": {"type": "array", "items": {"type": "string"}},
                "languages": {"type": "array", "items": {"type": "string"}},
                "countries": {"type": "array", "items": {"type": "string"}},
                "decades": {"type": "array", "items": {"type": "integer"}},
                "energy_min": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "energy_max": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "instrumental": {"type": "boolean"},
                "ballad": {"type": "boolean"},
                "concept_album": {"type": "boolean"},
            },
        },
        "size": {"type": "integer"},
    },
    "required": ["intent", "canonical_terms", "expanded_terms", "moods"],
}


def expansion_messages(user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EXPANSION_SYSTEM},
        {
            "role": "user",
            "content": (
                f'INPUT:\n"{user_prompt}"\n\n'
                "OUTPUT (JSON):\n"
                '{ "intent": "playlist", "canonical_terms": [],'
                ' "expanded_terms": [], "moods": [], "reference": "",'
                ' "filters": { "year_min": null, "year_max": null,'
                ' "exclude_genres": [], "include_genres": [],'
                ' "languages": [], "countries": [], "decades": [],'
                ' "energy_min": null, "energy_max": null,'
                ' "instrumental": null, "ballad": null,'
                ' "concept_album": null }, "size": 30 }'
            ),
        },
    ]


RERANK_SYSTEM = """\
Te doy un pedido y una lista de CANDIDATOS REALES de la biblioteca (id, artista,
álbum, título, facetas). Elegí los que mejor encajan. Devolvés JSON con IDs EXACTOS
de la lista dada. PROHIBIDO inventar IDs o títulos.
"""

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "playlist_name": {"type": "string"},
        "track_ids": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["playlist_name", "track_ids"],
}


def rerank_messages(
    user_prompt: str,
    candidates: list[dict],
    size: int = 30,
    moods: list[str] | None = None,
    reference: str = "",
) -> list[dict[str, str]]:
    import json as _json

    extra = ""
    if moods:
        extra += f"Moods buscados: {', '.join(moods)}\n"
    if reference:
        extra += f"Referencia del mundo: {reference}\n"
    return [
        {"role": "system", "content": RERANK_SYSTEM},
        {
            "role": "user",
            "content": (
                f"INPUT:\nPedido: {user_prompt}\n"
                f"{extra}"
                f"Candidatos: {_json.dumps(candidates, ensure_ascii=False)}\n"
                f"Cantidad deseada: {size}\n\n"
                "OUTPUT (JSON):\n"
                '{ "playlist_name": "", "track_ids": [], "reasoning": "" }'
            ),
        },
    ]


AGENT_SYSTEM = """\
Sos Bardo, arquitecto de playlists. Tenés tools. Objetivo: a partir del pedido del
usuario, armar una playlist con temas REALES de su biblioteca.
Reglas:
- Nunca inventes track IDs. Usá solo IDs devueltos por los tools.
- Flujo típico: search_candidates(pedido) → (si hace falta) list_albums/list_tracks →
  (opcional) web_search para validar un dato del mundo → create_playlist(ids).
- Máximo 6 tool calls. Sé eficiente. Si el pedido es ambiguo, asumí y aclaralo al final.
- Devolvé el nombre de la playlist y una justificación breve.

Filtros disponibles en search_candidates (usalos para pedidos por idioma, país,
época, energía o estilo):
- languages: ISO 639-1 (es, en, pt, ja). "música en español" → ["es"].
- countries: ISO 3166-1 (AR, ES, MX, US). "bandas argentinas" → ["AR"].
- decades: [1980], [1990]. "de los 80" → [1980].
- include_genres / exclude_genres: "solo metal" / "sin pop".
- energy_min/energy_max (0..1), instrumental, ballad, concept_album.
- Para pedidos amplios ("música en español") preferí languages y un limit alto,
  y NO inventes subgéneros: después filtrá con los candidatos que devuelva el tool.
"""

WEB_SEARCH_SYSTEM = """\
Verificás UN dato factual sobre música. Devolvés JSON {"answer": true|false, "evidence": ""}.
"""
