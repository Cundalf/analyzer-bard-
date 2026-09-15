from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import httpx

from app.config import Settings, get_settings

log = logging.getLogger("bardo.subsonic")

API_VERSION = "1.16.1"
CLIENT_NAME = "bardo"


class SubsonicError(Exception):
    def __init__(self, code: int | None, message: str, endpoint: str = ""):
        self.code = code
        self.message = message
        self.endpoint = endpoint
        super().__init__(f"Subsonic error {code}: {message} ({endpoint})")


@dataclass
class Track:
    id: str
    title: str = ""
    album: str = ""
    album_id: str = ""
    artist: str = ""
    artist_id: str = ""
    track: int | None = None
    year: int | None = None
    genre: str = ""
    duration: int | None = None
    path: str = ""
    suffix: str = ""
    music_brainz_id: str = ""
    cover_art: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Track":
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            album=data.get("album", ""),
            album_id=data.get("albumId", ""),
            artist=data.get("artist", ""),
            artist_id=data.get("artistId", ""),
            track=_int_or_none(data.get("track")),
            year=_int_or_none(data.get("year")),
            genre=data.get("genre", "") or "",
            duration=_int_or_none(data.get("duration")),
            path=data.get("path", "") or "",
            suffix=data.get("suffix", "") or "",
            music_brainz_id=data.get("musicBrainzId", "") or "",
            cover_art=data.get("coverArt", "") or "",
        )


@dataclass
class Album:
    id: str
    name: str = ""
    artist: str = ""
    artist_id: str = ""
    year: int | None = None
    genre: str = ""
    song_count: int = 0
    duration: int = 0
    created: str = ""
    cover_art: str = ""
    songs: list[Track] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Album":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", "") or data.get("title", ""),
            artist=data.get("artist", "") or "",
            artist_id=data.get("artistId", "") or "",
            year=_int_or_none(data.get("year")),
            genre=data.get("genre", "") or "",
            song_count=_int_or_none(data.get("songCount")) or 0,
            duration=_int_or_none(data.get("duration")) or 0,
            created=data.get("created", "") or "",
            cover_art=data.get("coverArt", "") or "",
            songs=[Track.from_json(s) for s in data.get("song", []) or []],
        )


@dataclass
class Artist:
    id: str
    name: str = ""
    album_count: int = 0
    cover_art: str = ""
    music_brainz_id: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Artist":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            album_count=_int_or_none(data.get("albumCount")) or 0,
            cover_art=data.get("coverArt", "") or "",
            music_brainz_id=data.get("musicBrainzId", "") or "",
        )


@dataclass
class Playlist:
    id: str
    name: str = ""
    song_count: int = 0
    duration: int = 0
    owner: str = ""
    entries: list[Track] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Playlist":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            song_count=_int_or_none(data.get("songCount")) or 0,
            duration=_int_or_none(data.get("duration")) or 0,
            owner=data.get("owner", "") or "",
            entries=[Track.from_json(s) for s in data.get("entry", []) or []],
        )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class SubsonicClient:
    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.settings = settings or get_settings()
        self.base_url = self.settings.subsonic_url.rstrip("/")
        self.user = self.settings.subsonic_user
        self.password = self.settings.subsonic_pass
        self._client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0),
            transport=transport,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SubsonicClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _auth_params(self) -> dict[str, str]:
        salt = secrets.token_hex(8)
        token = hashlib.md5((self.password + salt).encode("utf-8")).hexdigest()
        return {
            "u": self.user,
            "t": token,
            "s": salt,
            "v": API_VERSION,
            "c": CLIENT_NAME,
            "f": "json",
        }

    def url_for(self, endpoint: str, **params: Any) -> str:
        query = self._auth_params()
        query.update(
            {k: v for k, v in params.items() if v is not None}
        )
        request = httpx.Request(
            "GET", f"{self.base_url}/rest/{endpoint}", params=query
        )
        return str(request.url)

    def _request(
        self,
        endpoint: str,
        *,
        retries: int = 2,
        params: list[tuple[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        query: list[tuple[str, Any]] = list(self._auth_params().items())
        query.extend(kwargs.items())
        if params:
            query.extend(params)
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = self._client.get(
                    f"{self.base_url}/rest/{endpoint}", params=query
                )
                response.raise_for_status()
                payload = response.json()
                body = payload.get("subsonic-response", {})
                if body.get("status") == "failed":
                    error = body.get("error", {})
                    raise SubsonicError(
                        _int_or_none(error.get("code")),
                        error.get("message", "unknown"),
                        endpoint,
                    )
                return body
            except SubsonicError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                last_exc = exc
                log.warning(
                    "subsonic %s attempt %s failed: %s",
                    endpoint,
                    attempt + 1,
                    exc,
                )
        raise SubsonicError(None, f"connection failed: {last_exc}", endpoint)

    def ping(self) -> dict[str, Any]:
        return self._request("ping.view")

    def get_artists(self) -> list[Artist]:
        body = self._request("getArtists.view")
        out: list[Artist] = []
        for index in body.get("artists", {}).get("index", []) or []:
            for artist in index.get("artist", []) or []:
                out.append(Artist.from_json(artist))
        return out

    def get_artist(self, artist_id: str) -> tuple[Artist, list[Album]]:
        body = self._request("getArtist.view", id=artist_id)
        data = body.get("artist", {})
        artist = Artist.from_json(data)
        albums = [Album.from_json(a) for a in data.get("album", []) or []]
        return artist, albums

    def get_album(self, album_id: str) -> Album:
        body = self._request("getAlbum.view", id=album_id)
        return Album.from_json(body.get("album", {}))

    def get_album_list2(
        self, list_type: str = "alphabeticalByName", size: int = 500, offset: int = 0
    ) -> list[Album]:
        body = self._request(
            "getAlbumList2.view", type=list_type, size=size, offset=offset
        )
        return [
            Album.from_json(a)
            for a in body.get("albumList2", {}).get("album", []) or []
        ]

    def iter_all_albums(self, page_size: int = 500) -> Iterable[Album]:
        offset = 0
        while True:
            page = self.get_album_list2(size=page_size, offset=offset)
            if not page:
                return
            yield from page
            if len(page) < page_size:
                return
            offset += page_size

    def search3(
        self,
        query: str,
        artist_count: int = 20,
        album_count: int = 20,
        song_count: int = 20,
        artist_offset: int = 0,
        album_offset: int = 0,
        song_offset: int = 0,
    ) -> dict[str, list[Any]]:
        body = self._request(
            "search3.view",
            query=query,
            artistCount=artist_count,
            artistOffset=artist_offset,
            albumCount=album_count,
            albumOffset=album_offset,
            songCount=song_count,
            songOffset=song_offset,
        )
        result = body.get("searchResult3", {})
        return {
            "artists": [Artist.from_json(a) for a in result.get("artist", []) or []],
            "albums": [Album.from_json(a) for a in result.get("album", []) or []],
            "songs": [Track.from_json(s) for s in result.get("song", []) or []],
        }

    def get_playlists(self) -> list[Playlist]:
        body = self._request("getPlaylists.view")
        return [
            Playlist.from_json(p)
            for p in body.get("playlists", {}).get("playlist", []) or []
        ]

    def get_playlist(self, playlist_id: str) -> Playlist:
        body = self._request("getPlaylist.view", id=playlist_id)
        return Playlist.from_json(body.get("playlist", {}))

    def create_playlist(self, name: str, track_ids: Sequence[str]) -> Playlist:
        params: list[tuple[str, Any]] = [("name", name)]
        params.extend(("songId", tid) for tid in track_ids)
        body = self._request("createPlaylist.view", params=params)
        return Playlist.from_json(body.get("playlist", {}))

    def update_playlist(
        self,
        playlist_id: str,
        name: str | None = None,
        song_ids_to_add: Sequence[str] = (),
        song_indexes_to_remove: Sequence[int] = (),
    ) -> Playlist:
        params: list[tuple[str, Any]] = [("playlistId", playlist_id)]
        if name is not None:
            params.append(("name", name))
        params.extend(("songIdToAdd", tid) for tid in song_ids_to_add)
        params.extend(("songIndexToRemove", idx) for idx in song_indexes_to_remove)
        body = self._request("updatePlaylist.view", params=params)
        return Playlist.from_json(body.get("playlist", {}))

    def delete_playlist(self, playlist_id: str) -> None:
        self._request("deletePlaylist.view", id=playlist_id)

    def star(self, track_id: str) -> None:
        self._request("star.view", id=track_id)

    def unstar(self, track_id: str) -> None:
        self._request("unstar.view", id=track_id)

    def start_scan(self, full: bool = False) -> dict[str, Any]:
        body = self._request("startScan.view", fullScan="true" if full else None)
        return body.get("scanStatus", {})

    def get_scan_status(self) -> dict[str, Any]:
        body = self._request("getScanStatus.view")
        return body.get("scanStatus", {})

    def stream_url(self, track_id: str, max_bit_rate: int | None = None) -> str:
        params: dict[str, Any] = {"id": track_id}
        if max_bit_rate:
            params["maxBitRate"] = max_bit_rate
        return self.url_for("stream.view", **params)

    def cover_url(self, cover_id: str, size: int = 300) -> str:
        return self.url_for("getCoverArt.view", id=cover_id, size=size)


def sync_library(
    conn: Any,
    client: SubsonicClient,
    *,
    with_tracks: bool = True,
    max_albums: int | None = None,
) -> dict[str, int]:
    """Espeja artistas/álbumes/tracks de Navidrome a SQLite.

    Incremental: los álbumes ya sincronizados no se vuelven a bajar salvo
    que `with_tracks` sea True y nunca se hayan bajado sus tracks.
    """
    from app.db import utcnow

    now = utcnow()
    stats = {"artists": 0, "albums": 0, "tracks": 0}

    artists = client.get_artists()
    for artist in artists:
        conn.execute(
            """
            INSERT INTO artists(id, navidrome_id, name, synced_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(navidrome_id) DO UPDATE SET
              name = excluded.name, synced_at = excluded.synced_at
            """,
            (f"artist:{artist.id}", artist.id, artist.name, now),
        )
        stats["artists"] += 1

    albums = list(client.iter_all_albums())
    if max_albums is not None:
        albums = albums[:max_albums]

    existing_track_counts = {
        row["navidrome_id"]: row["n"]
        for row in conn.execute(
            """
            SELECT al.navidrome_id AS navidrome_id, COUNT(t.id) AS n
            FROM tracks t
            JOIN albums al ON al.id = t.album_id
            GROUP BY al.navidrome_id
            """
        ).fetchall()
    }

    for album in albums:
        artist_row = (
            conn.execute(
                "SELECT id FROM artists WHERE navidrome_id = ?",
                (album.artist_id,),
            ).fetchone()
            if album.artist_id
            else None
        )
        if artist_row:
            artist_pk: str | None = artist_row["id"]
        elif album.artist_id:
            artist_pk = f"artist:{album.artist_id}"
            conn.execute(
                """
                INSERT INTO artists(id, navidrome_id, name, synced_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(navidrome_id) DO NOTHING
                """,
                (artist_pk, album.artist_id, album.artist, now),
            )
        else:
            artist_pk = None
        conn.execute(
            """
            INSERT INTO albums(id, navidrome_id, artist_id, name, year, genre, synced_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(navidrome_id) DO UPDATE SET
              artist_id = excluded.artist_id, name = excluded.name,
              year = excluded.year, genre = excluded.genre,
              synced_at = excluded.synced_at
            """,
            (
                f"album:{album.id}",
                album.id,
                artist_pk,
                album.name,
                album.year,
                album.genre,
                now,
            ),
        )
        stats["albums"] += 1

        needs_tracks = with_tracks and (
            existing_track_counts.get(album.id, 0) == 0
        )
        if needs_tracks:
            full = client.get_album(album.id)
            for track in full.songs:
                _upsert_track(conn, track, artist_pk, now)
                stats["tracks"] += 1

    conn.commit()
    return stats


def _upsert_track(
    conn: Any, track: Track, artist_pk: str | None, now: str
) -> None:
    album_row = (
        conn.execute(
            "SELECT id FROM albums WHERE navidrome_id = ?", (track.album_id,)
        ).fetchone()
        if track.album_id
        else None
    )
    if album_row:
        album_pk: str | None = album_row["id"]
    elif track.album_id:
        album_pk = f"album:{track.album_id}"
        conn.execute(
            """
            INSERT INTO albums(id, navidrome_id, name, synced_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(navidrome_id) DO NOTHING
            """,
            (album_pk, track.album_id, track.album, now),
        )
    else:
        album_pk = None

    if track.artist_id:
        current = (artist_pk or "").removeprefix("artist:")
        if not artist_pk or track.artist_id != current or current == "":
            artist_row = conn.execute(
                "SELECT id FROM artists WHERE navidrome_id = ?",
                (track.artist_id,),
            ).fetchone()
            if artist_row:
                artist_pk = artist_row["id"]
            else:
                artist_pk = f"artist:{track.artist_id}"
                conn.execute(
                    """
                    INSERT INTO artists(id, navidrome_id, name, synced_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(navidrome_id) DO NOTHING
                    """,
                    (artist_pk, track.artist_id, track.artist, now),
                )
    conn.execute(
        """
        INSERT INTO tracks(id, navidrome_id, album_id, artist_id, title,
                           duration, path, synced_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(navidrome_id) DO UPDATE SET
          album_id = excluded.album_id, artist_id = excluded.artist_id,
          title = excluded.title, duration = excluded.duration,
          path = excluded.path, synced_at = excluded.synced_at
        """,
        (
            f"track:{track.id}",
            track.id,
            album_pk,
            artist_pk,
            track.title,
            track.duration,
            track.path,
            now,
        ),
    )
