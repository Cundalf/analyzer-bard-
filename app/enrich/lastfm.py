from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.config import Settings, get_settings

log = logging.getLogger("bardo.enrich.lastfm")

API_ROOT = "https://ws.audioscrobbler.com/2.0/"


class LastFmClient:
    def __init__(self, settings: Settings | None = None, api_key: str | None = None):
        self.settings = settings or get_settings()
        self.api_key = api_key or self.settings.lastfm_api_key
        self._client = httpx.Client(timeout=10.0)
        self._last_call = 0.0

    def close(self) -> None:
        self._client.close()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get(self, method: str, **params: Any) -> dict[str, Any]:
        if not self.available:
            return {}
        elapsed = time.monotonic() - self._last_call
        if elapsed < 0.25:
            time.sleep(0.25 - elapsed)
        query = {
            "method": method,
            "api_key": self.api_key,
            "format": "json",
            "autocorrect": 1,
            **params,
        }
        try:
            response = self._client.get(API_ROOT, params=query)
            response.raise_for_status()
            self._last_call = time.monotonic()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("last.fm %s failed: %s", method, exc)
            return {}

    def artist_top_tags(self, artist: str, limit: int = 15) -> list[str]:
        data = self._get("artist.getTopTags", artist=artist)
        tags = data.get("toptags", {}).get("tag", []) or []
        return [
            t.get("name", "")
            for t in tags[:limit]
            if t.get("name") and int(t.get("count", 0) or 0) > 0
        ]

    def track_top_tags(self, artist: str, track: str, limit: int = 10) -> list[str]:
        data = self._get("track.getTopTags", artist=artist, track=track)
        tags = data.get("toptags", {}).get("tag", []) or []
        return [
            t.get("name", "")
            for t in tags[:limit]
            if t.get("name") and int(t.get("count", 0) or 0) > 0
        ]

    def fetch_for(
        self, conn: Any, entity_type: str, key: str, artist: str, track: str = ""
    ) -> list[str]:
        import json
        from datetime import UTC, datetime, timedelta

        row = conn.execute(
            "SELECT payload, fetched_at FROM lastfm_cache WHERE entity_type = ? AND entity_key = ?",
            (entity_type, key),
        ).fetchone()
        if row:
            try:
                fetched = datetime.fromisoformat(row["fetched_at"])
                if datetime.now(UTC) - fetched < timedelta(days=30):
                    return json.loads(row["payload"])
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        tags = (
            self.track_top_tags(artist, track)
            if entity_type == "track"
            else self.artist_top_tags(artist)
        )
        conn.execute(
            "INSERT INTO lastfm_cache(entity_type, entity_key, payload, fetched_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(entity_type, entity_key) "
            "DO UPDATE SET payload = excluded.payload, fetched_at = excluded.fetched_at",
            (
                entity_type,
                key,
                json.dumps(tags, ensure_ascii=False),
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return tags
