from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Iterable, Sequence

import httpx

from app.config import Settings, get_settings

log = logging.getLogger("bardo.ollama")


class OllamaError(Exception):
    pass


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _iter_json_candidates(text: str) -> Iterable[str]:
    """Genera substrings JSON balanceados, desde cada '{' o '['.

    Respeta strings y escapes, así que no se confunde con llaves dentro de
    comillas. Permite recuperar el objeto válido aunque haya basura antes.
    """
    openers = {"{": "}", "[": "]"}
    n = len(text)
    for start, char in enumerate(text):
        if char not in openers:
            continue
        stack = [openers[char]]
        in_string = False
        escaped = False
        for i in range(start + 1, n):
            current = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current in openers:
                stack.append(openers[current])
            elif current in ("}", "]"):
                if not stack or current != stack[-1]:
                    break
                stack.pop()
                if not stack:
                    yield text[start : i + 1]
                    break


def extract_json(text: str) -> Any:
    """Extrae el primer valor JSON de una respuesta de LLM.

    Tolera fences de markdown, texto antes/después y comas finales.
    """
    if text is None:
        raise ValueError("empty response")
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    last_error: Exception | None = None
    for candidate in _iter_json_candidates(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                last_error = exc
    if last_error:
        raise ValueError(f"invalid JSON: {last_error}") from last_error
    raise ValueError("no JSON object found in response")


class OllamaClient:
    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.settings = settings or get_settings()
        self.base_url = self.settings.ollama_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.ollama_timeout, connect=10.0),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def list_models(self) -> list[str]:
        response = await self._client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        fmt: Any | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.settings.ollama_chat_model,
            "messages": list(messages),
            "stream": False,
        }
        if tools:
            payload["tools"] = list(tools)
        if fmt is not None:
            payload["format"] = fmt
        if options:
            payload["options"] = options
        try:
            response = await self._client.post(
                f"{self.base_url}/api/chat", json=payload
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"ollama chat failed: {exc}") from exc
        return response.json()

    async def chat_json(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        retries: int = 2,
    ) -> Any:
        last_error: Exception | None = None
        working = list(messages)
        attempt = 0
        while attempt <= retries:
            response = await self.chat(
                working, model=model, fmt=schema or "json", options=options
            )
            content = response.get("message", {}).get("content", "")
            try:
                return extract_json(content)
            except ValueError as exc:
                last_error = exc
                attempt += 1
                working = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            "Tu respuesta anterior no era JSON válido. "
                            "Respondé SOLAMENTE con JSON, "
                            "sin markdown ni texto extra."
                        ),
                    }
                ]
        raise OllamaError(f"could not parse JSON: {last_error}")

    async def embed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": model or self.settings.ollama_embed_model,
            "input": list(texts),
        }
        try:
            response = await self._client.post(
                f"{self.base_url}/api/embed", json=payload
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"ollama embed failed: {exc}") from exc
        return response.json().get("embeddings", [])

    async def embed_one(self, text: str, *, model: str | None = None) -> list[float]:
        vectors = await self.embed([text], model=model)
        if not vectors:
            raise OllamaError("empty embedding")
        return vectors[0]

    async def ping(self) -> bool:
        try:
            await self._client.get(f"{self.base_url}/api/tags")
            return True
        except httpx.HTTPError:
            return False


async def embed_many(
    client: OllamaClient,
    texts: Iterable[str],
    *,
    batch_size: int = 16,
    model: str | None = None,
) -> list[list[float]]:
    items = list(texts)
    out: list[list[float]] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        vectors = await client.embed(batch, model=model)
        out.extend(vectors)
        await asyncio.sleep(0)
    return out
