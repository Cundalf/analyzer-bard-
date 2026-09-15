from __future__ import annotations

import json

import httpx
import pytest

from app.ollama import OllamaClient, OllamaError, embed_many, extract_json


# ------------------------------------------------------------ extract_json

@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('  {"a": 1}  ', {"a": 1}),
        ('[1, 2, 3]', [1, 2, 3]),
        ('"solo un string"', "solo un string"),
        ("123", 123),
        ("null", None),
        ("true", True),
    ],
)
def test_extract_json_valid_values(text, expected):
    assert extract_json(text) == expected


@pytest.mark.parametrize(
    "text,key",
    [
        ('```json\n{"a": 1}\n```', "a"),
        ('```JSON\n{"a": 1}\n```', "a"),
        ('```\n{"a": 1}\n```', "a"),
        ('```json\n{\n  "a": 1\n}\n```', "a"),
        ('Texto antes\n```json\n{"a": 1}\n```\nTexto después', "a"),
    ],
)
def test_extract_json_fences(text, key):
    assert extract_json(text)[key] == 1


def test_extract_json_text_around():
    assert extract_json('Claro:\n{"a": true}\nListo.') == {"a": True}


def test_extract_json_nested():
    text = 'bla {"outer": {"inner": [1, {"deep": true}]}} bla'
    assert extract_json(text)["outer"]["inner"][1]["deep"] is True


def test_extract_json_trailing_comma():
    assert extract_json('{"a": [1, 2,], "b": 3,}') == {"a": [1, 2], "b": 3}


def test_extract_json_trailing_comma_nested():
    assert extract_json('{"a": {"b": 1,},}') == {"a": {"b": 1}}


def test_extract_json_fence_with_bad_json_uses_object():
    text = '```json\n{esto no es json}\n```\n{"a": 1}'
    assert extract_json(text) == {"a": 1}


@pytest.mark.parametrize("text", ["", "   ", "sin json", "{incompleto", "```json\n```"])
def test_extract_json_invalid(text):
    with pytest.raises(ValueError):
        extract_json(text)


def test_extract_json_none_raises():
    with pytest.raises(ValueError):
        extract_json(None)


# ------------------------------------------------------------ cliente

def embed_transport(vectors=None, fail_times=0, status=200):
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] <= fail_times:
            raise httpx.ConnectError("boom", request=request)
        payload = json.loads(request.content)
        n = len(payload["input"])
        return httpx.Response(status, json={"embeddings": vectors or [[0.1] * 768] * n})

    return httpx.MockTransport(handler), state


@pytest.mark.anyio
async def test_embed_empty_returns_empty(settings):
    transport, state = embed_transport()
    client = OllamaClient(settings, transport=transport)
    assert await client.embed([]) == []
    assert state["n"] == 0
    await client.close()


@pytest.mark.anyio
async def test_embed_posts_model_and_input(settings):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.5] * 768]})

    client = OllamaClient(settings, transport=httpx.MockTransport(handler))
    result = await client.embed(["hola"])
    assert captured["model"] == settings.ollama_embed_model
    assert captured["input"] == ["hola"]
    assert result == [[0.5] * 768]
    await client.close()


@pytest.mark.anyio
async def test_embed_one_ok_and_error(settings):
    transport, _ = embed_transport()
    client = OllamaClient(settings, transport=transport)
    vector = await client.embed_one("x")
    assert len(vector) == 768
    await client.close()

    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": []})

    client2 = OllamaClient(settings, transport=httpx.MockTransport(empty))
    with pytest.raises(OllamaError):
        await client2.embed_one("x")
    await client2.close()


@pytest.mark.anyio
async def test_embed_http_error_raises(settings):
    client = OllamaClient(
        settings, transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    )
    with pytest.raises(OllamaError):
        await client.embed(["x"])
    await client.close()


@pytest.mark.anyio
async def test_chat_payload_includes_tools_format_options(settings):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    client = OllamaClient(settings, transport=httpx.MockTransport(handler))
    await client.chat(
        [{"role": "user", "content": "hola"}],
        model="m",
        tools=[{"type": "function", "function": {"name": "t"}}],
        fmt="json",
        options={"temperature": 0},
    )
    assert captured["model"] == "m"
    assert captured["stream"] is False
    assert captured["format"] == "json"
    assert captured["options"] == {"temperature": 0}
    assert len(captured["tools"]) == 1
    await client.close()


@pytest.mark.anyio
async def test_chat_default_model(settings):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {}})

    client = OllamaClient(settings, transport=httpx.MockTransport(handler))
    await client.chat([{"role": "user", "content": "x"}])
    assert captured["model"] == settings.ollama_chat_model
    assert "tools" not in captured
    assert "format" not in captured
    await client.close()


@pytest.mark.anyio
async def test_chat_http_error_raises(settings):
    client = OllamaClient(
        settings, transport=httpx.MockTransport(lambda r: httpx.Response(404))
    )
    with pytest.raises(OllamaError):
        await client.chat([{"role": "user", "content": "x"}])
    await client.close()


@pytest.mark.anyio
async def test_chat_json_success_first_try(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": '{"a": 1}'}})

    client = OllamaClient(settings, transport=httpx.MockTransport(handler))
    assert await client.chat_json([{"role": "user", "content": "x"}]) == {"a": 1}
    await client.close()


@pytest.mark.anyio
async def test_chat_json_retries_then_succeeds(settings):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        content = "no json" if calls["n"] < 3 else '{"ok": true}'
        return httpx.Response(200, json={"message": {"content": content}})

    client = OllamaClient(settings, transport=httpx.MockTransport(handler))
    assert await client.chat_json([{"role": "user", "content": "x"}]) == {"ok": True}
    assert calls["n"] == 3
    await client.close()


@pytest.mark.anyio
async def test_chat_json_retries_exhausted_raises(settings):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"message": {"content": "nunca json"}})

    client = OllamaClient(settings, transport=httpx.MockTransport(handler), )
    with pytest.raises(OllamaError):
        await client.chat_json([{"role": "user", "content": "x"}], retries=1)
    assert calls["n"] == 2
    await client.close()


@pytest.mark.anyio
async def test_chat_json_sends_schema(settings):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    client = OllamaClient(settings, transport=httpx.MockTransport(handler))
    await client.chat_json([{"role": "user", "content": "x"}], schema=schema)
    assert captured["format"] == schema
    await client.close()


@pytest.mark.anyio
async def test_chat_json_retry_message_included(settings):
    captured_messages = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_messages.append(json.loads(request.content)["messages"])
        content = "no json" if len(captured_messages) == 1 else '{"ok": 1}'
        return httpx.Response(200, json={"message": {"content": content}})

    client = OllamaClient(settings, transport=httpx.MockTransport(handler))
    await client.chat_json([{"role": "user", "content": "original"}])
    assert len(captured_messages) == 2
    assert captured_messages[0][0]["content"] == "original"
    assert any("JSON válido" in m["content"] for m in captured_messages[1])
    await client.close()


@pytest.mark.anyio
async def test_list_models(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "a"}, {"name": "b"}]})

    client = OllamaClient(settings, transport=httpx.MockTransport(handler))
    assert await client.list_models() == ["a", "b"]
    await client.close()


@pytest.mark.anyio
async def test_list_models_error_propagates(settings):
    client = OllamaClient(
        settings, transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    with pytest.raises(httpx.HTTPStatusError):
        await client.list_models()
    await client.close()


@pytest.mark.anyio
async def test_ping_true_and_false(settings):
    client = OllamaClient(
        settings,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"models": []})),
    )
    assert await client.ping() is True
    await client.close()

    client2 = OllamaClient(
        settings,
        transport=httpx.MockTransport(
            lambda r: (_ for _ in ()).throw(httpx.ConnectError("x", request=r))
        ),
    )
    assert await client2.ping() is False
    await client2.close()


@pytest.mark.anyio
async def test_embed_many_batches(settings):
    batches = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        batches.append(len(payload["input"]))
        return httpx.Response(
            200, json={"embeddings": [[0.1] * 4] * len(payload["input"])}
        )

    client = OllamaClient(settings, transport=httpx.MockTransport(handler))
    vectors = await embed_many(client, [f"t{i}" for i in range(10)], batch_size=4)
    assert len(vectors) == 10
    assert batches == [4, 4, 2]
    await client.close()


@pytest.mark.anyio
async def test_embed_many_empty(settings):
    transport, state = embed_transport()
    client = OllamaClient(settings, transport=transport)
    assert await embed_many(client, [], batch_size=4) == []
    assert state["n"] == 0
    await client.close()


@pytest.mark.anyio
async def test_context_manager(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json={"embeddings": [[0.1] * 768]})

    async with OllamaClient(settings, transport=httpx.MockTransport(handler)) as client:
        assert await client.ping() is True
    assert client._client.is_closed
