from __future__ import annotations

from app.agent.prompts import (
    AGENT_SYSTEM,
    EXPANSION_SCHEMA,
    EXPANSION_SYSTEM,
    RERANK_SCHEMA,
    RERANK_SYSTEM,
    WEB_SEARCH_SYSTEM,
    expansion_messages,
    rerank_messages,
)


def test_agent_prompts_reexport():
    assert "Bardo" in AGENT_SYSTEM
    assert EXPANSION_SCHEMA["type"] == "object"
    assert RERANK_SCHEMA["type"] == "object"
    assert "taberna" in EXPANSION_SYSTEM
    assert "PROHIBIDO" in RERANK_SYSTEM
    assert "answer" in WEB_SEARCH_SYSTEM


def test_expansion_messages_shape():
    messages = expansion_messages("fiesta de taberna")
    assert messages[0]["role"] == "system"
    assert "fiesta de taberna" in messages[1]["content"]


def test_rerank_messages_includes_candidates_and_size():
    candidates = [{"track_id": "t1", "title": "A"}]
    messages = rerank_messages("taberna", candidates, size=10)
    assert "t1" in messages[1]["content"]
    assert "Cantidad deseada: 10" in messages[1]["content"]


def test_rerank_messages_unicode_preserved():
    candidates = [{"track_id": "t1", "title": "Ñandú"}]
    messages = rerank_messages("taberna", candidates, size=5)
    assert "Ñandú" in messages[1]["content"]


def test_rerank_messages_with_moods_and_reference():
    from app.enrich.prompts import rerank_messages

    messages = rerank_messages(
        "taberna", [{"track_id": "t1"}], 10, moods=["fiesta"], reference="tolkien"
    )
    content = messages[1]["content"]
    assert "Moods buscados: fiesta" in content
    assert "Referencia del mundo: tolkien" in content


def test_rerank_messages_without_extras():
    from app.enrich.prompts import rerank_messages

    content = rerank_messages("taberna", [], 5)[1]["content"]
    assert "Moods" not in content
    assert "Referencia" not in content
