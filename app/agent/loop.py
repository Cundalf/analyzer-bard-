from __future__ import annotations

import json
import logging
from typing import Any, Callable

from app.agent import tools as tools_mod
from app.agent.tools import ToolContext, tool_result_message, tool_schemas
from app.config import get_settings
from app.enrich.canonicalize import Canonicalizer
from app.enrich.prompts import AGENT_SYSTEM
from app.ollama import OllamaClient

log = logging.getLogger("bardo.agent.loop")

Progress = Callable[[str, dict[str, Any]], None]


def _noop(stage: str, payload: dict[str, Any]) -> None:
    pass


async def run_agent(
    conn: Any,
    prompt: str,
    *,
    ctx: ToolContext | None = None,
    settings: Any | None = None,
    ollama: OllamaClient | None = None,
    max_tool_calls: int | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    progress = progress or _noop
    owns = ollama is None
    ollama = ollama or OllamaClient(settings)
    max_tool_calls = max_tool_calls or settings.agent_max_tool_calls

    if ctx is None:
        ctx = ToolContext(
            conn=conn,
            ollama=ollama,
            settings=settings,
            canon=Canonicalizer.from_db(conn),
        )
    ctx.ollama = ollama

    tool_defs = tool_schemas(settings)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Pedido: {prompt}\n"
                f"Objetivo: entre {settings.playlist_default_size // 2} y "
                f"{settings.playlist_default_size} temas. "
                "Cuando tengas la selección final, llamá a create_playlist."
            ),
        },
    ]

    tool_calls_made = 0
    trace: list[dict[str, Any]] = []
    try:
        while tool_calls_made < max_tool_calls:
            response = await ollama.chat(messages, tools=tool_defs)
            message = response.get("message", {})
            calls = message.get("tool_calls") or []
            if not calls:
                final_text = (message.get("content") or "").strip()
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": calls,
                }
            )
            for call in calls:
                if tool_calls_made >= max_tool_calls:
                    break
                fn = call.get("function", {})
                name = fn.get("name", "")
                arguments = fn.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                tool_calls_made += 1
                progress(
                    "tool_call",
                    {"name": name, "arguments": arguments, "n": tool_calls_made},
                )
                result = await tools_mod.execute_tool(name, arguments, ctx)
                trace.append(
                    {
                        "tool": name,
                        "arguments": arguments,
                        "result_preview": str(result)[:500],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": tool_result_message(name, result),
                    }
                )
        else:
            final_text = ""

        return {
            "text": final_text if "final_text" in locals() else "",
            "tool_calls": tool_calls_made,
            "trace": trace,
            "created_playlists": ctx.created_playlists,
            "validation": ctx.validation,
            "messages": messages[-1:] if tool_calls_made else messages,
        }
    finally:
        if owns:
            await ollama.close()
