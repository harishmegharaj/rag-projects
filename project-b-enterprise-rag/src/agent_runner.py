"""LangGraph ReAct agent: the model chooses retrieval, calculator, or allowlisted HTTP GET."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from . import config
from .agent_tools import build_agent_tools
from .guardrails import check_query

_AGENT_SYSTEM = (
    "You are a careful assistant with tools.\n"
    "- Use search_documents for facts that might appear in the indexed document corpus.\n"
    "- Use calculator for arithmetic the user asks you to compute.\n"
    "- Use fetch_internal_http only when the user gives an internal URL you must read "
    "(hosts must be allowlisted via AGENT_HTTP_ALLOWLIST).\n"
    "After tools return, answer concisely in plain language. If sources were used, cite as [1], [2] when helpful."
)


def _final_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if not isinstance(m, (AIMessage, AIMessageChunk)):
            continue
        c = m.content
        if isinstance(c, str) and c.strip():
            return c.strip()
        if isinstance(c, list):
            parts = []
            for block in c:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            joined = "".join(parts).strip()
            if joined:
                return joined
    return ""


def run_agent(
    question: str,
    chroma_dir: Path | None = None,
    bm25_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one ReAct turn. Requires OPENAI_API_KEY. Reuses guardrails (length, blocklist, PII redaction)."""
    g = check_query(question)
    if not g.ok:
        return {
            "answer": f"I cannot process this request ({g.reason})",
            "blocked": True,
            "messages": [],
        }

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "answer": "Set OPENAI_API_KEY to run the tool agent (the ReAct loop needs a tool-calling chat model).",
            "blocked": False,
            "messages": [],
        }

    q = g.redacted_query or question
    chroma = chroma_dir or config.chroma_persist_dir()
    bm25 = bm25_dir or config.bm25_index_dir()
    tools = build_agent_tools(chroma, bm25)

    model = ChatOpenAI(
        model=config.openai_chat_model(),
        temperature=0.2,
        api_key=api_key,
    )
    graph = create_react_agent(model, tools, prompt=_AGENT_SYSTEM)
    result = graph.invoke({"messages": [("user", q)]})
    msgs = list(result.get("messages") or [])
    return {
        "answer": _final_text(msgs) or "(No assistant text in final message.)",
        "blocked": False,
        "messages": msgs,
    }
