"""Document summarization: stuff (single pass) and map-reduce for long text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from openai import OpenAI

from src.config import Settings


Strategy = Literal["auto", "stuff", "map_reduce"]


@dataclass
class SummaryResult:
    summary: str
    strategy_used: str
    llm_mode: Literal["openai", "stub"]
    chunk_count: int


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def _stub_summary(text: str, max_out: int = 800) -> str:
    head = text[:max_out].strip()
    if len(text) > max_out:
        head += "\n\n[Truncated for stub mode — set OPENAI_API_KEY for full summarization.]"
    return head


def _openai_chat(client: OpenAI, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    choice = resp.choices[0].message.content
    return (choice or "").strip()


def summarize_text(
    text: str,
    settings: Settings,
    strategy: Strategy = "auto",
    title: str | None = None,
) -> SummaryResult:
    text = text.strip()
    if not text:
        return SummaryResult(
            summary="",
            strategy_used="empty",
            llm_mode="stub",
            chunk_count=0,
        )

    key = settings.openai_api_key
    use_openai = bool(key and key.strip())

    if not use_openai:
        return SummaryResult(
            summary=_stub_summary(text),
            strategy_used="stub",
            llm_mode="stub",
            chunk_count=1,
        )

    client = OpenAI(api_key=key)
    model = settings.openai_chat_model
    title_hint = f"Document title (if any): {title}\n\n" if title else ""

    if strategy == "auto":
        if len(text) <= settings.stuff_max_chars:
            strategy = "stuff"
        else:
            strategy = "map_reduce"

    if strategy == "stuff":
        user = f"{title_hint}Summarize the following document clearly and concisely. "
        user += "Use short sections or bullets if helpful. Preserve important names, numbers, and dates.\n\n"
        user += text
        summary = _openai_chat(
            client,
            model,
            "You are a precise technical summarizer.",
            user,
        )
        return SummaryResult(
            summary=summary,
            strategy_used="stuff",
            llm_mode="openai",
            chunk_count=1,
        )

    # map_reduce
    chunks = _chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    partials: list[str] = []
    sys_part = "Summarize this excerpt in 2–4 sentences. Keep key facts and terminology."
    for i, ch in enumerate(chunks):
        u = f"Excerpt {i + 1} of {len(chunks)}:\n\n{ch}"
        partials.append(_openai_chat(client, model, sys_part, u))

    combined = "\n\n".join(f"[Part {i + 1}]\n{p}" for i, p in enumerate(partials))
    final_user = (
        f"{title_hint}You are given partial summaries of a long document. "
        "Produce one cohesive summary: overview, main points, and any critical details.\n\n"
        f"{combined}"
    )
    summary = _openai_chat(
        client,
        model,
        "You synthesize partial summaries into a single accurate document summary.",
        final_user,
    )
    return SummaryResult(
        summary=summary,
        strategy_used="map_reduce",
        llm_mode="openai",
        chunk_count=len(chunks),
    )
