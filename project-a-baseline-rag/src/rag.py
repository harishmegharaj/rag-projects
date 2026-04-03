"""
RAG: retrieve → build prompt with citations → call LLM (OpenAI if key set; else stub).
"""
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from .retrieve import query_collection

load_dotenv()


def format_context(hits: List[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        meta = h.get("metadata") or {}
        src = meta.get("source", "unknown")
        page = meta.get("page")
        page_s = f" p.{page}" if page is not None and page >= 0 else ""
        blocks.append(f"[{i}] (source: {src}{page_s})\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def answer_with_rag(
    question: str,
    chroma_dir: Path,
    k: int = 5,
) -> dict:
    hits = query_collection(chroma_dir, question, k=k)
    context = format_context(hits)

    system = (
        "You are a careful assistant. Answer only using the provided context. "
        "If the context is insufficient, say so. Cite sources using [1], [2] notation."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        r = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        answer = r.choices[0].message.content or ""
    else:
        answer = (
            "[Stub — set OPENAI_API_KEY in .env for real answers]\n\n"
            f"Retrieved {len(hits)} chunks. First snippet:\n{hits[0]['text'][:500] if hits else '(no chunks)'}"
        )

    return {"answer": answer, "retrieved": hits}
