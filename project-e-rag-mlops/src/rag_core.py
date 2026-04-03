"""Lightweight TF-IDF RAG + optional OpenAI; stub mode for tests."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import CHUNKS_PATH, CORPUS_DIR, STORE_DIR, TOP_K, VECTORIZER_PATH


def _split_chunks(text: str, source: str) -> list[dict[str, Any]]:
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return [{"text": p, "source": source, "chunk_id": f"{source}#{i}"} for i, p in enumerate(parts)]


def load_corpus_chunks() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    if not CORPUS_DIR.is_dir():
        return chunks
    for path in sorted(CORPUS_DIR.rglob("*.md")):
        rel = str(path.relative_to(CORPUS_DIR))
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks.extend(_split_chunks(text, rel))
    return chunks


def build_index() -> dict[str, Any]:
    """Fit TF-IDF on corpus chunks; persist vectorizer + chunk metadata."""
    chunks = load_corpus_chunks()
    if not chunks:
        raise RuntimeError(f"No chunks found under {CORPUS_DIR}")
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(max_features=8192, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, VECTORIZER_PATH)
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    return {"num_chunks": len(chunks), "vectorizer_path": str(VECTORIZER_PATH)}


def _load_index():
    if not VECTORIZER_PATH.is_file() or not CHUNKS_PATH.is_file():
        return None
    data = joblib.load(VECTORIZER_PATH)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return data["vectorizer"], data["matrix"], chunks


@dataclass
class RAGResult:
    answer: str
    chunks: list[dict[str, Any]]
    retrieval_hits: int
    llm_mode: str
    token_estimate: int


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _generate_stub(question: str, ctx: list[dict[str, Any]]) -> tuple[str, int]:
    snippet = " ".join(c["text"][:200] for c in ctx[:2])
    answer = (
        f"[stub-llm] Based on the retrieved context, here is a concise answer.\n"
        f"Q: {question}\nContext preview: {snippet[:400]}..."
    )
    return answer, _estimate_tokens(answer) + _estimate_tokens(question)


def _generate_openai(question: str, ctx: list[dict[str, Any]]) -> tuple[str, int]:
    from openai import OpenAI

    client = OpenAI()
    sys_prompt = "Answer using only the provided context. Cite sources by filename if relevant."
    user = f"Question: {question}\n\nContext:\n" + "\n---\n".join(
        f"[{c['source']}] {c['text']}" for c in ctx
    )
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    if usage and getattr(usage, "total_tokens", None):
        tok = int(usage.total_tokens)
    else:
        tok = _estimate_tokens(text) + _estimate_tokens(user)
    return text, tok


def retrieve(question: str, k: int = TOP_K) -> list[dict[str, Any]]:
    loaded = _load_index()
    if not loaded:
        return []
    vectorizer, matrix, chunks = loaded
    qv = vectorizer.transform([question])
    sims = cosine_similarity(qv, matrix).ravel()
    idx = np.argsort(-sims)[:k]
    out = []
    for i in idx:
        if sims[i] <= 0:
            continue
        c = dict(chunks[int(i)])
        c["score"] = float(sims[i])
        out.append(c)
    return out


def ask(question: str) -> RAGResult:
    chunks = retrieve(question)
    n = len(chunks)
    use_llm = os.environ.get("OPENAI_API_KEY")
    if use_llm:
        answer, tok = _generate_openai(question, chunks)
        mode = "openai"
    else:
        answer, tok = _generate_stub(question, chunks)
        mode = "stub"
    return RAGResult(
        answer=answer,
        chunks=chunks,
        retrieval_hits=n,
        llm_mode=mode,
        token_estimate=tok,
    )
