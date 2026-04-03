"""Offline evaluation against a JSONL gold set (training-ish / LLMOps loop)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import bm25_index_dir, chroma_persist_dir
from .rag_pipeline import run_pipeline


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def keyword_overlap(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    lower = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in lower)
    return hits / len(keywords)


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    q = str(case["question"]).strip()
    chroma = chroma_persist_dir()
    bm25 = bm25_index_dir()
    t0 = time.perf_counter()
    out = run_pipeline(q, chroma, bm25)
    latency = time.perf_counter() - t0
    keys = case.get("expected_keywords") or []
    if isinstance(keys, str):
        keys = [keys]
    overlap = keyword_overlap(out.get("answer", ""), list(keys))
    return {
        "id": case.get("id", ""),
        "question": q,
        "latency_s": round(latency, 4),
        "keyword_overlap": round(overlap, 4),
        "blocked": out.get("blocked", False),
        "no_context": out.get("no_context", False),
        "error": out.get("error", False),
        "error_code": out.get("error_code"),
        "answer_preview": (out.get("answer") or "")[:500],
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n": 0}
    latencies = [float(r["latency_s"]) for r in results]
    overlaps = [float(r["keyword_overlap"]) for r in results]
    sorted_lat = sorted(latencies)
    p95_idx = min(int(0.95 * (n - 1)), n - 1) if n > 1 else 0
    return {
        "n": n,
        "mean_latency_s": round(sum(latencies) / n, 4),
        "p95_latency_s": round(sorted_lat[p95_idx], 4),
        "mean_keyword_overlap": round(sum(overlaps) / n, 4),
        "error_count": sum(1 for r in results if r.get("error")),
        "blocked_count": sum(1 for r in results if r.get("blocked")),
        "no_context_count": sum(1 for r in results if r.get("no_context")),
    }
