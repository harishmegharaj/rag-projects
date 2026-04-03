"""Prometheus metrics for RAG / LLM operations."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_LATENCY = Histogram(
    "rag_http_request_latency_seconds",
    "HTTP request latency",
    ["route"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

RAG_ERRORS = Counter(
    "rag_errors_total",
    "RAG pipeline errors",
    ["stage"],
)

RAG_RETRIEVAL_CHUNKS = Histogram(
    "rag_retrieval_chunks",
    "Number of chunks returned by retrieval",
    buckets=(0, 1, 2, 3, 4, 5, 8, 16),
)

RAG_RETRIEVAL_HIT = Counter(
    "rag_retrieval_hit_total",
    "Whether retrieval returned at least one chunk (1=yes, 0=no)",
    ["hit"],
)

LLM_TOKENS_ESTIMATE = Counter(
    "rag_llm_tokens_estimated_total",
    "Rough token estimate for LLM calls (chars/4)",
)

LLM_CALLS = Counter(
    "rag_llm_calls_total",
    "LLM invocations",
    ["mode"],
)
