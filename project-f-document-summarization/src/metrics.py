"""Prometheus metrics."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_LATENCY = Histogram(
    "summary_http_request_latency_seconds",
    "Latency for HTTP routes",
    ["route"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

SUMMARY_JOBS = Counter(
    "summary_jobs_total",
    "Summarization jobs by outcome",
    ["status"],
)

SUMMARY_STRATEGY = Counter(
    "summary_strategy_total",
    "Summaries by strategy",
    ["strategy"],
)

LLM_MODE = Counter(
    "summary_llm_mode_total",
    "LLM mode used",
    ["mode"],
)
