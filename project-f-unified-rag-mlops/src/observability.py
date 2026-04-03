"""Metrics, structured logging, and index readiness for Phase 4 (LLMOps)."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from . import config

logger = logging.getLogger("unified_rag_mlops")

_ASK_SECONDS = Histogram(
    "unified_rag_ask_duration_seconds",
    "Wall time for run_pipeline inside POST /v1/ask",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
_ASK_OUTCOMES = Counter(
    "unified_rag_ask_outcomes_total",
    "Outcomes from ask pipeline",
    ["outcome"],
)
_REBUILD_SECONDS = Histogram(
    "unified_rag_index_rebuild_duration_seconds",
    "Wall time for full index rebuild",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)
_INTENT_SECONDS = Histogram(
    "unified_rag_intent_duration_seconds",
    "Wall time for intent classifier inference",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
_FEEDBACK_TOTAL = Counter(
    "unified_rag_feedback_total",
    "Feedback submissions",
    ["rating"],
)


def build_id() -> str:
    return os.getenv("BUILD_ID", os.getenv("GIT_SHA", "dev")).strip() or "dev"


def service_version() -> str:
    return os.getenv("SERVICE_VERSION", "0.2.0").strip() or "0.2.0"


def metrics_require_auth() -> bool:
    return os.getenv("METRICS_REQUIRE_AUTH", "").lower() in ("1", "true", "yes")


def setup_logging() -> None:
    """JSON logs when LOG_JSON=1 (one object per line); else plain text."""
    root = logging.getLogger()
    if getattr(root, "_unified_rag_logging_configured", False):
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level, logging.INFO))
    use_json = os.getenv("LOG_JSON", "").lower() in ("1", "true", "yes")
    handler = logging.StreamHandler(sys.stdout)
    if use_json:

        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload: dict[str, Any] = {
                    "ts": self.formatTime(record, self.datefmt),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    payload["exc_info"] = self.formatException(record.exc_info)
                rid = getattr(record, "request_id", None)
                if rid:
                    payload["request_id"] = rid
                return json.dumps(payload, ensure_ascii=False)

        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.handlers.clear()
    root.addHandler(handler)
    setattr(root, "_unified_rag_logging_configured", True)


def index_readiness(chroma_dir: Path | None = None, bm25_dir: Path | None = None) -> dict[str, Any]:
    """Cheap checks: Chroma dir non-empty; BM25 artifacts present."""
    c = chroma_dir or config.chroma_persist_dir()
    b = bm25_dir or config.bm25_index_dir()
    chroma_ok = c.is_dir() and any(c.iterdir()) if c.exists() else False
    chunks = b / "chunks.json"
    bm25_pkl = b / "bm25.pkl"
    bm25_ok = chunks.is_file() and bm25_pkl.is_file()
    ready = chroma_ok and bm25_ok
    return {
        "ready": ready,
        "chroma_dir": str(c),
        "bm25_dir": str(b),
        "chroma_populated": chroma_ok,
        "bm25_artifacts_present": bm25_ok,
    }


def service_readiness() -> dict[str, Any]:
    """RAG indexes + optional intent model when INTENT_REQUIRED_FOR_READY=1."""
    from .intent_model import intent_artifact_status

    rag = index_readiness()
    intent_info = intent_artifact_status()
    intent_req = config.intent_required_for_ready()
    rag_ready = bool(rag["ready"])
    intent_ok = bool(intent_info.get("present"))
    ready = rag_ready and (not intent_req or intent_ok)
    return {
        "ready": ready,
        "rag": rag,
        "intent": intent_info,
        "intent_required_for_ready": intent_req,
    }


def record_ask(duration_s: float, outcome: str) -> None:
    """outcome: ok | error | blocked | no_context"""
    _ASK_SECONDS.observe(duration_s)
    _ASK_OUTCOMES.labels(outcome=outcome).inc()


def record_rebuild(duration_s: float) -> None:
    _REBUILD_SECONDS.observe(duration_s)


def record_intent(duration_s: float) -> None:
    _INTENT_SECONDS.observe(duration_s)


def record_feedback_metric(rating: str) -> None:
    _FEEDBACK_TOTAL.labels(rating=rating).inc()


def prometheus_metrics_body() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


def log_ask_event(
    *,
    request_id: str,
    duration_s: float,
    outcome: str,
    error_code: str | None,
    question_preview: str,
) -> None:
    logger.info(
        "ask_complete outcome=%s duration_s=%.3f error_code=%s q_preview=%s",
        outcome,
        duration_s,
        error_code or "",
        question_preview[:200],
        extra={"request_id": request_id},
    )
