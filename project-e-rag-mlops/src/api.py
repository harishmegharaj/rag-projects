"""FastAPI RAG service with metrics and feedback."""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.feedback_store import record_feedback
from src.metrics import (
    LLM_CALLS,
    LLM_TOKENS_ESTIMATE,
    RAG_ERRORS,
    RAG_RETRIEVAL_CHUNKS,
    RAG_RETRIEVAL_HIT,
    REQUEST_LATENCY,
)
from src.rag_core import RAGResult, ask, build_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.config import CHUNKS_PATH, VECTORIZER_PATH

    if not (VECTORIZER_PATH.is_file() and CHUNKS_PATH.is_file()):
        build_index()
    yield


app = FastAPI(title="Project E — RAG MLOps", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)


class AskResponse(BaseModel):
    request_id: str
    answer: str
    retrieval_hits: int
    chunk_sources: list[str]
    llm_mode: str
    latency_ms: float


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=4, max_length=64)
    rating: Literal["up", "down"]
    question: str | None = None
    answer_preview: str | None = None
    correction: str | None = Field(None, max_length=8000)


def _observe_rag(result: RAGResult) -> None:
    RAG_RETRIEVAL_CHUNKS.observe(result.retrieval_hits)
    RAG_RETRIEVAL_HIT.labels(hit="1" if result.retrieval_hits > 0 else "0").inc()
    LLM_TOKENS_ESTIMATE.inc(result.token_estimate)
    LLM_CALLS.labels(mode=result.llm_mode).inc()


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        path = request.url.path
        if path.startswith("/v1/") or path == "/health":
            REQUEST_LATENCY.labels(route=path).observe(time.perf_counter() - t0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/ask", response_model=AskResponse)
def rag_ask(body: AskRequest):
    rid = uuid.uuid4().hex[:16]
    t0 = time.perf_counter()
    try:
        result = ask(body.question)
    except Exception:
        RAG_ERRORS.labels(stage="ask").inc()
        raise
    _observe_rag(result)
    ms = (time.perf_counter() - t0) * 1000
    return AskResponse(
        request_id=rid,
        answer=result.answer,
        retrieval_hits=result.retrieval_hits,
        chunk_sources=list({c.get("source", "?") for c in result.chunks}),
        llm_mode=result.llm_mode,
        latency_ms=round(ms, 3),
    )


@app.post("/v1/feedback")
def feedback(body: FeedbackRequest):
    try:
        fid = record_feedback(
            request_id=body.request_id,
            rating=body.rating,
            question=body.question,
            answer_preview=body.answer_preview,
            correction=body.correction,
        )
    except Exception:
        RAG_ERRORS.labels(stage="feedback").inc()
        raise
    return {"id": fid, "status": "stored"}


def main():
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8010"))
    uvicorn.run("src.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
