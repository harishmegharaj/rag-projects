"""HTTP API: enterprise RAG (Project B) + intent classifier (Project D) + feedback (Project E)."""
from __future__ import annotations

import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from .config import (
    FEEDBACK_DB,
    bm25_index_dir,
    chroma_persist_dir,
    documents_raw_dir,
    intent_registry_path,
    models_dir,
)
from .config import (
    api_key as config_api_key,
)
from .feedback_store import record_feedback
from .index_builder import rebuild_index
from .intent_model import intent_artifact_status, predict_intent
from .observability import (
    build_id,
    log_ask_event,
    metrics_require_auth,
    prometheus_metrics_body,
    record_ask,
    record_feedback_metric,
    record_intent,
    record_rebuild,
    service_readiness,
    service_version,
    setup_logging,
)
from .rag_pipeline import debug_json, run_pipeline

ALLOWED_SUFFIXES = frozenset({".pdf", ".md", ".markdown"})
_UNSAFE = re.compile(r"[^\w.\- ()\[\]]")


def _sanitize_stem(name: str, max_len: int = 120) -> str:
    base = Path(name).name
    stem = Path(base).stem
    stem = _UNSAFE.sub("_", stem).strip("._") or "document"
    return stem[:max_len]


async def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    expected = config_api_key()
    if not expected:
        return
    token: str | None = None
    if x_api_key:
        token = x_api_key.strip()
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = app
    setup_logging()
    yield


app = FastAPI(title="Unified RAG + MLOps", version=service_version(), lifespan=lifespan)

app.add_middleware(RequestIDMiddleware)


@app.get("/health")
async def health() -> dict:
    """Liveness: process is up (use for Kubernetes livenessProbe)."""
    return {"status": "ok", "version": service_version(), "build_id": build_id()}


@app.get("/ready")
async def ready() -> dict:
    """Readiness: RAG indexes; optional intent model if INTENT_REQUIRED_FOR_READY=1."""
    info = service_readiness()
    if not info["ready"]:
        return JSONResponse(status_code=503, content={"status": "not_ready", **info})
    return {"status": "ready", **info}


@app.get("/v1/version")
async def version_info(_: None = Depends(verify_api_key)) -> dict:
    """Build identity for tracing deploys to metrics and logs."""
    return {
        "service": "unified-rag-mlops",
        "version": service_version(),
        "build_id": build_id(),
    }


@app.get("/metrics")
async def metrics(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> Response:
    """Prometheus scrape endpoint. Set METRICS_REQUIRE_AUTH=1 to require API key."""
    if metrics_require_auth():
        await verify_api_key(x_api_key, authorization)
    body, ctype = prometheus_metrics_body()
    return Response(content=body, media_type=ctype)


@app.get("/v1/config/paths")
async def paths(_: None = Depends(verify_api_key)) -> dict:
    return {
        "documents_raw_dir": str(documents_raw_dir()),
        "chroma_persist_dir": str(chroma_persist_dir()),
        "bm25_index_dir": str(bm25_index_dir()),
        "models_dir": str(models_dir()),
        "intent_registry_path": str(intent_registry_path()),
        "feedback_db": str(FEEDBACK_DB),
    }


class AskBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)


class FeedbackBody(BaseModel):
    request_id: str = Field(..., min_length=4, max_length=64)
    rating: Literal["up", "down"]
    question: str | None = None
    answer_preview: str | None = None
    correction: str | None = Field(None, max_length=8000)


class IntentPredictBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


def _ask_payload(out: dict, request_id: str, intent: dict | None) -> dict:
    payload = {
        "request_id": request_id,
        "answer": out["answer"],
        "blocked": out.get("blocked", False),
        "no_context": out.get("no_context", False),
        "error": out.get("error", False),
        "error_code": out.get("error_code"),
        "intent": intent,
        "retrieved": [
            {
                "metadata": h.get("metadata"),
                "rerank_score": h.get("rerank_score"),
                "text_preview": (h.get("text") or "")[:500],
            }
            for h in out.get("retrieved", [])
        ],
        "debug_json": debug_json(out),
    }
    return payload


def _ask_outcome(out: dict) -> str:
    if out.get("blocked"):
        return "blocked"
    if out.get("error"):
        return "error"
    if out.get("no_context"):
        return "no_context"
    return "ok"


@app.post("/v1/ask")
async def ask(body: AskBody, request: Request, _: None = Depends(verify_api_key)):
    rid = getattr(request.state, "request_id", "-")
    t_intent0 = time.perf_counter()
    intent_out = predict_intent(body.question.strip())
    record_intent(time.perf_counter() - t_intent0)

    t0 = time.perf_counter()
    out = run_pipeline(body.question.strip(), chroma_persist_dir(), bm25_index_dir())
    duration = time.perf_counter() - t0
    outcome = _ask_outcome(out)
    record_ask(duration, outcome)
    log_ask_event(
        request_id=rid,
        duration_s=duration,
        outcome=outcome,
        error_code=out.get("error_code"),
        question_preview=body.question.strip(),
    )
    body_out = _ask_payload(out, rid, intent_out)
    if out.get("error"):
        return JSONResponse(status_code=503, content=body_out)
    return body_out


@app.post("/v1/feedback")
async def feedback(body: FeedbackBody, _: None = Depends(verify_api_key)):
    try:
        fid = record_feedback(
            request_id=body.request_id,
            rating=body.rating,
            question=body.question,
            answer_preview=body.answer_preview,
            correction=body.correction,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    record_feedback_metric(body.rating)
    return {"id": fid, "status": "stored"}


@app.post("/v1/intent/predict")
async def intent_predict(body: IntentPredictBody, _: None = Depends(verify_api_key)):
    t0 = time.perf_counter()
    pred = predict_intent(body.text.strip())
    record_intent(time.perf_counter() - t0)
    if pred is None:
        raise HTTPException(
            status_code=503,
            detail="Intent model not loaded. Run `python scripts/train_intent.py` or set INTENT_REGISTRY_PATH.",
        )
    return pred


@app.get("/v1/intent/model")
async def intent_model_info(_: None = Depends(verify_api_key)):
    st = intent_artifact_status()
    if not st.get("present"):
        return {"active": False, **st}
    reg_path = intent_registry_path()
    meta: dict | None = None
    if reg_path.is_file():
        meta = json.loads(reg_path.read_text(encoding="utf-8"))
    return {
        "active": True,
        "active_version": meta.get("active_version") if meta else None,
        "active_model_path": meta.get("active_model_path") if meta else None,
        **st,
    }


@app.post("/v1/documents")
async def upload_documents(
    files: list[UploadFile] = File(...),
    rebuild: bool = Query(True, description="Run full reindex after saving uploads"),
    _: None = Depends(verify_api_key),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    dest = documents_raw_dir()
    dest.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for f in files:
        name = f.filename or "unnamed"
        suf = Path(name).suffix.lower()
        if suf == ".markdown":
            suf = ".md"
        if suf not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type {suf!r} for {name}. Use .pdf or .md.",
            )
        stem = _sanitize_stem(name)
        final_name = f"{uuid.uuid4().hex[:10]}_{stem}{suf}"
        out_path = dest / final_name
        data = await f.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"Empty file: {name}")
        out_path.write_bytes(data)
        saved.append(str(out_path.relative_to(dest)))

    result: dict = {"saved": saved, "directory": str(dest)}
    if rebuild:
        t0 = time.perf_counter()
        try:
            built = rebuild_index(dest)
            result["index"] = dict(built)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        record_rebuild(time.perf_counter() - t0)
    else:
        result["note"] = "Rebuild skipped; call POST /v1/index/rebuild when ready."

    return result


@app.post("/v1/index/rebuild")
async def index_rebuild(_: None = Depends(verify_api_key)) -> dict:
    t0 = time.perf_counter()
    try:
        built = rebuild_index()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    record_rebuild(time.perf_counter() - t0)
    return {"index": dict(built)}
