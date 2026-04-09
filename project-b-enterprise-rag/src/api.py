"""HTTP API: upload documents, rebuild index, ask questions."""
from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.websockets import WebSocketDisconnect

from .config import api_key as config_api_key
from .config import bm25_index_dir, chroma_persist_dir, documents_raw_dir
from .config import sql_database_url, sql_sync_query
from .index_builder import rebuild_index, sync_sql_incremental
from .observability import (
    build_id,
    get_langfuse_callbacks,
    index_readiness,
    langfuse_state,
    langsmith_state,
    log_ask_event,
    metrics_require_auth,
    prometheus_metrics_body,
    record_ask,
    record_rebuild,
    service_version,
    setup_langfuse,
    setup_langsmith,
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


def verify_ws_api_key(websocket: WebSocket) -> bool:
    expected = config_api_key()
    if not expected:
        return True

    token: str | None = websocket.headers.get("x-api-key")
    if token:
        token = token.strip()
    if not token:
        auth = websocket.headers.get("authorization")
        if auth and auth.lower().startswith("bearer "):
            token = auth[7:].strip()
    if not token:
        token = (websocket.query_params.get("api_key") or "").strip() or None
    return token == expected


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


app = FastAPI(title="Enterprise RAG", version=service_version())


@app.on_event("startup")
def _startup() -> None:
    setup_logging()
    setup_langsmith()
    setup_langfuse()


app.add_middleware(RequestIDMiddleware)


@app.get("/health")
async def health() -> dict:
    """Liveness: process is up (use for Kubernetes livenessProbe)."""
    return {"status": "ok", "version": service_version(), "build_id": build_id()}


@app.get("/ready")
async def ready() -> dict:
    """Readiness: indexes look usable (use for readinessProbe)."""
    info = index_readiness()
    if not info["ready"]:
        return JSONResponse(status_code=503, content={"status": "not_ready", **info})
    return {"status": "ready", **info}


@app.get("/v1/version")
async def version_info(_: None = Depends(verify_api_key)) -> dict:
    """Build identity for tracing deploys to metrics and logs."""
    return {
        "service": "enterprise-rag",
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
        "sql_enabled": bool(sql_database_url()),
        "sql_sync_query": sql_sync_query(),
    }


@app.get("/v1/config/tracing")
async def tracing_config(_: None = Depends(verify_api_key)) -> dict:
    return {
        "langsmith": langsmith_state(),
        "langfuse": langfuse_state(),
    }


class AskBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)


def _ask_payload(out: dict, request_id: str) -> dict:
    payload = {
        "request_id": request_id,
        "answer": out["answer"],
        "blocked": out.get("blocked", False),
        "no_context": out.get("no_context", False),
        "error": out.get("error", False),
        "error_code": out.get("error_code"),
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


async def _ws_send(websocket: WebSocket, payload: dict) -> None:
    await websocket.send_json(payload)


@app.post("/v1/ask")
async def ask(body: AskBody, request: Request, _: None = Depends(verify_api_key)):
    rid = getattr(request.state, "request_id", "-")
    callbacks, langfuse_trace_id = get_langfuse_callbacks(request_id=rid)
    t0 = time.perf_counter()
    out = run_pipeline(body.question.strip(), chroma_persist_dir(), bm25_index_dir(), callbacks=callbacks)
    duration = time.perf_counter() - t0
    outcome = _ask_outcome(out)
    record_ask(duration, outcome)
    log_ask_event(
        request_id=rid,
        langfuse_trace_id=langfuse_trace_id,
        duration_s=duration,
        outcome=outcome,
        error_code=out.get("error_code"),
        question_preview=body.question.strip(),
    )
    body_out = _ask_payload(out, rid)
    if out.get("error"):
        return JSONResponse(status_code=503, content=body_out)
    return body_out


@app.websocket("/v1/ws/ask")
async def ask_ws(websocket: WebSocket) -> None:
    if not verify_ws_api_key(websocket):
        await websocket.close(code=1008, reason="Invalid or missing API key")
        return

    await websocket.accept()
    rid = websocket.headers.get("x-request-id") or str(uuid.uuid4())
    await _ws_send(websocket, {"type": "ack", "request_id": rid})

    try:
        while True:
            msg = await websocket.receive_json()
            question = str(msg.get("question", "")).strip()
            if not question:
                await _ws_send(
                    websocket,
                    {
                        "type": "error",
                        "request_id": rid,
                        "error": "question is required",
                        "max_length": 8000,
                    },
                )
                continue
            if len(question) > 8000:
                await _ws_send(
                    websocket,
                    {
                        "type": "error",
                        "request_id": rid,
                        "error": "question exceeds max_length",
                        "max_length": 8000,
                    },
                )
                continue

            callbacks, langfuse_trace_id = get_langfuse_callbacks(request_id=rid)
            t0 = time.perf_counter()
            await _ws_send(websocket, {"type": "status", "request_id": rid, "stage": "retrieval_started"})
            out = run_pipeline(question, chroma_persist_dir(), bm25_index_dir(), callbacks=callbacks)
            await _ws_send(websocket, {"type": "status", "request_id": rid, "stage": "retrieval_finished"})

            duration = time.perf_counter() - t0
            outcome = _ask_outcome(out)
            record_ask(duration, outcome)
            log_ask_event(
                request_id=rid,
                langfuse_trace_id=langfuse_trace_id,
                duration_s=duration,
                outcome=outcome,
                error_code=out.get("error_code"),
                question_preview=question,
            )

            payload = _ask_payload(out, rid)
            payload["type"] = "final"
            await _ws_send(websocket, payload)
    except WebSocketDisconnect:
        return


@app.post("/v1/documents")
async def upload_documents(
    files: list[UploadFile] = File(...),
    rebuild: bool = Query(True, description="Run full reindex after saving uploads"),
    include_sql: bool = Query(True, description="Include SQL rows (if configured) in reindex"),
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
            built = rebuild_index(dest, include_sql=include_sql)
            result["index"] = dict(built)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        record_rebuild(time.perf_counter() - t0)
    else:
        result["note"] = "Rebuild skipped; call POST /v1/index/rebuild when ready."

    return result


@app.post("/v1/index/rebuild")
async def index_rebuild(
    include_sql: bool = Query(True, description="Include SQL rows (if configured)"),
    _: None = Depends(verify_api_key),
) -> dict:
    t0 = time.perf_counter()
    try:
        built = rebuild_index(include_sql=include_sql)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    record_rebuild(time.perf_counter() - t0)
    return {"index": dict(built)}


@app.post("/v1/db/sync")
async def db_sync(
    mode: str = Query("incremental", description="Sync mode: incremental or full"),
    _: None = Depends(verify_api_key),
) -> dict:
    """Live-pull SQL rows and sync index with DB data."""
    t0 = time.perf_counter()
    try:
        m = mode.strip().lower()
        if m == "incremental":
            built = sync_sql_incremental()
        elif m == "full":
            built = rebuild_index(include_sql=True, require_sql=True)
        else:
            raise HTTPException(status_code=400, detail="mode must be 'incremental' or 'full'")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    record_rebuild(time.perf_counter() - t0)
    return {
        "synced": True,
        "index": dict(built),
        "note": "SQL rows were synced at request time.",
    }
