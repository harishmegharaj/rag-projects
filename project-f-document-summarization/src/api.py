"""FastAPI: sync summarization, async jobs, webhooks, metrics."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from src.config import Settings, get_settings
from src.extract import SUPPORTED_SUFFIXES, extract_text
from src.jobs_store import create_job, get_job, init_db, update_job
from src.metrics import LLM_MODE, REQUEST_LATENCY, SUMMARY_JOBS, SUMMARY_STRATEGY
from src.schemas import JobCreateResponse, JobStatusResponse, SummarizeResponse, SummarizeTextBody
from src.summarize import Strategy, summarize_text
from src.webhooks import deliver_webhook

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(cfg.jobs_db)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        yield

    app = FastAPI(
        title="Project F — Document summarization",
        version="1.0.0",
        lifespan=lifespan,
    )

    def require_api_key(
        x_api_key: Annotated[str | None, Header()] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        expected = cfg.api_key
        if not expected:
            return
        if x_api_key and x_api_key == expected:
            return
        if authorization and authorization.startswith("Bearer "):
            token = authorization.removeprefix("Bearer ").strip()
            if token == expected:
                return
        raise HTTPException(status_code=401, detail="Unauthorized")

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        t0 = time.perf_counter()
        try:
            return await call_next(request)
        finally:
            path = request.url.path
            if path.startswith("/v1/") or path in ("/health", "/metrics"):
                REQUEST_LATENCY.labels(route=path).observe(time.perf_counter() - t0)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "project-f-document-summarization"}

    @app.get("/metrics")
    def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/v1/summarize/text",
        response_model=SummarizeResponse,
        dependencies=[Depends(require_api_key)],
    )
    def summarize_inline(summarize_in: SummarizeTextBody):
        rid = uuid.uuid4().hex[:16]
        try:
            result = summarize_text(
                summarize_in.text, cfg, strategy=summarize_in.strategy, title=summarize_in.title
            )
        except Exception as e:
            logger.exception("summarize failed")
            raise HTTPException(status_code=500, detail=str(e)) from e
        SUMMARY_STRATEGY.labels(strategy=result.strategy_used).inc()
        LLM_MODE.labels(mode=result.llm_mode).inc()
        return SummarizeResponse(
            request_id=rid,
            summary=result.summary,
            strategy_used=result.strategy_used,
            llm_mode=result.llm_mode,
            chunk_count=result.chunk_count,
            source_filename=None,
        )

    @app.post(
        "/v1/summarize",
        response_model=SummarizeResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def summarize_upload(
        file: UploadFile = File(...),
        strategy: Strategy = Form("auto"),
        title: str | None = Form(None),
    ):
        name = file.filename or "upload"
        suffix = name.lower().split(".")[-1] if "." in name else ""
        if f".{suffix}" not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported type .{suffix}. Allowed: {sorted(SUPPORTED_SUFFIXES)}",
            )
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")
        try:
            text = extract_text(data, name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        rid = uuid.uuid4().hex[:16]
        try:
            result = summarize_text(text, cfg, strategy=strategy, title=title)
        except Exception as e:
            logger.exception("summarize failed")
            raise HTTPException(status_code=500, detail=str(e)) from e
        SUMMARY_STRATEGY.labels(strategy=result.strategy_used).inc()
        LLM_MODE.labels(mode=result.llm_mode).inc()
        return SummarizeResponse(
            request_id=rid,
            summary=result.summary,
            strategy_used=result.strategy_used,
            llm_mode=result.llm_mode,
            chunk_count=result.chunk_count,
            source_filename=name,
        )

    def run_summarize_job(
        job_id: str,
        raw: bytes,
        filename: str,
        strategy: Strategy,
        callback_url: str | None,
    ) -> None:
        update_job(cfg.jobs_db, job_id, status="running")
        try:
            text = extract_text(raw, filename)
            out = summarize_text(text, cfg, strategy=strategy, title=None)
            payload = {
                "job_id": job_id,
                "status": "completed",
                "summary": out.summary,
                "strategy_used": out.strategy_used,
                "llm_mode": out.llm_mode,
                "chunk_count": out.chunk_count,
                "filename": filename,
            }
            update_job(
                cfg.jobs_db,
                job_id,
                status="completed",
                result_json=json.dumps(payload),
            )
            SUMMARY_JOBS.labels(status="completed").inc()
            SUMMARY_STRATEGY.labels(strategy=out.strategy_used).inc()
            LLM_MODE.labels(mode=out.llm_mode).inc()
            if callback_url:
                code, _ = deliver_webhook(callback_url, cfg.webhook_secret, payload)
                if code != 200:
                    logger.warning("Webhook returned %s for job %s", code, job_id)
        except Exception as e:
            logger.exception("Job %s failed", job_id)
            update_job(cfg.jobs_db, job_id, status="failed", error=str(e))
            SUMMARY_JOBS.labels(status="failed").inc()
            if callback_url:
                deliver_webhook(
                    callback_url,
                    cfg.webhook_secret,
                    {"job_id": job_id, "status": "failed", "error": str(e)},
                )

    @app.post(
        "/v1/jobs/summarize",
        response_model=JobCreateResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def enqueue_summarize(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        strategy: Strategy = Form("auto"),
        callback_url: str | None = Form(None),
    ):
        name = file.filename or "upload"
        suffix = name.lower().split(".")[-1] if "." in name else ""
        if f".{suffix}" not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported type .{suffix}. Allowed: {sorted(SUPPORTED_SUFFIXES)}",
            )
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")
        jid = create_job(
            cfg.jobs_db,
            filename=name,
            strategy=strategy,
            callback_url=callback_url,
        )
        background_tasks.add_task(run_summarize_job, jid, data, name, strategy, callback_url)
        SUMMARY_JOBS.labels(status="queued").inc()
        return JobCreateResponse(job_id=jid, status="pending")

    @app.get(
        "/v1/jobs/{job_id}",
        response_model=JobStatusResponse,
        dependencies=[Depends(require_api_key)],
    )
    def job_status(job_id: str):
        rec = get_job(cfg.jobs_db, job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return JobStatusResponse(
            job_id=rec.id,
            status=rec.status,
            filename=rec.filename,
            strategy=rec.strategy,
            callback_url=rec.callback_url,
            result=rec.result,
            error=rec.error,
        )

    return app


# Default app for uvicorn
app = create_app()
