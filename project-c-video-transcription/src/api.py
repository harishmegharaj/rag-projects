"""HTTP API: upload videos, logical indexes, transcription status and text."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import uuid
from collections.abc import Generator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketDisconnect

from .config import api_key as config_api_key
from .config import max_upload_bytes, project_root, video_storage_dir
from .db import SessionLocal, init_db
from .models import MediaIndex, Video, VideoStatus
from .realtime_agent import realtime_agent, realtime_hub
from .sales_insights import analyze_transcript_for_sales, analyze_transcript_for_sales_llm
from .worker import worker_loop

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = frozenset({".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"})
_UNSAFE = re.compile(r"[^\w.\- ()\[\]]")
_INDEX_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._\-]{0,62}$")


def _sanitize_stem(name: str, max_len: int = 120) -> str:
    base = Path(name).name
    stem = Path(base).stem
    stem = _UNSAFE.sub("_", stem).strip("._") or "video"
    return stem[:max_len]


def _validate_index_id(index_id: str) -> str:
    s = index_id.strip()
    if not _INDEX_ID_RE.match(s):
        raise HTTPException(
            status_code=400,
            detail="index_id must match ^[a-zA-Z0-9][a-zA-Z0-9._\\-]{0,62}$",
        )
    return s


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


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


Db = Annotated[Session, Depends(get_db)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    init_db()
    worker_task = asyncio.create_task(worker_loop())
    logger.info("Worker started")
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    logger.info("Worker stopped")


app = FastAPI(
    title="Video Transcription API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/realtime/demo")
async def realtime_demo_page() -> FileResponse:
    page = project_root() / "docs" / "realtime-client.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="Realtime demo page not found")
    return FileResponse(page)


@app.get("/ready")
async def ready() -> dict:
    ffmpeg_ok = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
    if not ffmpeg_ok:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "ffmpeg or ffprobe missing on PATH"},
        )
    return {"status": "ready", "ffmpeg": True}


class IndexCreateBody(BaseModel):
    label: str | None = Field(default=None, max_length=256)


class SalesInsightsBody(BaseModel):
    transcript_text: str = Field(min_length=1)
    top_k: int = Field(default=15, ge=1, le=50)
    use_llm: bool = Field(default=False)
    reasoning_model: str | None = Field(default=None, max_length=80)


class RealtimeMessageBody(BaseModel):
    session_id: str = Field(..., min_length=3, max_length=128)
    message: str = Field(..., min_length=1, max_length=6000)


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


def _realtime_event(session_id: str, result: dict) -> dict:
    return {
        "type": "final",
        "session_id": session_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        **result,
    }


@app.post("/v1/indexes", dependencies=[Depends(verify_api_key)])
def create_index(db: Db, body: IndexCreateBody = Body(default_factory=IndexCreateBody)) -> dict:
    idx_id = str(uuid.uuid4())
    label = body.label
    row = MediaIndex(id=idx_id, label=label)
    db.add(row)
    db.commit()
    return {"index_id": idx_id, "label": label}


@app.get("/v1/indexes/{index_id}", dependencies=[Depends(verify_api_key)])
def get_index(index_id: str, db: Db) -> dict:
    iid = _validate_index_id(index_id)
    index = db.get(MediaIndex, iid)
    if index is None:
        raise HTTPException(status_code=404, detail="Index not found")
    n = db.scalar(select(func.count()).select_from(Video).where(Video.index_id == iid)) or 0
    return {
        "index_id": index.id,
        "label": index.label,
        "video_count": int(n),
        "created_at": index.created_at.isoformat() if index.created_at else None,
    }


@app.get("/v1/indexes/{index_id}/videos", dependencies=[Depends(verify_api_key)])
def list_index_videos(index_id: str, db: Db) -> dict:
    iid = _validate_index_id(index_id)
    index = db.get(MediaIndex, iid)
    if index is None:
        raise HTTPException(status_code=404, detail="Index not found")
    rows = db.scalars(select(Video).where(Video.index_id == iid).order_by(Video.created_at)).all()
    return {
        "index_id": iid,
        "videos": [
            {
                "id": v.id,
                "original_filename": v.original_filename,
                "status": v.status,
                "created_at": v.created_at.isoformat() if v.created_at else None,
                "has_transcript": v.transcript_text is not None,
            }
            for v in rows
        ],
    }


@app.post("/v1/videos", dependencies=[Depends(verify_api_key)])
async def upload_video(
    db: Db,
    file: UploadFile = File(...),
    index_id: str | None = Form(default=None),
    index_label: str | None = Form(default=None),
) -> dict:
    name = file.filename or "unnamed"
    suf = Path(name).suffix.lower()
    if suf not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type {suf!r}. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    max_b = max_upload_bytes()
    chunks: list[bytes] = []
    total = 0
    while True:
        block = await file.read(1024 * 1024)
        if not block:
            break
        total += len(block)
        if total > max_b:
            raise HTTPException(status_code=413, detail=f"File exceeds max_upload_bytes ({max_b})")
        chunks.append(block)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    if index_id is not None and str(index_id).strip():
        iid = _validate_index_id(str(index_id).strip())
        index = db.get(MediaIndex, iid)
        if index is None:
            index = MediaIndex(id=iid, label=(index_label.strip() if index_label else None))
            db.add(index)
            db.commit()
        elif index_label and index_label.strip():
            index.label = index_label.strip()[:256]
            db.commit()
    else:
        iid = str(uuid.uuid4())
        index = MediaIndex(id=iid, label=(index_label.strip() if index_label else None))
        db.add(index)
        db.commit()

    vid = str(uuid.uuid4())
    stem = _sanitize_stem(name)
    final_name = f"{vid}_{stem}{suf}"
    base = video_storage_dir()
    dest_dir = (base / iid).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / final_name
    out_path.write_bytes(data)

    video = Video(
        id=vid,
        index_id=iid,
        original_filename=name,
        storage_path=str(out_path),
        mime_type=file.content_type,
        size_bytes=len(data),
        status=VideoStatus.pending.value,
    )
    db.add(video)
    db.commit()

    return {
        "video_id": vid,
        "index_id": iid,
        "status": video.status,
        "original_filename": name,
        "size_bytes": len(data),
    }


def _format_mmss(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 60:02d}:{s % 60:02d}"


def _segments_have_speakers(segments: list | None) -> bool:
    if not segments:
        return False
    return any(isinstance(s, dict) and s.get("speaker") is not None for s in segments)


def _conversation_text_from_segments(segments: list) -> str | None:
    if not _segments_have_speakers(segments):
        return None
    lines: list[str] = []
    for s in segments:
        if not isinstance(s, dict):
            continue
        text = (s.get("text") or "").strip()
        if not text:
            continue
        sp = s.get("speaker")
        label = s.get("speaker_label") or (
            f"Speaker {int(sp) + 1}" if sp is not None else "?"
        )
        t0 = float(s.get("start") or 0)
        lines.append(f"[{_format_mmss(t0)}] {label}: {text}")
    return "\n".join(lines) if lines else None


def _video_response(v: Video) -> dict:
    return {
        "video_id": v.id,
        "index_id": v.index_id,
        "original_filename": v.original_filename,
        "status": v.status,
        "error_message": v.error_message,
        "transcript_language": v.transcript_language,
        "whisper_model": v.whisper_model,
        "created_at": v.created_at.isoformat() if v.created_at else None,
        "updated_at": v.updated_at.isoformat() if v.updated_at else None,
    }


@app.get("/v1/videos/{video_id}", dependencies=[Depends(verify_api_key)])
def get_video(video_id: str, db: Db) -> dict:
    v = db.get(Video, video_id)
    if v is None:
        raise HTTPException(status_code=404, detail="Video not found")
    out = _video_response(v)
    if v.status == VideoStatus.completed.value and v.transcript_text:
        out["transcript_preview"] = v.transcript_text[:500] + ("…" if len(v.transcript_text) > 500 else "")
    return out


@app.get("/v1/videos/{video_id}/transcript", dependencies=[Depends(verify_api_key)])
def get_transcript(video_id: str, db: Db) -> dict:
    v = db.get(Video, video_id)
    if v is None:
        raise HTTPException(status_code=404, detail="Video not found")
    if v.status == VideoStatus.pending.value or v.status == VideoStatus.processing.value:
        raise HTTPException(status_code=409, detail="Transcript not ready yet")
    if v.status == VideoStatus.failed.value:
        return JSONResponse(
            status_code=502,
            content={
                "video_id": v.id,
                "status": v.status,
                "error_message": v.error_message,
            },
        )
    segments = v.transcript_segments or []
    out: dict = {
        "video_id": v.id,
        "index_id": v.index_id,
        "language": v.transcript_language,
        "model": v.whisper_model,
        "text": v.transcript_text or "",
        "segments": segments,
        "has_speaker_labels": _segments_have_speakers(segments),
    }
    conv = _conversation_text_from_segments(segments)
    if conv:
        out["conversation_text"] = conv
    return out


@app.post("/v1/nlp/sales-insights", dependencies=[Depends(verify_api_key)])
def analyze_sales_text(body: SalesInsightsBody) -> dict:
    if body.use_llm:
        try:
            return analyze_transcript_for_sales_llm(
                body.transcript_text,
                top_k=body.top_k,
                reasoning_model=body.reasoning_model,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
    return analyze_transcript_for_sales(body.transcript_text, top_k=body.top_k)


@app.get("/v1/videos/{video_id}/sales-insights", dependencies=[Depends(verify_api_key)])
def get_video_sales_insights(
    video_id: str,
    db: Db,
    top_k: int = 15,
    use_llm: bool = False,
    reasoning_model: str | None = None,
) -> dict:
    v = db.get(Video, video_id)
    if v is None:
        raise HTTPException(status_code=404, detail="Video not found")
    if v.status == VideoStatus.pending.value or v.status == VideoStatus.processing.value:
        raise HTTPException(status_code=409, detail="Transcript not ready yet")
    if v.status == VideoStatus.failed.value:
        return JSONResponse(
            status_code=502,
            content={
                "video_id": v.id,
                "status": v.status,
                "error_message": v.error_message,
            },
        )
    text = (v.transcript_text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Transcript is empty")
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 50")

    if use_llm:
        try:
            out = analyze_transcript_for_sales_llm(
                text,
                top_k=top_k,
                reasoning_model=reasoning_model,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
    else:
        out = analyze_transcript_for_sales(text, top_k=top_k)

    out["video_id"] = v.id
    out["index_id"] = v.index_id
    out["model_used_for_transcript"] = v.whisper_model
    out["transcript_language"] = v.transcript_language
    return out


@app.post("/v1/realtime/messages", dependencies=[Depends(verify_api_key)])
async def realtime_message(body: RealtimeMessageBody) -> dict:
    result = realtime_agent.respond(body.session_id, body.message.strip())
    event = _realtime_event(body.session_id, result)
    await realtime_hub.publish(body.session_id, event)
    return event


@app.get("/v1/realtime/sse/{session_id}", dependencies=[Depends(verify_api_key)])
async def realtime_sse(session_id: str):
    sid = session_id.strip()
    if len(sid) < 3:
        raise HTTPException(status_code=422, detail="Invalid session_id")

    async def event_gen():
        q = realtime_hub.subscribe(sid, maxsize=200)
        hello = {"type": "ack", "session_id": sid, "transport": "sse"}
        yield f"data: {json.dumps(hello)}\\n\\n"
        try:
            while True:
                ev = await q.get()
                yield f"data: {json.dumps(ev)}\\n\\n"
        finally:
            realtime_hub.unsubscribe(sid, q)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.websocket("/v1/realtime/ws")
async def realtime_ws(websocket: WebSocket) -> None:
    if not verify_ws_api_key(websocket):
        await websocket.close(code=1008, reason="Invalid or missing API key")
        return

    await websocket.accept()
    session_id = (websocket.query_params.get("session_id") or "").strip()
    if len(session_id) < 3:
        await websocket.send_json({"type": "error", "error": "session_id query param required"})
        await websocket.close(code=1008)
        return

    await websocket.send_json({"type": "ack", "session_id": session_id})

    try:
        while True:
            msg = await websocket.receive_json()
            text = str(msg.get("message", "")).strip()
            if not text:
                await websocket.send_json({"type": "error", "error": "message is required"})
                continue
            if len(text) > 6000:
                await websocket.send_json({"type": "error", "error": "message exceeds max_length", "max_length": 6000})
                continue

            await websocket.send_json({"type": "status", "stage": "agent_started"})
            result = realtime_agent.respond(session_id, text)
            event = _realtime_event(session_id, result)
            await websocket.send_json(event)
            await realtime_hub.publish(session_id, event)
    except WebSocketDisconnect:
        return
