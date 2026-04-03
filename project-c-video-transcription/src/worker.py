"""Background worker: pending videos → transcription → persisted transcript."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import SessionLocal
from .models import Video, VideoStatus
from .transcribe import TranscriptionError, transcribe_video_path

logger = logging.getLogger(__name__)


def process_one(session: Session) -> bool:
    video = session.scalar(
        select(Video)
        .where(Video.status == VideoStatus.pending.value)
        .order_by(Video.created_at)
        .limit(1)
    )
    if video is None:
        return False

    video.status = VideoStatus.processing.value
    video.error_message = None
    session.commit()

    path = Path(video.storage_path)
    try:
        out = transcribe_video_path(path)
    except TranscriptionError as e:
        video.status = VideoStatus.failed.value
        video.error_message = str(e)[:8000]
        session.commit()
        logger.warning("Transcription failed for video_id=%s: %s", video.id, e)
        return True
    except Exception as e:  # noqa: BLE001 — persist failure, do not crash worker
        video.status = VideoStatus.failed.value
        video.error_message = str(e)[:8000]
        session.commit()
        logger.exception("Unexpected error for video_id=%s", video.id)
        return True

    video.transcript_text = out.get("text")
    video.transcript_segments = out.get("segments")
    video.transcript_language = out.get("language")
    video.whisper_model = out.get("model")
    video.status = VideoStatus.completed.value
    session.commit()
    return True


async def worker_loop(poll_interval: float = 2.0) -> None:
    while True:
        try:

            def _job() -> bool:
                s = SessionLocal()
                try:
                    return process_one(s)
                finally:
                    s.close()

            did = await asyncio.to_thread(_job)
            if not did:
                await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            break
