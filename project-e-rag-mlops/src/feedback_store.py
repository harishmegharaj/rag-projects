"""SQLite-backed feedback for thumbs and optional corrections."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from sqlalchemy import DateTime, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.config import FEEDBACK_DB


class Base(DeclarativeBase):
    pass


class FeedbackRow(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    rating: Mapped[str] = mapped_column(String(8))
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer_preview: Mapped[str | None] = mapped_column(Text, nullable=True)
    correction: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


_engine = None
_Session = None


def _session_factory():
    global _engine, _Session
    if _Session is None:
        FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{FEEDBACK_DB}", future=True)
        Base.metadata.create_all(_engine)
        _Session = sessionmaker(_engine, expire_on_commit=False)
    return _Session()


def record_feedback(
    *,
    request_id: str,
    rating: Literal["up", "down"],
    question: str | None = None,
    answer_preview: str | None = None,
    correction: str | None = None,
) -> str:
    sid = str(uuid.uuid4())
    row = FeedbackRow(
        id=sid,
        request_id=request_id,
        rating=rating,
        question=question,
        answer_preview=answer_preview,
        correction=correction,
        created_at=datetime.now(timezone.utc),
    )
    session = _session_factory()
    with session.begin():
        session.add(row)
    return sid
