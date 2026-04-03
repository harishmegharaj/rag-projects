"""Request/response models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.summarize import Strategy


class SummarizeTextBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=2_000_000)
    title: str | None = Field(None, max_length=512)
    strategy: Strategy = "auto"


class SummarizeResponse(BaseModel):
    request_id: str
    summary: str
    strategy_used: str
    llm_mode: str
    chunk_count: int
    source_filename: str | None = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: Literal["pending"]


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    filename: str | None
    strategy: str | None
    callback_url: str | None
    result: dict | None
    error: str | None
