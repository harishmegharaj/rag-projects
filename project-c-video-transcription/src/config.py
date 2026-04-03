"""Paths and environment. Loads project `.env` from package root."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


def project_root() -> Path:
    return _PROJECT_ROOT


def api_key() -> str | None:
    v = os.getenv("API_KEY", "").strip()
    return v or None


def database_url() -> str:
    override = os.getenv("DATABASE_URL", "").strip()
    if override:
        return override
    data = _PROJECT_ROOT / "data"
    data.mkdir(parents=True, exist_ok=True)
    db_path = (data / "app.sqlite3").resolve()
    return f"sqlite:///{db_path}"


def video_storage_dir() -> Path:
    override = os.getenv("VIDEO_STORAGE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    d = (_PROJECT_ROOT / "data" / "videos").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def openai_api_key() -> str | None:
    v = os.getenv("OPENAI_API_KEY", "").strip()
    return v or None


def deepgram_api_key() -> str | None:
    v = os.getenv("DEEPGRAM_API_KEY", "").strip()
    return v or None


def deepgram_model() -> str:
    return os.getenv("DEEPGRAM_MODEL", "nova-2").strip() or "nova-2"


def transcription_backend() -> str:
    """
    openai — Whisper HTTP API (needs OPENAI_API_KEY).
    local — faster-whisper on this machine (see requirements-local.txt).
    deepgram — Deepgram prerecorded API with speaker diarization (needs DEEPGRAM_API_KEY).
    auto — use openai if OPENAI_API_KEY is set, otherwise local.
    """
    raw = os.getenv("TRANSCRIPTION_BACKEND", "auto").strip().lower()
    if raw == "openai":
        return "openai"
    if raw == "local":
        return "local"
    if raw == "deepgram":
        return "deepgram"
    if raw == "auto" or not raw:
        return "openai" if openai_api_key() else "local"
    return "openai" if openai_api_key() else "local"


def whisper_model() -> str:
    return os.getenv("WHISPER_MODEL", "whisper-1").strip() or "whisper-1"


def transcribe_segment_seconds() -> int:
    v = os.getenv("TRANSCRIBE_SEGMENT_SECONDS", "480")
    try:
        n = int(v)
        return max(60, min(n, 3600))
    except ValueError:
        return 480


def max_upload_bytes() -> int:
    v = os.getenv("MAX_UPLOAD_BYTES", "524288000")
    try:
        return max(1_048_576, int(v))
    except ValueError:
        return 524_288_000


def local_whisper_model() -> str:
    """faster-whisper size name: tiny, base, small, medium, large-v2, large-v3, …"""
    return os.getenv("LOCAL_WHISPER_MODEL", "base").strip() or "base"


def local_whisper_device() -> str:
    return os.getenv("LOCAL_WHISPER_DEVICE", "cpu").strip() or "cpu"


def local_whisper_compute_type() -> str:
    """e.g. int8 (CPU), float16 (GPU cuda), int8_float16"""
    return os.getenv("LOCAL_WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
