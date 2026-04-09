"""Paths and environment. Loads project `.env` from package root."""
import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


def project_root() -> Path:
    return _PROJECT_ROOT


def chroma_persist_dir() -> Path:
    override = os.getenv("CHROMA_PERSIST_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (_PROJECT_ROOT / "vector_store" / "chroma").resolve()


def bm25_index_dir() -> Path:
    override = os.getenv("BM25_INDEX_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (_PROJECT_ROOT / "vector_store" / "bm25").resolve()


def documents_raw_dir() -> Path:
    """Directory where PDF/Markdown corpus files live (upload API + manual drops)."""
    override = os.getenv("DOCUMENTS_RAW_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (_PROJECT_ROOT / "data" / "raw").resolve()


def api_key() -> str | None:
    v = os.getenv("API_KEY", "").strip()
    return v or None


def openai_chat_model() -> str:
    return os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def sql_database_url() -> str | None:
    v = os.getenv("SQL_DATABASE_URL", "").strip()
    return v or None


def sql_sync_query() -> str:
    return os.getenv("SQL_SYNC_QUERY", "SELECT * FROM documents")


def sql_id_column() -> str:
    return os.getenv("SQL_ID_COLUMN", "id").strip() or "id"


def sql_updated_at_column() -> str:
    return os.getenv("SQL_UPDATED_AT_COLUMN", "updated_at").strip() or "updated_at"


def sql_source_name() -> str:
    return os.getenv("SQL_SOURCE_NAME", "sql_db").strip() or "sql_db"


def sql_text_columns() -> list[str]:
    raw = os.getenv("SQL_TEXT_COLUMNS", "").strip()
    if not raw:
        return []
    return [c.strip() for c in raw.split(",") if c.strip()]
