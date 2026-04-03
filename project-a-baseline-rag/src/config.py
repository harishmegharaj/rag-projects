"""Environment-backed settings (vector backend, paths). Loads project `.env` from package root."""
import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


def vector_backend() -> str:
    v = os.getenv("VECTOR_BACKEND", "chroma").strip().lower()
    return v if v in ("chroma", "pinecone") else "chroma"


def pinecone_api_key() -> str | None:
    return os.getenv("PINECONE_API_KEY")


def pinecone_index_name() -> str:
    return os.getenv("PINECONE_INDEX_NAME", "baseline-rag")


def pinecone_namespace() -> str:
    return os.getenv("PINECONE_NAMESPACE", "")


def chroma_persist_dir(project_root: Path) -> Path:
    override = os.getenv("CHROMA_PERSIST_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (project_root / "vector_store" / "chroma").resolve()
