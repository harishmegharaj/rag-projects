"""Rebuild Chroma + BM25 from a directory of PDF/Markdown files (used by CLI and HTTP API)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import TypedDict

from .bm25_index import build_bm25, save_index
from .config import bm25_index_dir, chroma_persist_dir, documents_raw_dir
from .embed_store import COLLECTION, build_collection, get_chroma, upsert_chunks
from .ingest import ingest_directory

_rebuild_lock = threading.Lock()


class IndexBuildResult(TypedDict):
    chunk_count: int
    chroma_dir: str
    bm25_dir: str
    raw_dir: str


def rebuild_index(raw_dir: Path | None = None) -> IndexBuildResult:
    """
    Full reindex: delete Chroma collection, re-embed all chunks, rebuild BM25.
    Thread-safe: concurrent callers block (one rebuild at a time).
    """
    raw = (raw_dir or documents_raw_dir()).resolve()
    chroma_dir = chroma_persist_dir()
    bm25_path = bm25_index_dir()

    if not raw.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {raw}")

    with _rebuild_lock:
        chunks = ingest_directory(raw)
        if not chunks:
            raise ValueError(
                f"No PDF/Markdown chunks under {raw}. Add .pdf or .md files, or upload via the API."
            )

        client = get_chroma(chroma_dir)
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        col = build_collection(client)
        upsert_chunks(col, chunks)

        bm25 = build_bm25(chunks)
        save_index(chunks, bm25, bm25_path)

        return IndexBuildResult(
            chunk_count=len(chunks),
            chroma_dir=str(chroma_dir),
            bm25_dir=str(bm25_path),
            raw_dir=str(raw),
        )
