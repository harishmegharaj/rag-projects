"""Rebuild Chroma + BM25 from a directory of PDF/Markdown files (used by CLI and HTTP API)."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TypedDict

from .bm25_index import build_bm25, load_index, save_index
from .config import (
    bm25_index_dir,
    chroma_persist_dir,
    documents_raw_dir,
    sql_database_url,
    sql_id_column,
    sql_source_name,
    sql_sync_query,
    sql_text_columns,
    sql_updated_at_column,
)
from .embed_store import COLLECTION, build_collection, get_chroma, upsert_chunks
from .ingest import ingest_directory
from .sql_ingest import ingest_sql_rows

_rebuild_lock = threading.Lock()


class IndexBuildResult(TypedDict):
    chunk_count: int
    file_chunk_count: int
    sql_chunk_count: int
    sql_row_count: int
    sql_enabled: bool
    mode: str
    no_change: bool
    chroma_dir: str
    bm25_dir: str
    raw_dir: str


def _sql_state_file_path(bm25_path: Path) -> Path:
    return bm25_path / "sql_sync_state.json"


def _load_sql_state(bm25_path: Path) -> dict:
    path = _sql_state_file_path(bm25_path)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_sql_state(bm25_path: Path, state: dict) -> None:
    bm25_path.mkdir(parents=True, exist_ok=True)
    _sql_state_file_path(bm25_path).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _is_sql_chunk(chunk: dict) -> bool:
    metadata = chunk.get("metadata") or {}
    source = str(metadata.get("source", ""))
    return source.startswith("sql:")


def _count_file_chunks(chunks: list[dict]) -> int:
    return sum(1 for c in chunks if not _is_sql_chunk(c))


def _count_sql_chunks(chunks: list[dict]) -> int:
    return sum(1 for c in chunks if _is_sql_chunk(c))


def rebuild_index(
    raw_dir: Path | None = None,
    *,
    include_sql: bool = True,
    require_sql: bool = False,
) -> IndexBuildResult:
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
        file_chunks = ingest_directory(raw)
        chunks = list(file_chunks)

        sql_chunk_count = 0
        sql_row_count = 0
        sql_enabled = False
        db_url = sql_database_url()
        if include_sql and db_url:
            sql_enabled = True
            sql_result = ingest_sql_rows(
                database_url=db_url,
                query=sql_sync_query(),
                id_column=sql_id_column(),
                updated_at_column=sql_updated_at_column(),
                source_name=sql_source_name(),
                text_columns=sql_text_columns(),
            )
            sql_chunks = sql_result["chunks"]
            sql_row_count = sql_result["row_count"]
            sql_chunk_count = len(sql_chunks)
            chunks.extend(sql_chunks)

            max_updated_at = sql_result.get("max_updated_at")
            if max_updated_at:
                state = _load_sql_state(bm25_path)
                state[sql_source_name()] = {"last_synced_updated_at": max_updated_at}
                _save_sql_state(bm25_path, state)
        elif include_sql and require_sql:
            raise ValueError("SQL sync requested but SQL_DATABASE_URL is not configured")

        if not chunks:
            raise ValueError(
                f"No chunks available from files or SQL. Add .pdf/.md files under {raw}, or configure SQL sync."
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
            file_chunk_count=len(file_chunks),
            sql_chunk_count=sql_chunk_count,
            sql_row_count=sql_row_count,
            sql_enabled=sql_enabled,
            mode="full",
            no_change=False,
            chroma_dir=str(chroma_dir),
            bm25_dir=str(bm25_path),
            raw_dir=str(raw),
        )


def sync_sql_incremental(raw_dir: Path | None = None) -> IndexBuildResult:
    """
    Incremental SQL sync: fetch changed rows since last sync watermark and upsert only those.
    Requires an existing index (BM25 + chunks) to avoid silent partial corpora.
    """
    raw = (raw_dir or documents_raw_dir()).resolve()
    chroma_dir = chroma_persist_dir()
    bm25_path = bm25_index_dir()

    if not raw.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {raw}")

    db_url = sql_database_url()
    if not db_url:
        raise ValueError("SQL sync requested but SQL_DATABASE_URL is not configured")

    chunks_file = bm25_path / "chunks.json"
    bm25_file = bm25_path / "bm25.pkl"
    if not (chunks_file.is_file() and bm25_file.is_file()):
        raise ValueError("Incremental SQL sync requires an existing index. Run full rebuild first.")

    source_name = sql_source_name()
    source_tag = f"sql:{source_name}"

    with _rebuild_lock:
        _, existing_chunks = load_index(bm25_path)

        state = _load_sql_state(bm25_path)
        source_state = state.get(source_name) or {}
        updated_after = source_state.get("last_synced_updated_at")
        if isinstance(updated_after, str):
            updated_after = updated_after.strip() or None
        else:
            updated_after = None

        sql_result = ingest_sql_rows(
            database_url=db_url,
            query=sql_sync_query(),
            id_column=sql_id_column(),
            updated_at_column=sql_updated_at_column(),
            source_name=source_name,
            text_columns=sql_text_columns(),
            updated_after=updated_after,
        )
        sql_chunks = sql_result["chunks"]
        changed_row_ids = set(sql_result["row_ids"])

        if not changed_row_ids:
            return IndexBuildResult(
                chunk_count=len(existing_chunks),
                file_chunk_count=_count_file_chunks(existing_chunks),
                sql_chunk_count=0,
                sql_row_count=0,
                sql_enabled=True,
                mode="incremental_sql",
                no_change=True,
                chroma_dir=str(chroma_dir),
                bm25_dir=str(bm25_path),
                raw_dir=str(raw),
            )

        ids_to_delete: list[str] = []
        kept_chunks: list[dict] = []
        for chunk in existing_chunks:
            metadata = chunk.get("metadata") or {}
            if metadata.get("source") == source_tag and str(metadata.get("sql_id")) in changed_row_ids:
                ids_to_delete.append(chunk["id"])
                continue
            kept_chunks.append(chunk)

        merged_chunks = kept_chunks + sql_chunks
        if not merged_chunks:
            raise ValueError("Incremental SQL sync produced an empty corpus")

        client = get_chroma(chroma_dir)
        try:
            collection = client.get_collection(COLLECTION)
        except Exception:
            collection = build_collection(client)

        if ids_to_delete:
            try:
                collection.delete(ids=ids_to_delete)
            except Exception:
                pass
        if sql_chunks:
            upsert_chunks(collection, sql_chunks)

        bm25 = build_bm25(merged_chunks)
        save_index(merged_chunks, bm25, bm25_path)

        max_updated_at = sql_result.get("max_updated_at")
        if max_updated_at:
            state[source_name] = {"last_synced_updated_at": max_updated_at}
            _save_sql_state(bm25_path, state)

        return IndexBuildResult(
            chunk_count=len(merged_chunks),
            file_chunk_count=_count_file_chunks(merged_chunks),
            sql_chunk_count=len(sql_chunks),
            sql_row_count=sql_result["row_count"],
            sql_enabled=True,
            mode="incremental_sql",
            no_change=False,
            chroma_dir=str(chroma_dir),
            bm25_dir=str(bm25_path),
            raw_dir=str(raw),
        )
