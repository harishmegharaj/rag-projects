"""Pull rows from SQL and convert them into chunked docs for indexing."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, text


class SqlIngestResult(TypedDict):
    chunks: list[dict]
    row_count: int
    row_ids: list[str]
    max_updated_at: str | None


def _is_safe_read_query(query: str) -> bool:
    prefix = query.strip().lower()
    return prefix.startswith("select") or prefix.startswith("with")


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _build_row_text(
    row: Mapping[str, Any],
    text_columns: list[str],
    id_column: str,
    updated_at_column: str,
) -> str:
    parts: list[str] = []
    if text_columns:
        for col in text_columns:
            if col in row:
                val = _to_text(row[col])
                if val:
                    parts.append(f"{col}: {val}")
    else:
        for key, value in row.items():
            if key in {id_column, updated_at_column}:
                continue
            val = _to_text(value)
            if val:
                parts.append(f"{key}: {val}")
    return "\n".join(parts).strip()


def ingest_sql_rows(
    *,
    database_url: str,
    query: str,
    id_column: str,
    updated_at_column: str,
    source_name: str,
    text_columns: list[str] | None = None,
    updated_after: str | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> SqlIngestResult:
    """Execute a read query and convert rows to chunk dicts expected by embedding stores."""
    if not database_url.strip():
        raise ValueError("SQL_DATABASE_URL is empty")
    if not _is_safe_read_query(query):
        raise ValueError("SQL sync query must start with SELECT or WITH")

    run_query = query
    params: dict[str, Any] = {}
    if updated_after:
        run_query = (
            "SELECT * FROM ("
            f"{query}"
            f") AS sync_src WHERE sync_src.{updated_at_column} > :updated_after"
        )
        params["updated_after"] = updated_after

    engine = create_engine(database_url, future=True)
    with engine.connect() as conn:
        rows = conn.execute(text(run_query), params).mappings().all()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    source = f"sql:{source_name}"
    docs: list[dict] = []
    cols = text_columns or []
    row_ids: list[str] = []
    max_updated_at: str | None = None

    for row_index, row in enumerate(rows):
        row_id = _to_text(row.get(id_column)) or str(row_index)
        row_ids.append(row_id)
        row_text = _build_row_text(row, cols, id_column, updated_at_column)

        updated = _to_text(row.get(updated_at_column))
        if updated and (max_updated_at is None or updated > max_updated_at):
            max_updated_at = updated

        if not row_text:
            continue

        metadata_base = {
            "source": source,
            "page": None,
            "sql_source": source_name,
            "sql_id": row_id,
            "sql_row_index": row_index,
        }
        if updated:
            metadata_base["sql_updated_at"] = updated

        chunks = splitter.create_documents([row_text], metadatas=[metadata_base])
        for i, chunk in enumerate(chunks):
            start = chunk.metadata.get("start_index", i)
            doc_id = f"{source}:{row_id}:{start}"
            docs.append(
                {
                    "id": doc_id,
                    "text": chunk.page_content,
                    "metadata": {
                        **metadata_base,
                        "start_index": start,
                    },
                }
            )

    return SqlIngestResult(
        chunks=docs,
        row_count=len(rows),
        row_ids=row_ids,
        max_updated_at=max_updated_at,
    )
