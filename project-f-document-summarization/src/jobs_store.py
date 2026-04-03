"""SQLite-backed job status for async summarization."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_lock = threading.Lock()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path) -> None:
    with _lock:
        conn = _connect(path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    filename TEXT,
                    strategy TEXT,
                    callback_url TEXT,
                    result_json TEXT,
                    error TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


@dataclass
class JobRecord:
    id: str
    status: str
    filename: str | None
    strategy: str | None
    callback_url: str | None
    result: dict[str, Any] | None
    error: str | None


def create_job(
    path: Path,
    *,
    filename: str | None,
    strategy: str,
    callback_url: str | None,
) -> str:
    import time

    jid = uuid.uuid4().hex
    now = time.time()
    with _lock:
        conn = _connect(path)
        try:
            conn.execute(
                """
                INSERT INTO jobs (id, status, created_at, updated_at, filename, strategy, callback_url, result_json, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (jid, "pending", now, now, filename, strategy, callback_url),
            )
            conn.commit()
        finally:
            conn.close()
    return jid


def update_job(path: Path, jid: str, **fields: Any) -> None:
    import time

    allowed = {"status", "result_json", "error"}
    sets: list[str] = []
    vals: list[Any] = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        sets.append(f"{k} = ?")
        vals.append(v)
    if not sets:
        return
    sets.append("updated_at = ?")
    vals.append(time.time())
    vals.append(jid)
    with _lock:
        conn = _connect(path)
        try:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals)
            conn.commit()
        finally:
            conn.close()


def get_job(path: Path, jid: str) -> JobRecord | None:
    with _lock:
        conn = _connect(path)
        try:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
        finally:
            conn.close()
    if row is None:
        return None
    rj = row["result_json"]
    result = json.loads(rj) if rj else None
    return JobRecord(
        id=row["id"],
        status=row["status"],
        filename=row["filename"],
        strategy=row["strategy"],
        callback_url=row["callback_url"],
        result=result,
        error=row["error"],
    )
