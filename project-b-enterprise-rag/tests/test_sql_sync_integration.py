from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


_REQUIRED_DEPS = ["rank_bm25", "sqlalchemy", "langchain_text_splitters"]
_MISSING_DEPS = [name for name in _REQUIRED_DEPS if importlib.util.find_spec(name) is None]


class _FakeCollection:
    def __init__(self) -> None:
        self.deleted_ids: list[str] = []
        self.upserted_chunks: list[dict] = []

    def delete(self, ids: list[str]) -> None:
        self.deleted_ids.extend(ids)


class _FakeChromaClient:
    def __init__(self, collection: _FakeCollection) -> None:
        self._collection = collection

    def delete_collection(self, _name: str) -> None:
        return

    def get_collection(self, _name: str) -> _FakeCollection:
        return self._collection


def _fake_upsert(collection: _FakeCollection, chunks: list[dict], **_kwargs) -> None:
    collection.upserted_chunks.extend(chunks)


class SqlSyncIntegrationTest(unittest.TestCase):
    @unittest.skipIf(
        bool(_MISSING_DEPS),
        f"Missing optional deps for SQL sync integration test: {', '.join(_MISSING_DEPS)}",
    )
    def test_full_then_incremental_sql_sync(self) -> None:
        from src.index_builder import rebuild_index, sync_sql_incremental

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            chroma_dir = root / "chroma"
            bm25_dir = root / "bm25"
            raw_dir.mkdir(parents=True, exist_ok=True)
            chroma_dir.mkdir(parents=True, exist_ok=True)
            bm25_dir.mkdir(parents=True, exist_ok=True)

            db_path = root / "app.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                """
                CREATE TABLE documents (
                    id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    title TEXT,
                    body TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO documents (id, updated_at, title, body) VALUES (?, ?, ?, ?)",
                ("1", "2026-04-09T10:00:00", "Doc One", "First body text"),
            )
            conn.commit()

            env = {
                "DOCUMENTS_RAW_DIR": str(raw_dir),
                "CHROMA_PERSIST_DIR": str(chroma_dir),
                "BM25_INDEX_DIR": str(bm25_dir),
                "SQL_DATABASE_URL": f"sqlite:///{db_path}",
                "SQL_SYNC_QUERY": "SELECT id, updated_at, title, body FROM documents",
                "SQL_ID_COLUMN": "id",
                "SQL_UPDATED_AT_COLUMN": "updated_at",
                "SQL_SOURCE_NAME": "test_db",
                "SQL_TEXT_COLUMNS": "title,body",
            }

            fake_collection = _FakeCollection()
            fake_client = _FakeChromaClient(fake_collection)

            with patch.dict(os.environ, env, clear=False):
                with patch("src.index_builder.get_chroma", return_value=fake_client):
                    with patch("src.index_builder.build_collection", return_value=fake_collection):
                        with patch("src.index_builder.upsert_chunks", side_effect=_fake_upsert):
                            full = rebuild_index(raw_dir=raw_dir, include_sql=True, require_sql=True)
                            self.assertEqual(full["mode"], "full")
                            self.assertEqual(full["sql_row_count"], 1)
                            self.assertGreater(full["sql_chunk_count"], 0)

                            conn.execute(
                                "UPDATE documents SET updated_at = ?, body = ? WHERE id = ?",
                                ("2026-04-09T11:00:00", "Updated body text", "1"),
                            )
                            conn.execute(
                                "INSERT INTO documents (id, updated_at, title, body) VALUES (?, ?, ?, ?)",
                                ("2", "2026-04-09T11:05:00", "Doc Two", "Second body text"),
                            )
                            conn.commit()

                            inc = sync_sql_incremental(raw_dir=raw_dir)
                            self.assertEqual(inc["mode"], "incremental_sql")
                            self.assertFalse(inc["no_change"])
                            self.assertEqual(inc["sql_row_count"], 2)
                            self.assertGreater(len(fake_collection.deleted_ids), 0)

                            chunks = json.loads((bm25_dir / "chunks.json").read_text(encoding="utf-8"))
                            sql_chunks = [
                                c
                                for c in chunks
                                if (c.get("metadata") or {}).get("source") == "sql:test_db"
                            ]
                            joined = "\n".join(c.get("text", "") for c in sql_chunks)
                            self.assertIn("Updated body text", joined)
                            self.assertIn("Second body text", joined)
                            self.assertNotIn("First body text", joined)

                            no_change = sync_sql_incremental(raw_dir=raw_dir)
                            self.assertTrue(no_change["no_change"])
                            self.assertEqual(no_change["sql_row_count"], 0)

            conn.close()


if __name__ == "__main__":
    unittest.main()
