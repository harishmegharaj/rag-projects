#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import documents_raw_dir
from src.index_builder import rebuild_index


def main():
    raw = documents_raw_dir()
    try:
        out = rebuild_index(raw)
    except ValueError as e:
        print(e)
        sys.exit(1)
    print(
        f"Ingested {out['chunk_count']} chunks "
        f"(files={out['file_chunk_count']}, sql={out['sql_chunk_count']}, sql_rows={out['sql_row_count']})."
    )
    print(f"Chroma → {out['chroma_dir']}\nBM25 + chunks → {out['bm25_dir']}")


if __name__ == "__main__":
    main()
