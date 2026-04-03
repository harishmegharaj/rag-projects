#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import documents_raw_dir
from src.index_builder import rebuild_index


def main():
    raw = documents_raw_dir()
    if not any(p.is_file() for p in raw.rglob("*") if p.name != ".gitkeep"):
        print(f"Add PDF or Markdown files under {raw} first (or use the upload API).")
        sys.exit(1)
    try:
        out = rebuild_index(raw)
    except ValueError as e:
        print(e)
        sys.exit(1)
    print(f"Ingested {out['chunk_count']} chunks.")
    print(f"Chroma → {out['chroma_dir']}\nBM25 + chunks → {out['bm25_dir']}")


if __name__ == "__main__":
    main()
