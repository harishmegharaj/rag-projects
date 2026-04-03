#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import bm25_index_dir, chroma_persist_dir
from src.rag_pipeline import debug_json, run_pipeline

logging.basicConfig(level=logging.INFO)


def main():
    q = " ".join(sys.argv[1:]).strip() or "Summarize the main topics in the documents."
    chroma_dir = chroma_persist_dir()
    bm25_dir = bm25_index_dir()
    out = run_pipeline(q, chroma_dir, bm25_dir)
    print(out["answer"])
    print("\n--- Debug ---")
    print(debug_json(out))


if __name__ == "__main__":
    main()
