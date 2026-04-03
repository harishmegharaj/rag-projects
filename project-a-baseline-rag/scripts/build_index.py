#!/usr/bin/env python3
"""Build vector index from data/raw (Chroma locally or Pinecone when VECTOR_BACKEND=pinecone)."""
import sys
from pathlib import Path

# project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import chroma_persist_dir, pinecone_index_name, pinecone_namespace, vector_backend
from src.embed_store import build_collection, get_chroma, upsert_chunks
from src.ingest import ingest_directory


def main():
    raw = ROOT / "data" / "raw"
    chroma_dir = chroma_persist_dir(ROOT)
    if not any(raw.iterdir()):
        print(f"Add PDF or Markdown files under {raw} first.")
        sys.exit(1)
    chunks = ingest_directory(raw)
    if not chunks:
        print("No PDF/Markdown chunks found. Add files under data/raw/ (not just .gitkeep).")
        sys.exit(1)
    print(f"Ingested {len(chunks)} chunks.")

    backend = vector_backend()
    if backend == "pinecone":
        from src.pinecone_store import reset_namespace, upsert_chunks as pinecone_upsert

        reset_namespace()
        pinecone_upsert(chunks)
        ns = pinecone_namespace()
        ns_label = repr(ns) if ns else "default"
        print(f"Upserted to Pinecone index {pinecone_index_name()!r} (namespace={ns_label}).")
        return

    client = get_chroma(chroma_dir)
    build_collection(client)
    try:
        client.delete_collection("baseline_rag")
    except Exception:
        pass
    col = build_collection(client)
    upsert_chunks(col, chunks)
    print(f"Index written to {chroma_dir}")


if __name__ == "__main__":
    main()
