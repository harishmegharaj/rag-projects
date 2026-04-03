#!/usr/bin/env python3
"""Ask a question: python scripts/ask.py \"Your question here\""""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag import answer_with_rag


def main():
    q = " ".join(sys.argv[1:]).strip() or "Summarize the main topics in the documents."
    chroma_dir = ROOT / "vector_store" / "chroma"
    out = answer_with_rag(q, chroma_dir)
    print(out["answer"])
    print("\n--- Retrieved ---")
    print(json.dumps([{"metadata": h["metadata"], "preview": h["text"][:200]} for h in out["retrieved"]], indent=2))


if __name__ == "__main__":
    main()
