#!/usr/bin/env python3
"""Optional LangGraph ReAct agent: model picks search_documents, calculator, or fetch_internal_http."""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent_runner import run_agent
from src.config import bm25_index_dir, chroma_persist_dir

logging.basicConfig(level=logging.INFO)


def main() -> None:
    q = " ".join(sys.argv[1:]).strip() or (
        "What topics appear in the indexed documents? Also compute (19 + 23) * 2."
    )
    out = run_agent(q, chroma_persist_dir(), bm25_index_dir())
    print(out["answer"])
    if out.get("messages"):
        print("\n--- Messages (tool loop) ---")
        for i, m in enumerate(out["messages"]):
            t = type(m).__name__
            c = getattr(m, "content", "")
            c_preview = (c[:200] + "…") if isinstance(c, str) and len(c) > 200 else c
            print(f"{i:02d} {t}: {c_preview}")


if __name__ == "__main__":
    main()
