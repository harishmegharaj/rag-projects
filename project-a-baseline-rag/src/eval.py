"""
Minimal eval scaffold: add (question, expected_keywords or gold spans) and run after RAG.
"""
from pathlib import Path
from typing import List

from .rag import answer_with_rag


def keyword_overlap(answer: str, keywords: List[str]) -> float:
    lower = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in lower)
    return hits / len(keywords) if keywords else 0.0


def run_smoke_eval(chroma_dir: Path, cases: List[tuple[str, List[str]]]) -> None:
    for q, keys in cases:
        out = answer_with_rag(q, chroma_dir)
        score = keyword_overlap(out["answer"], keys)
        print(f"Q: {q}\n  keyword_recall≈{score:.2f}\n  answer: {out['answer'][:300]}...\n")


# Example — replace with your own questions after indexing real docs:
EXAMPLE_CASES: List[tuple[str, List[str]]] = [
    ("What is this document about?", ["document"]),
]

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    chroma = root / "vector_store" / "chroma"
    run_smoke_eval(chroma, EXAMPLE_CASES)
