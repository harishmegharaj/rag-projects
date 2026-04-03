"""Persist BM25 index + chunk list for hybrid retrieval."""
import json
import pickle
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def build_bm25(chunks: List[dict]) -> BM25Okapi:
    corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus)


def save_index(chunks: List[dict], bm25: BM25Okapi, dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    with open(dir_path / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(dir_path / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)


def load_index(dir_path: Path) -> tuple[BM25Okapi, List[dict]]:
    with open(dir_path / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(dir_path / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    return bm25, chunks
