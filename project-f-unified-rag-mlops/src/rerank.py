"""Cross-encoder re-ranking for (query, passage) pairs."""
from typing import List

from sentence_transformers import CrossEncoder

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(query: str, candidates: List[dict], model_name: str = DEFAULT_RERANKER, top_n: int = 5) -> List[dict]:
    if not candidates:
        return []
    model = CrossEncoder(model_name)
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs)
    scored = sorted(zip(candidates, scores), key=lambda x: -x[1])
    out = []
    for c, s in scored[:top_n]:
        out.append({**c, "rerank_score": float(s)})
    return out
