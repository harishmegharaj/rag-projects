"""BM25 + vector Chroma with reciprocal rank fusion."""
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer

from .bm25_index import load_index, tokenize
from .embed_store import COLLECTION, DEFAULT_MODEL, get_chroma


def reciprocal_rank_fusion(rank_lists: List[List[str]], k: int = 60) -> List[tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranked in rank_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_retrieve(
    query: str,
    chroma_dir: Path,
    bm25_dir: Path,
    bm25_k: int = 20,
    vector_k: int = 20,
    rrf_k: int = 60,
    fusion_top_n: int = 15,
    model_name: str = DEFAULT_MODEL,
) -> List[dict]:
    bm25, chunks = load_index(bm25_dir)
    id_to_chunk = {c["id"]: c for c in chunks}

    q_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_order = sorted(range(len(chunks)), key=lambda i: -bm25_scores[i])
    bm25_ids = [chunks[i]["id"] for i in bm25_order[:bm25_k]]

    client = get_chroma(chroma_dir)
    collection = client.get_collection(COLLECTION)
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query]).tolist()
    vec_res = collection.query(query_embeddings=q_emb, n_results=vector_k)
    vector_ids = vec_res["ids"][0]

    fused = reciprocal_rank_fusion([bm25_ids, vector_ids], k=rrf_k)[:fusion_top_n]
    out: List[dict] = []
    for doc_id, rrf_score in fused:
        c = id_to_chunk.get(doc_id)
        if c:
            out.append({**c, "rrf_score": rrf_score})
    return out
