"""
Retrieve top-k chunks by embedding similarity (Chroma or Pinecone via ``VECTOR_BACKEND``).
"""
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer

from .config import vector_backend
from .embed_store import DEFAULT_MODEL, get_chroma


def query_collection(
    persist_dir: Path,
    query: str,
    k: int = 5,
    model_name: str = DEFAULT_MODEL,
    collection_name: str = "baseline_rag",
) -> List[dict]:
    if vector_backend() == "pinecone":
        from .pinecone_store import query_chunks

        return query_chunks(query, k=k, model_name=model_name)

    client = get_chroma(persist_dir)
    collection = client.get_collection(collection_name)
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query]).tolist()
    result = collection.query(query_embeddings=q_emb, n_results=k, include=["documents", "metadatas", "distances"])
    out: List[dict] = []
    for doc, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        out.append({"text": doc, "metadata": meta, "distance": dist})
    return out
