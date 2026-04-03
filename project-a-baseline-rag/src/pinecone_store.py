"""Pinecone-backed vectors for production-style deployments.

Chunk text is stored in vector metadata under ``text`` (Pinecone has no separate
document field like Chroma). Keep chunks within your embedding model / metadata size limits.
"""
import math
from typing import Any, List

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from .config import pinecone_api_key, pinecone_index_name, pinecone_namespace
from .embed_store import DEFAULT_MODEL

_METADATA_TEXT_MAX = 35000


def _pc() -> Pinecone:
    key = pinecone_api_key()
    if not key:
        raise ValueError("PINECONE_API_KEY is required when VECTOR_BACKEND=pinecone")
    return Pinecone(api_key=key)


def _index():
    return _pc().Index(pinecone_index_name())


def _ns():
    n = pinecone_namespace()
    return n if n else None


def reset_namespace() -> None:
    """Remove all vectors in the configured namespace (default namespace if unset)."""
    idx = _index()
    idx.delete(delete_all=True, namespace=_ns())


def _pinecone_metadata(chunk_meta: dict, text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in chunk_meta.items():
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        out[k] = int(v) if k == "page" and isinstance(v, float) else v
    body = text if len(text) <= _METADATA_TEXT_MAX else text[:_METADATA_TEXT_MAX]
    out["text"] = body
    return out


def upsert_chunks(
    chunks: List[dict],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 100,
) -> None:
    model = SentenceTransformer(model_name)
    idx = _index()
    ns = _ns()
    texts = [c["text"] for c in chunks]
    ids = [f"doc_{i}" for i in range(len(chunks))]

    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_chunks = chunks[start:end]
        embeddings = model.encode(batch_texts, show_progress_bar=True).tolist()
        vectors = []
        for vid, emb, ch in zip(batch_ids, embeddings, batch_chunks):
            m = dict(ch["metadata"])
            if m.get("page") is None:
                m["page"] = -1
            meta = _pinecone_metadata(m, ch["text"])
            vectors.append({"id": vid, "values": emb, "metadata": meta})
        idx.upsert(vectors=vectors, namespace=ns)


def query_chunks(
    query: str,
    k: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> List[dict]:
    model = SentenceTransformer(model_name)
    idx = _index()
    qv = model.encode([query]).tolist()[0]
    res = idx.query(vector=qv, top_k=max(1, k), namespace=_ns(), include_metadata=True)
    out: List[dict] = []
    for match in res.matches or []:
        meta = dict(match.metadata or {})
        text = str(meta.pop("text", "") or "")
        score = float(getattr(match, "score", 0.0) or 0.0)
        out.append({"text": text, "metadata": meta, "distance": 1.0 - score})
    return out
