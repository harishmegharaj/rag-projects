"""
Embed chunks and persist to Chroma (local).

For managed production indexes, set ``VECTOR_BACKEND=pinecone`` and use
``src.pinecone_store`` (see README). Same embedding model must be used for ingest and query.
"""
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"
# Vector size for DEFAULT_MODEL; Pinecone indexes must use this dimension.
EMBEDDING_DIMENSION = 384


def get_chroma(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def build_collection(client, name: str = "baseline_rag"):
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_chunks(
    collection,
    chunks: List[dict],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> None:
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    ids = [f"doc_{i}" for i in range(len(chunks))]
    metadatas = []
    for c in chunks:
        m = dict(c["metadata"])
        if m.get("page") is None:
            m["page"] = -1
        metadatas.append(m)

    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]
        embeddings = model.encode(batch_texts, show_progress_bar=True).tolist()
        collection.add(ids=batch_ids, documents=batch_texts, metadatas=batch_meta, embeddings=embeddings)
