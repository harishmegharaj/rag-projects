from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "enterprise_rag"


def get_chroma(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def build_collection(client, name: str = COLLECTION):
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_chunks(
    collection,
    chunks: List[dict],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> None:
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = []
    for c in chunks:
        m = dict(c["metadata"])
        if m.get("page") is None:
            m["page"] = -1
        metadatas.append(m)

    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        embeddings = model.encode(texts[start:end], show_progress_bar=True).tolist()
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings,
        )
