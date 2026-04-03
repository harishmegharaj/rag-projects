#!/usr/bin/env python3
"""
Create a Pinecone serverless index for this project (384 dimensions, cosine).

Prerequisites: PINECONE_API_KEY in .env. Index name defaults to PINECONE_INDEX_NAME.

Usage (from project root):
  python scripts/create_pinecone_index.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinecone import Pinecone, ServerlessSpec

from src.config import pinecone_api_key, pinecone_index_name
from src.embed_store import EMBEDDING_DIMENSION


def main() -> None:
    key = pinecone_api_key()
    if not key:
        print("Set PINECONE_API_KEY in .env (copy from .env.example).")
        sys.exit(1)
    name = pinecone_index_name()
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    pc = Pinecone(api_key=key)
    if pc.has_index(name):
        desc = pc.describe_index(name)
        print(f"Index {name!r} already exists (dimension={desc.dimension}, metric={desc.metric}).")
        return

    pc.create_index(
        name=name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    print(
        f"Created serverless index {name!r}: dimension={EMBEDDING_DIMENSION}, "
        f"metric=cosine, cloud={cloud}, region={region}"
    )


if __name__ == "__main__":
    main()
