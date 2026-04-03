import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = Path(os.environ.get("CORPUS_DIR", ROOT / "data/corpus"))
STORE_DIR = Path(os.environ.get("STORE_DIR", ROOT / "store"))
VECTORIZER_PATH = STORE_DIR / "tfidf_vectorizer.joblib"
CHUNKS_PATH = STORE_DIR / "chunks.json"
FEEDBACK_DB = Path(os.environ.get("FEEDBACK_DB", ROOT / "data/feedback.db"))

TOP_K = int(os.environ.get("RAG_TOP_K", "4"))
