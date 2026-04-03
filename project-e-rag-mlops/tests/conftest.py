import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def client(tmp_path, monkeypatch):
    corpus = ROOT / "data/corpus"
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    monkeypatch.setenv("STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedback.db"))

    import src.config as cfg

    importlib.reload(cfg)

    import src.feedback_store as feedback_store

    feedback_store._engine = None
    feedback_store._Session = None
    importlib.reload(feedback_store)

    import src.rag_core as rag_core

    importlib.reload(rag_core)

    import src.api as api

    importlib.reload(api)

    (tmp_path / "store").mkdir(parents=True, exist_ok=True)
    rag_core.build_index()
    from fastapi.testclient import TestClient

    return TestClient(api.app)
