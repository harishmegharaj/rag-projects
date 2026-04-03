import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def client(tmp_path, monkeypatch):
    from src.config import Settings
    from src.api import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setenv("OPENAI_API_KEY", "")
    db = tmp_path / "jobs.db"
    data = tmp_path / "data"
    cfg = Settings(
        jobs_db=db,
        data_dir=data,
        api_key=None,
        openai_api_key=None,
    )
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c
