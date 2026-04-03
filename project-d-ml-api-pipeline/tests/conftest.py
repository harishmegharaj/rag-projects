import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def ensure_trained_model():
    from src.config import REGISTRY_PATH
    from src.model_loader import load_active_model
    from src.train_pipeline import train_and_register

    need = True
    if REGISTRY_PATH.is_file():
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if data.get("active_model_path"):
            need = False
    if need:
        train_and_register()
    load_active_model.cache_clear()
    load_active_model()
