"""Load active model from registry."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib

from src.config import REGISTRY_PATH, ROOT


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


@lru_cache(maxsize=1)
def load_active_model():
    if not REGISTRY_PATH.is_file():
        raise RuntimeError(
            f"No registry at {REGISTRY_PATH}. Run `python scripts/train.py` first."
        )
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    rel = data.get("active_model_path")
    if not rel:
        raise RuntimeError("registry.json has no active_model_path.")
    path = _resolve_path(rel)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact missing: {path}")
    return joblib.load(path), data
