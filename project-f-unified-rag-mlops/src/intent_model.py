"""Intent classifier: load TF-IDF + sklearn pipeline from intent registry (Project D-style lineage)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import ROOT, intent_registry_path


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def intent_artifact_status() -> dict[str, Any]:
    reg_path = intent_registry_path()
    if not reg_path.is_file():
        return {"present": False, "reason": "no_registry"}
    data = json.loads(reg_path.read_text(encoding="utf-8"))
    rel = data.get("active_model_path")
    if not rel:
        return {"present": False, "reason": "no_active_model_path"}
    path = _resolve_path(rel)
    if not path.is_file():
        return {"present": False, "reason": "artifact_missing", "path": str(path)}
    return {
        "present": True,
        "version": data.get("active_version"),
        "path": str(path),
    }


@lru_cache(maxsize=1)
def _load_bundle() -> tuple[Any, dict[str, Any]] | None:
    st = intent_artifact_status()
    if not st["present"]:
        return None
    reg_path = intent_registry_path()
    data = json.loads(reg_path.read_text(encoding="utf-8"))
    rel = data.get("active_model_path")
    if not rel:
        return None
    path = _resolve_path(rel)
    return joblib.load(path), data


def clear_intent_cache() -> None:
    _load_bundle.cache_clear()


def predict_intent(text: str) -> dict[str, Any] | None:
    """Return top intent + probability map; None if no trained model on disk."""
    bundle = _load_bundle()
    if bundle is None:
        return None
    model, registry = bundle
    t = (text or "").strip()
    if not t:
        return None
    proba = model.predict_proba([t])[0]
    clf = model.named_steps["clf"]
    classes = clf.classes_
    labels = [str(c) for c in classes]
    probs = {labels[i]: float(proba[i]) for i in range(len(labels))}
    best = int(np.argmax(proba))
    return {
        "intent": labels[best],
        "probabilities": probs,
        "model_version": registry.get("active_version"),
    }
