"""Train intent classifier (text → label) and register artifact + lineage (Project D pattern)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import ROOT, intent_registry_path, intent_train_csv_path, models_dir


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("GIT_SHA") or os.environ.get("GITHUB_SHA")


def _rel_to_root(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(p.resolve())


def train_and_register_intent(
    *,
    data_path: Path | None = None,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict:
    """Fit TF-IDF + logistic regression on CSV columns `text`, `intent`."""
    csv = Path(data_path or intent_train_csv_path())
    if not csv.is_file():
        raise FileNotFoundError(f"Training data not found: {csv}")

    df = pd.read_csv(csv)
    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("CSV must have columns: text, intent")

    texts = df["text"].astype(str).tolist()
    labels = df["intent"].astype(str).tolist()

    counts = Counter(labels)
    strat = labels if len(labels) >= 4 and all(counts[k] >= 2 for k in counts) else None
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=strat
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=8192, ngram_range=(1, 2)),
            ),
            (
                "clf",
                LogisticRegression(max_iter=500, random_state=random_state),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    artifacts = models_dir() / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    data_hash = _file_sha256(csv)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = f"intent-{ts}"
    artifact_name = f"intent_{version}.joblib"
    artifact_path = artifacts / artifact_name
    joblib.dump(model, artifact_path)

    import sklearn

    label_names = sorted({str(x) for x in labels})
    record = {
        "version": version,
        "artifact": _rel_to_root(artifact_path),
        "data_path": _rel_to_root(csv),
        "data_sha256": data_hash,
        "metrics": {"accuracy_holdout": acc, "test_size": test_size, "random_state": random_state},
        "library_versions": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "git_sha": _git_sha(),
        "label_names": label_names,
        "task": "intent_classification",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    reg_path = intent_registry_path()
    old_models: list = []
    if reg_path.is_file():
        prev = json.loads(reg_path.read_text(encoding="utf-8"))
        if isinstance(prev.get("models"), list):
            old_models = prev["models"]
    registry = {
        "active_model_path": record["artifact"],
        "active_version": version,
        "models": old_models + [record],
    }

    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return record
