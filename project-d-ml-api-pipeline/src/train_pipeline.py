"""Train a small sklearn model, persist artifact + registry entry (lineage)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS_DIR, CLASS_NAMES, DATA_RAW, MODELS_DIR, REGISTRY_PATH, ROOT


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
            cwd=MODELS_DIR.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("GIT_SHA") or os.environ.get("GITHUB_SHA")


def train_and_register(
    *,
    data_path: Path | None = None,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict:
    """Fit logistic regression on tabular CSV; write joblib + update registry.json."""
    csv = Path(data_path or DATA_RAW)
    if not csv.is_file():
        raise FileNotFoundError(f"Training data not found: {csv}")

    df = pd.read_csv(csv)
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=200, random_state=random_state),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    data_hash = _file_sha256(csv)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = f"v1-{ts}"
    artifact_name = f"classifier_{version}.joblib"
    artifact_path = ARTIFACTS_DIR / artifact_name
    joblib.dump(model, artifact_path)

    import sklearn

    def _rel_to_root(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(ROOT.resolve()))
        except ValueError:
            return str(p.resolve())

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
        "class_names": list(CLASS_NAMES),
        "feature_columns": feature_cols,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    old_models: list = []
    if REGISTRY_PATH.is_file():
        prev = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if isinstance(prev.get("models"), list):
            old_models = prev["models"]
    registry = {
        "active_model_path": record["artifact"],
        "active_version": version,
        "models": old_models + [record],
    }

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return record
