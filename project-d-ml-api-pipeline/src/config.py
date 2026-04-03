import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = Path(os.environ.get("DATA_RAW_PATH", ROOT / "data/raw/iris.csv"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
REGISTRY_PATH = MODELS_DIR / "registry.json"

CLASS_NAMES = ("setosa", "versicolor", "virginica")
