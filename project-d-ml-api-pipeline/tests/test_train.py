import json
from pathlib import Path

import src.train_pipeline as tp


def test_train_and_register_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(tp, "ARTIFACTS_DIR", tmp_path / "artifacts")
    (tmp_path / "artifacts").mkdir(parents=True)
    csv = Path(__file__).resolve().parents[1] / "data/raw/iris.csv"
    rec = tp.train_and_register(data_path=csv, random_state=7)
    assert rec["metrics"]["accuracy_holdout"] >= 0.85
    reg = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert reg["active_version"] == rec["version"]
    assert (tmp_path / "artifacts" / Path(rec["artifact"]).name).is_file()
