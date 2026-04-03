from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_iris_like():
    r = client.post(
        "/v1/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["label_name"] == "setosa"
    assert "latency_ms" in body
