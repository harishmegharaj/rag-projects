from __future__ import annotations

from typing import Any


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_ready_without_rag_indexes(client):
    r = client.get("/ready")
    # Fresh checkout: empty chroma/bm25 → not ready
    assert r.status_code == 503
    data = r.json()
    assert data.get("status") == "not_ready"


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"unified_rag" in r.content


def test_ask_mocked_pipeline(client, monkeypatch):
    from src import api as api_mod

    def fake_run_pipeline(question: str, chroma_dir: Any, bm25_dir: Any) -> dict:
        return {
            "answer": "test answer",
            "retrieved": [],
            "blocked": False,
            "no_context": False,
            "error": False,
            "error_code": None,
        }

    monkeypatch.setattr(api_mod, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(api_mod, "predict_intent", lambda q: None)

    r = client.post("/v1/ask", json={"question": "hello?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "test answer"
    assert body["intent"] is None
    assert "request_id" in body


def test_feedback(client):
    r = client.post(
        "/v1/feedback",
        json={
            "request_id": "test-req-001",
            "rating": "up",
            "question": "q",
            "answer_preview": "a",
        },
    )
    assert r.status_code == 200
    assert r.json()["status"] == "stored"
