from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app


def _disable_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop_worker_loop(*args, **kwargs):
        return None

    monkeypatch.setattr("src.api.worker_loop", _noop_worker_loop)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    _disable_worker(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.sqlite3'}")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(app) as c:
        yield c


def test_sales_insights_heuristic_shape(client: TestClient) -> None:
    payload = {
        "transcript_text": (
            "We need faster onboarding and lower operational cost. "
            "Budget is tight, but a pilot can start next month."
        ),
        "top_k": 8,
    }
    resp = client.post("/v1/nlp/sales-insights", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert "keywords" in data and isinstance(data["keywords"], list)
    assert "signals" in data and isinstance(data["signals"], dict)
    assert "sales_plan" in data and isinstance(data["sales_plan"], list)
    assert data.get("reasoning", {}).get("mode") == "heuristic"


def test_sales_insights_llm_requires_key(client: TestClient) -> None:
    payload = {
        "transcript_text": "Need faster onboarding and lower cost.",
        "top_k": 6,
        "use_llm": True,
        "reasoning_model": "gpt-4.1-mini",
    }
    resp = client.post("/v1/nlp/sales-insights", json=payload)
    assert resp.status_code == 422
    assert "OPENAI_API_KEY is required" in resp.json()["detail"]


def test_sales_insights_llm_model_override_is_forwarded(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_llm(transcript_text: str, top_k: int, reasoning_model: str | None = None) -> dict:
        captured["transcript_text"] = transcript_text
        captured["top_k"] = top_k
        captured["reasoning_model"] = reasoning_model
        return {
            "summary": {"transcript_chars": len(transcript_text), "keyword_count": 1},
            "keywords": [{"term": "pilot", "frequency": 1, "score": 1.0}],
            "signals": {
                "pain_points": ["slow onboarding"],
                "business_goals": ["faster"],
                "objections": ["budget"],
                "buying_signals": ["pilot"],
            },
            "sales_plan": [
                {
                    "phase": "1. Discovery Summary",
                    "objective": "Align requirements",
                    "actions": ["Confirm timeline", "Confirm owners", "Confirm KPI"],
                }
            ],
            "recommended_models": [],
            "reasoning": {"mode": "llm", "model": reasoning_model},
        }

    monkeypatch.setattr("src.api.analyze_transcript_for_sales_llm", _fake_llm)

    payload = {
        "transcript_text": "Client asks for pilot and faster onboarding.",
        "top_k": 7,
        "use_llm": True,
        "reasoning_model": "gpt-4.1",
    }
    resp = client.post("/v1/nlp/sales-insights", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert body["reasoning"]["model"] == "gpt-4.1"
    assert captured["reasoning_model"] == "gpt-4.1"
    assert captured["top_k"] == 7
