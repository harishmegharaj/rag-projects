def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask_and_metrics(client):
    r = client.post("/v1/ask", json={"question": "What is Prometheus?"})
    assert r.status_code == 200
    body = r.json()
    assert "request_id" in body
    assert body["retrieval_hits"] >= 1
    assert body["llm_mode"] in ("stub", "openai")
    m = client.get("/metrics")
    assert m.status_code == 200
    assert b"rag_retrieval_chunks" in m.content


def test_feedback(client):
    r = client.post("/v1/ask", json={"question": "monitoring"})
    body = r.json()
    rid = body["request_id"]
    fb = client.post(
        "/v1/feedback",
        json={
            "request_id": rid,
            "rating": "up",
            "question": "monitoring",
            "answer_preview": body["answer"][:200],
        },
    )
    assert fb.status_code == 200
    assert fb.json()["status"] == "stored"
