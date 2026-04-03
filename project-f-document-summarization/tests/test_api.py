def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_summarize_text_stub(client):
    r = client.post(
        "/v1/summarize/text",
        json={"text": "Hello world. " * 50, "strategy": "stuff"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["llm_mode"] == "stub"
    assert "request_id" in body
    assert len(body["summary"]) > 0


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"summary_http_request_latency_seconds" in r.content


def test_summarize_upload_txt(client):
    files = {"file": ("note.txt", b"Chapter one.\n\nMore content here.\n", "text/plain")}
    r = client.post("/v1/summarize", files=files, data={"strategy": "stuff"})
    assert r.status_code == 200
    assert r.json()["llm_mode"] == "stub"


def test_job_lifecycle(client):
    files = {"file": ("doc.txt", b"Short doc for async job.", "text/plain")}
    r = client.post("/v1/jobs/summarize", files=files, data={"strategy": "auto"})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    r2 = client.get(f"/v1/jobs/{jid}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["job_id"] == jid
    assert body["status"] in ("pending", "running", "completed", "failed")


def test_api_key_rejected_when_set(tmp_path, monkeypatch):
    from src.config import Settings
    from src.api import create_app
    from fastapi.testclient import TestClient

    cfg = Settings(
        jobs_db=tmp_path / "j.db",
        data_dir=tmp_path / "d",
        api_key="secret",
        openai_api_key=None,
    )
    with TestClient(create_app(cfg)) as c:
        r = c.post("/v1/summarize/text", json={"text": "hello"})
        assert r.status_code == 401
        r2 = c.post(
            "/v1/summarize/text",
            json={"text": "hello"},
            headers={"X-API-Key": "secret"},
        )
        assert r2.status_code == 200


def test_webhook_signature_roundtrip():
    from src.webhooks import deliver_webhook, sign_payload
    import json
    from unittest.mock import patch

    payload = {"job_id": "abc", "status": "completed"}
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = sign_payload("testsecret", body)
    assert sig.startswith("sha256=")

    with patch("httpx.Client") as m:
        instance = m.return_value.__enter__.return_value
        instance.post.return_value.status_code = 200
        deliver_webhook("http://example.com/hook", "testsecret", payload)
        call_kw = instance.post.call_args
        assert call_kw[0][0] == "http://example.com/hook"
        sent_headers = call_kw[1]["headers"]
        assert "X-Signature" in sent_headers
