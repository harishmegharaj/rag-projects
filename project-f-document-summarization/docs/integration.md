# Integration patterns

This project shows how to **separate transport (HTTP) from upstream systems** so you can plug in real document sources and downstream consumers.

## 1. `DocumentSource` (connector boundary)

See `src/connectors/base.py`. Implement `DocumentSource.fetch(ref)` for objects in S3, Azure Blob, GCS, SharePoint, Google Drive, or an internal CMS. The API returns a `DocumentPayload` (`filename` + `bytes`); `extract.py` turns that into text for summarization.

Typical flow:

1. External system stores a reference (`ref`) in your DB or queue.
2. Worker or API resolves `ref` via your `DocumentSource` implementation.
3. You call the same summarization helpers used by `POST /v1/summarize` (or enqueue a job).

## 2. Async jobs + webhooks

`POST /v1/jobs/summarize` accepts multipart uploads and optional `callback_url`. When the job finishes, the service **POSTs** a JSON payload to that URL with header `X-Signature: sha256=<hmac>` where the HMAC is computed over the raw JSON body using `WEBHOOK_SECRET`.

Downstream systems should:

- Verify the signature before trusting the body.
- Respond with HTTP 2xx quickly; do heavy work asynchronously on their side.

## 3. Operational integration

- **Auth:** Set `API_KEY` and send `X-API-Key` or `Authorization: Bearer <token>`.
- **Metrics:** `GET /metrics` (Prometheus) for latency and job/strategy counters.
- **Observability:** Point your scraper or agent at `/metrics`; alert on error rates and latency histograms.

## 4. What is not in this repo

- Managed connectors (OAuth to SharePoint, etc.) — implement as separate packages or services that call your API with extracted files.
- Message queues (SQS, Pub/Sub) — replace `BackgroundTasks` with a worker that consumes jobs from a queue and calls the same `run_summarize_job` logic.
