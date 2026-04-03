# Project E — LLM/RAG “MLOps” layer

A compact **RAG API** (TF-IDF retrieval over chunked Markdown + optional OpenAI generation) with **Prometheus metrics**, a **feedback** API backed by **SQLite**, and a **refresh script** for re-indexing. Documentation covers **drift** thinking and a **managed ML** learning path (SageMaker / Vertex / Azure ML) separate from the core app.

## Features

| Area | What is included |
| --- | --- |
| **Monitor** | Latency histogram (`rag_http_request_latency_seconds`), retrieval chunk counts, hit/miss counter, LLM mode + rough token counter, error counter |
| **Feedback** | `POST /v1/feedback` stores thumbs up/down and optional correction text for later review |
| **Refresh** | `scripts/run_refresh_job.py` rebuilds the index from `data/corpus/` |
| **Docs** | [docs/drift-and-refresh.md](docs/drift-and-refresh.md), [docs/managed-ml-deployment.md](docs/managed-ml-deployment.md), [interview Q&A (D & E)](../docs/interview-qa-ml-api-and-rag-mlops.md) |

## Quickstart

```bash
cd project-e-rag-mlops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_index.py
uvicorn src.api:app --reload --host 0.0.0.0 --port 8010
```

- **Stub LLM** (default): no `OPENAI_API_KEY` — answers are templated from retrieved chunks.
- **OpenAI**: set `OPENAI_API_KEY` and optionally `OPENAI_CHAT_MODEL` (default `gpt-4o-mini`).

```bash
curl -s -X POST http://127.0.0.1:8010/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How is monitoring exposed?"}'
curl -s http://127.0.0.1:8010/metrics | head
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `CORPUS_DIR` | `data/corpus` | Markdown sources |
| `STORE_DIR` | `store` | TF-IDF artifact + `chunks.json` |
| `FEEDBACK_DB` | `data/feedback.db` | SQLite path |
| `RAG_TOP_K` | `4` | Chunks passed to the generator |
| `HOST` / `PORT` | `0.0.0.0` / `8010` | HTTP bind |

## Docker

```bash
docker build -t project-e-rag-mlops:local .
docker run --rm -p 8010:8010 project-e-rag-mlops:local
```

## CI

Monorepo workflows: `.github/workflows/project-e-ci.yml` (lint + tests on changes under `project-e-rag-mlops/**`).

## Tests

```bash
ruff check src scripts tests
pytest
```
