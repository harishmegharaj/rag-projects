# Project F — Unified RAG + MLOps

Production-style **monolith** that merges:

- **Project B — Enterprise RAG:** hybrid retrieval (BM25 + Chroma + RRF), cross-encoder rerank, LangChain orchestration, guardrails, document upload, `/health` / `/ready` / `/metrics`.
- **Project D — ML API + pipeline:** intent classifier training → **joblib** + **`intent_registry.json`** lineage (data hash, metrics, git SHA).
- **Project E — RAG MLOps:** SQLite **`POST /v1/feedback`**, extra Prometheus counters/histograms for intent and feedback.

Formal scope: [REQUIREMENTS.md](REQUIREMENTS.md).

## Quickstart

```bash
cd project-f-unified-rag-mlops
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Intent model (optional but recommended)
python scripts/train_intent.py
# RAG index
python scripts/build_index.py
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is enterprise RAG?"}'
curl -s -X POST http://127.0.0.1:8000/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"request_id":"abc123","rating":"up","question":"demo"}'
```

Set `OPENAI_API_KEY` in `.env` for full LLM answers; without it, the RAG path uses the stub/preview behavior from Project B.

## Key endpoints

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | Liveness |
| GET | `/ready` | RAG indexes; optional intent if `INTENT_REQUIRED_FOR_READY=1` |
| GET | `/metrics` | Prometheus |
| POST | `/v1/ask` | Answer + `intent` (null if no model) + citations |
| POST | `/v1/feedback` | Thumbs + optional correction |
| POST | `/v1/intent/predict` | Intent only |
| GET | `/v1/intent/model` | Registry / artifact status |
| POST | `/v1/documents` | Multipart upload + optional rebuild |
| POST | `/v1/index/rebuild` | Full reindex |

Protected routes use `X-API-Key` or `Authorization: Bearer` when `API_KEY` is set.

## Layout

```
project-f-unified-rag-mlops/
├── REQUIREMENTS.md
├── data/raw/              # corpus + intent_train.csv
├── models/artifacts/      # intent *.joblib
├── models/intent_registry.json   # created by train_intent
├── src/                   # RAG stack (from B) + intent + feedback
├── scripts/
├── vector_store/          # chroma + bm25 (gitignored contents)
└── tests/
```

## Docker

```bash
docker build -t unified-rag-mlops:local .
docker run --rm -p 8000:8000 -e OPENAI_API_KEY -v "$(pwd)/vector_store:/app/vector_store" unified-rag-mlops:local
```

Mount `models/`, `data/`, and indexes as needed.

## CI

Monorepo workflow: `.github/workflows/project-f-unified-rag-mlops-ci.yml`.
