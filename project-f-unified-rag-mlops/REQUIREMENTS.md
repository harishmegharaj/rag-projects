# Unified RAG + MLOps — requirements

## Product summary

Single service for **internal knowledge Q&A** over uploaded documents, with **intent classification** for routing/analytics and a **feedback** channel for human review—combining patterns from Projects B, D, and E.

## Functional requirements

| ID | Requirement |
| --- | --- |
| FR-1 | **Q&A:** Accept a natural-language question and return an answer with **retrieved chunk previews** using hybrid retrieval (BM25 + dense vectors + RRF), **cross-encoder reranking**, and optional OpenAI generation (Project B). |
| FR-2 | **Intent:** Classify each question into a discrete **intent** label with **probabilities** using a trained sklearn pipeline (TF-IDF + logistic regression) registered in **`models/intent_registry.json`** (Project D lineage pattern). |
| FR-3 | **Corpus:** Support PDF/Markdown under `data/raw/`, **upload** via API, and **full reindex** (serialized rebuild) producing paired Chroma + BM25 artifacts. |
| FR-4 | **Feedback:** Accept thumbs up/down and optional correction text keyed by `request_id`, persisted in **SQLite** (Project E pattern). |
| FR-5 | **Observability:** Expose **liveness** (`/health`), **readiness** (`/ready`), **Prometheus** (`/metrics`), structured logging, and **request IDs** (Projects B + E). |

## Non-functional requirements

| ID | Requirement |
| --- | --- |
| NFR-1 | **Security:** Optional `API_KEY` on protected routes; guardrails and log redaction on the RAG path (Project B). |
| NFR-2 | **Deployability:** Environment-driven paths for indexes, models, and feedback DB; Docker image runs `scripts/serve.py`. |
| NFR-3 | **Lineage:** Intent training records **data hash**, **metrics**, **library versions**, and **git SHA** in the intent registry (Project D). |
| NFR-4 | **Readiness:** Default readiness requires **RAG indexes** only; set `INTENT_REQUIRED_FOR_READY=1` to require a trained intent artifact as well. |
| NFR-5 | **CI:** Lint (Ruff) and tests (pytest) on changes to this project. |

## Out of scope (v1)

- Incremental / partial index updates (full rebuild only).
- Multi-tenant isolation and strong data boundaries.
- Managed vector DB (on-disk Chroma + BM25 as in Project B).
