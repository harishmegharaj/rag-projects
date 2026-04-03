# RAG learning projects

Hands-on work for **Project A** (baseline RAG), **Project B** (enterprise-style RAG), **Project D** (ML API + CI/CD + lineage), **Project E** (RAG MLOps-style monitoring and feedback), and **Project F** (document summarization + integration patterns), aligned with the AI/ML Architect learning path.

| Folder | Focus |
|--------|--------|
| [`project-a-baseline-rag/`](project-a-baseline-rag/) | Chunking, embeddings, vector store, prompting, citations, simple eval |
| [`project-b-enterprise-rag/`](project-b-enterprise-rag/) | Hybrid search, re-ranking, orchestration, guardrails, logging |
| [`project-c-video-transcription/`](project-c-video-transcription/) | Video/audio transcription API (separate track) |
| [`project-d-ml-api-pipeline/`](project-d-ml-api-pipeline/) | Tabular classifier FastAPI, Docker, CI/CD, model registry / lineage |
| [`project-e-rag-mlops/`](project-e-rag-mlops/) | RAG + Prometheus metrics, feedback store, refresh job, drift docs |
| [`project-f-document-summarization/`](project-f-document-summarization/) | Document summarization (stuff + map–reduce), async jobs, webhooks, connector interface |

**Suggested order:** Complete Project A first, then Project B. Use D for API/container/CI practice and E for observability and refresh patterns around RAG (you can copy patterns from A/B or refactor shared code later). Use **F** when you want summarization-focused APIs and integration-style patterns (callbacks, optional auth, metrics) without building a full RAG retrieval stack.

**Interview prep:** all Q&A files live under [`docs/`](docs/) — see the [index](docs/README.md) or jump to [NLP (senior)](docs/interview-qa-nlp-senior.md), [RAG (senior / staff)](docs/interview-qa-rag-senior.md), [ML API & RAG MLOps (D & E)](docs/interview-qa-ml-api-and-rag-mlops.md), [enterprise RAG (B)](docs/interview-qa-enterprise-rag.md), [document summarization & integration (F)](docs/interview-qa-document-summarization.md).

## Quick start

```bash
cd project-a-baseline-rag
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add API keys if using OpenAI, etc.
```

Repeat for `project-b-enterprise-rag` in a separate venv if you prefer isolated environments.
