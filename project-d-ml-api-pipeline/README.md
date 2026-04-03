# Project D — ML API + pipeline

Small **tabular classifier** (Iris-style features) exposed through **FastAPI**, **Docker**-ready, with **CI** (lint + tests) and **CD** (image build to GHCR for a staging tag). Training writes a **model registry** JSON for basic **lineage** (data hash, metrics, library versions, git SHA).

**Interview Q&A (topics D & E):** [docs/interview-qa-ml-api-and-rag-mlops.md](../docs/interview-qa-ml-api-and-rag-mlops.md)

## What you get

1. **API** — `POST /v1/predict` with four numeric features; `GET /health`, `GET /v1/model`.
2. **Training** — `python scripts/train.py` → `models/artifacts/*.joblib` + `models/registry.json`.
3. **Container** — multi-stage-friendly Dockerfile; image runs `train` at build so the artifact is baked in (swap for volume mount in production if you prefer).
4. **CI/CD** — monorepo root `.github/workflows/project-d-ci.yml` and `project-d-cd-staging.yml` (Ruff + pytest; Docker push to `ghcr.io` with `staging-*` tags).

## Quickstart (local)

```bash
cd project-d-ml-api-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Docker

```bash
docker build -t project-d-ml-api:local .
docker run --rm -p 8000:8000 project-d-ml-api:local
```

## CI/CD layout

Workflows for this monorepo live at **repository root**: `.github/workflows/project-d-ci.yml` and `project-d-cd-staging.yml` (path filters on `project-d-ml-api-pipeline/**`). If Project D is its **own** Git repository, copy those files into `.github/workflows/`, drop the `project-d-ml-api-pipeline/` prefix from `paths`, `working-directory`, and Docker `context`, and set `context: .`.

**Staging deploy:** after the image is in GHCR, add a second job (or separate workflow) that deploys to your platform:

- **Google Cloud Run** — `gcloud run deploy` with the pushed image.
- **AWS App Runner / Lambda container** — map port 8000.
- **Azure Container Apps** — create revision from registry.
- **Fly.io** — `fly deploy` with the image reference.

## Lineage

See [docs/lineage.md](docs/lineage.md).

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATA_RAW_PATH` | `data/raw/iris.csv` | Training CSV path |
| `MODELS_DIR` | `models/` | Registry + artifacts directory |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | HTTP bind |

## Tests

```bash
ruff check src scripts tests
pytest
```
