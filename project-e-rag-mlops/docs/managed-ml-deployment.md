# Managed ML deployment (toy path)

The core API is plain **FastAPI + Docker**. To practice a **managed** deployment surface, pick **one** cloud and run a minimal “model as a service” exercise in parallel (it does not have to power this RAG repo end-to-end).

## AWS SageMaker

1. Package a small sklearn model (same Iris-style artifact as Project D) into `model.tar.gz` with `inference.py` that loads `joblib` and serves `POST /invocations`.
2. Create a SageMaker **Model** + **Endpoint** (single instance for dev).
3. Call the endpoint from a notebook or curl using the AWS SDK.

**Learn:** VPC endpoints, instance types, cold start vs. multi-variant endpoints.

## Google Vertex AI

1. Upload the container image to **Artifact Registry**.
2. Deploy to **Vertex AI Endpoints** with a custom prediction container (same FastAPI pattern with a Vertex-compatible entrypoint).
3. Send online prediction requests via the Console or API.

**Learn:** regions, autoscaling min/max replicas, request logging.

## Azure Machine Learning

1. Register the model in the **Model registry**.
2. Deploy as a **Managed online endpoint** from the studio UI or CLI.
3. Enable Application Insights for latency and failure tracking.

**Learn:** deployment blue/green, workspace RBAC.

## Mapping back to this project

- **Project E** focuses on **metrics + feedback + refresh**; hosting can stay on Cloud Run, ECS, or AKS.
- **Managed endpoints** are useful when you need SLA-backed scaling, private networking, and unified billing for GPU/CPU—swap the LLM or reranker tier there while keeping the RAG API elsewhere.
