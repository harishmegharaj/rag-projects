# Runtime image: application code + Python deps. Indexes are not baked in by default—
# mount a volume or sync from object storage at startup (see README — Production deployment).
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

RUN mkdir -p /opt/hf

ARG BUILD_ID=local
ENV BUILD_ID=${BUILD_ID}

# Default: HTTP API (upload + ask). Mount vector_store + data/raw volumes as needed.
# Override with CHROMA_PERSIST_DIR / BM25_INDEX_DIR / DOCUMENTS_RAW_DIR for mounted paths.
ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
CMD ["python", "scripts/serve.py"]
