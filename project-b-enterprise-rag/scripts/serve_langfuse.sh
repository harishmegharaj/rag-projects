#!/usr/bin/env bash
# serve_langfuse.sh — start the Enterprise RAG API with Langfuse tracing enabled.
#
# Usage:
#   # preferred: store keys in .env
#   # optional: export LANGFUSE_PUBLIC_KEY=pk-lf-...
#   # optional: export LANGFUSE_SECRET_KEY=sk-lf-...
#   PORT=8000 ./scripts/serve_langfuse.sh
#
# Prerequisites:
#   docker compose -f docker-compose.langfuse.yml up -d  # start Langfuse first
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
ENV_FILE="$ROOT/.env"

if [[ ! -x "$PY" ]]; then
  echo "[serve_langfuse] ERROR: venv not found at $ROOT/.venv" >&2
  echo "[serve_langfuse] Run: cd $ROOT && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key##[[:space:]]}"
    key="${key%%[[:space:]]}"
    [[ -z "$key" ]] && continue
    if [[ -z "${!key+x}" ]]; then
      export "$key=$value"
    fi
  done < "$ENV_FILE"
fi

# Defaults (can be overridden by caller via export)
export LANGFUSE_TRACING="${LANGFUSE_TRACING:-true}"
export LANGFUSE_HOST="${LANGFUSE_HOST:-http://localhost:3000}"
export LOG_JSON="${LOG_JSON:-1}"
export PORT="${PORT:-8000}"
export HOST="${HOST:-0.0.0.0}"

echo "[serve_langfuse] ROOT=$ROOT"
echo "[serve_langfuse] HOST=$HOST PORT=$PORT LANGFUSE_TRACING=$LANGFUSE_TRACING LANGFUSE_HOST=$LANGFUSE_HOST LOG_JSON=$LOG_JSON"

if [[ -z "${LANGFUSE_PUBLIC_KEY:-}" || -z "${LANGFUSE_SECRET_KEY:-}" ]]; then
  echo "[serve_langfuse] WARNING: LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set." >&2
  echo "[serve_langfuse]   Open $LANGFUSE_HOST → Settings → API keys, then:" >&2
  echo "[serve_langfuse]   export LANGFUSE_PUBLIC_KEY=pk-lf-...  LANGFUSE_SECRET_KEY=sk-lf-..." >&2
fi

cd "$ROOT"
exec "$PY" "$ROOT/scripts/serve.py"
