#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
ENV_FILE="$ROOT/.env"

if [[ ! -x "$PY" ]]; then
  echo "[serve_langsmith] Missing venv interpreter at: $PY"
  echo "[serve_langsmith] Create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
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

: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LANGSMITH_TRACING:=true}"
: "${LOG_JSON:=1}"

if [[ -z "${LANGSMITH_API_KEY:-}" ]]; then
  echo "[serve_langsmith] LANGSMITH_API_KEY is not set. Tracing status will be 'misconfigured'."
fi

echo "[serve_langsmith] ROOT=$ROOT"
echo "[serve_langsmith] HOST=$HOST PORT=$PORT LANGSMITH_TRACING=$LANGSMITH_TRACING LOG_JSON=$LOG_JSON"

cd "$ROOT"
exec "$PY" "$ROOT/scripts/serve.py"
