#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

ensure_venv() {
  if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c "import sys" >/dev/null 2>&1; then
    return 0
  fi
  echo "[test.sh] Creating or repairing virtualenv at $VENV_DIR"
  python3 -m venv --clear "$VENV_DIR"
}

ensure_deps() {
  if "$VENV_PY" -c "import fastapi, pytest" >/dev/null 2>&1; then
    return 0
  fi
  echo "[test.sh] Installing test dependencies"
  "$VENV_PY" -m pip install --upgrade pip >/dev/null
  "$VENV_PY" -m pip install -r "$ROOT_DIR/requirements.txt" pytest >/dev/null
}

ensure_venv
ensure_deps

TARGET="${1:-tests/test_sales_insights_api.py}"
shift || true

cd "$ROOT_DIR"
exec "$VENV_PY" -m pytest -q "$TARGET" "$@"
