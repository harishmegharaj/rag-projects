#!/usr/bin/env python3
"""One-shot index build (same logic as run_refresh_job without report file)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_core import build_index  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(build_index(), indent=2))
