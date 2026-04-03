#!/usr/bin/env python3
"""
Rebuild TF-IDF index from corpus (scheduled job / cron / K8s CronJob).

Example:
  python scripts/run_refresh_job.py

Wire to your scheduler (GitHub Actions cron, Cloud Scheduler + Cloud Run job, etc.).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_core import build_index  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild RAG index from corpus.")
    p.add_argument("--report", type=Path, default=None, help="Write JSON status to this path")
    args = p.parse_args()
    info = build_index()
    info["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(info, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
