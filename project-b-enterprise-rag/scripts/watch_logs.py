#!/usr/bin/env python3
"""Watch and filter Enterprise RAG JSON logs.

Examples:
  python scripts/watch_logs.py --file logs/server.jsonl --follow
  python scripts/watch_logs.py --file logs/server.jsonl --event ask_complete
  tail -f logs/server.jsonl | python scripts/watch_logs.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor Enterprise RAG JSON logs")
    p.add_argument("--file", default="", help="Log file path to read")
    p.add_argument("--follow", action="store_true", help="Follow the file for new lines")
    p.add_argument("--event", default="", help="Only show messages containing this event token")
    p.add_argument("--level", default="", help="Only show logs at this level (INFO, ERROR, ...) ")
    return p.parse_args()


def _line_iter_file(path: Path, follow: bool):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        if follow:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    yield line
                    continue
                time.sleep(0.2)
        else:
            for line in f:
                yield line


def _line_iter_stdin():
    for line in sys.stdin:
        yield line


def _format_log(obj: dict[str, Any]) -> str:
    ts = str(obj.get("ts", "-")).strip()
    level = str(obj.get("level", "-")).strip()
    logger = str(obj.get("logger", "-")).strip()
    msg = str(obj.get("msg", "")).strip()
    rid = str(obj.get("request_id", "")).strip() or "-"

    event = "-"
    outcome = "-"
    duration = "-"

    if msg.startswith("ask_complete"):
        event = "ask_complete"
        parts = msg.split()
        kv: dict[str, str] = {}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k] = v
        outcome = kv.get("outcome", "-")
        duration = kv.get("duration_s", "-")

    return (
        f"[{ts}] level={level} logger={logger} rid={rid} "
        f"event={event} outcome={outcome} duration_s={duration} msg={msg}"
    )


def run() -> int:
    args = parse_args()

    if args.file:
        src = _line_iter_file(Path(args.file), args.follow)
    else:
        src = _line_iter_stdin()

    want_level = args.level.strip().upper()
    want_event = args.event.strip()

    for raw in src:
        line = raw.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Keep non-JSON lines visible for troubleshooting startup issues.
            print(line)
            continue

        level = str(obj.get("level", "")).upper()
        msg = str(obj.get("msg", ""))

        if want_level and level != want_level:
            continue
        if want_event and want_event not in msg:
            continue

        print(_format_log(obj))

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
