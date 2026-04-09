#!/usr/bin/env python3
"""Simple WebSocket client for /v1/ws/ask end-to-end checks."""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

import websockets


async def run_once(url: str, question: str, api_key: str | None, show_debug: bool) -> int:
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    async with websockets.connect(url, extra_headers=headers) as ws:
        ack_raw = await ws.recv()
        ack = json.loads(ack_raw)
        print(json.dumps(ack, indent=2))

        await ws.send(json.dumps({"question": question}))

        while True:
            raw = await ws.recv()
            msg: dict[str, Any] = json.loads(raw)
            msg_type = str(msg.get("type", ""))

            if msg_type == "status":
                print(f"status: {msg.get('stage')}")
                continue

            if msg_type == "error":
                print(json.dumps(msg, indent=2))
                return 1

            if msg_type == "final":
                print("\nanswer:\n")
                print(msg.get("answer", ""))
                if show_debug:
                    print("\nfinal payload:\n")
                    print(json.dumps(msg, indent=2))
                return 0

            print(json.dumps(msg, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ask the Enterprise RAG websocket endpoint")
    p.add_argument("question", help="Question text to send")
    p.add_argument("--url", default="ws://127.0.0.1:8000/v1/ws/ask", help="WebSocket URL")
    p.add_argument("--api-key", default=None, help="API key if API_KEY is configured")
    p.add_argument("--show-debug", action="store_true", help="Print full final JSON payload")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(
        run_once(
            url=args.url,
            question=args.question.strip(),
            api_key=args.api_key,
            show_debug=args.show_debug,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
