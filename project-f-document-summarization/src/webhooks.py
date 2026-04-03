"""Outbound webhook delivery with HMAC-SHA256 signature."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def sign_payload(secret: str, body_bytes: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def deliver_webhook(url: str, secret: str, payload: dict[str, Any], timeout_s: float = 30.0) -> tuple[int, str]:
    body_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = sign_payload(secret, body_bytes)
    headers = {
        "Content-Type": "application/json",
        "X-Signature": sig,
        "User-Agent": "project-f-summarization/1.0",
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(url, content=body_bytes, headers=headers)
            return r.status_code, r.text[:500]
    except Exception as e:
        logger.warning("Webhook delivery failed: %s", e)
        return -1, str(e)[:500]
