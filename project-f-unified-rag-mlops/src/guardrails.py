"""Lightweight input/output checks — extend for your policies."""
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class GuardResult:
    ok: bool
    reason: Optional[str] = None
    redacted_query: Optional[str] = None


# Very naive patterns for learning — replace with real DLP/compliance tooling in production.
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.I)
_PHONE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")


def redact_pii(text: str) -> str:
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _PHONE.sub("[REDACTED_PHONE]", text)
    return text


def check_query(q: str, min_len: int = 3, max_len: int = 4000) -> GuardResult:
    s = (q or "").strip()
    if len(s) < min_len:
        return GuardResult(False, "Query too short or empty.")
    if len(s) > max_len:
        return GuardResult(False, "Query too long.")
    blocked = ("ignore previous", "system prompt", "jailbreak")
    low = s.lower()
    if any(b in low for b in blocked):
        return GuardResult(False, "Query blocked by policy.")
    return GuardResult(True, redacted_query=redact_pii(s))
