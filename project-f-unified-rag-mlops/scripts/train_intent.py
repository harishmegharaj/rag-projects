#!/usr/bin/env python3
"""Train intent classifier and update models/intent_registry.json."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.intent_model import clear_intent_cache  # noqa: E402
from src.intent_train_pipeline import train_and_register_intent  # noqa: E402


def main() -> None:
    rec = train_and_register_intent()
    clear_intent_cache()
    print("registered:", rec["version"])
    print("artifact:", rec["artifact"])
    print("metrics:", rec["metrics"])


if __name__ == "__main__":
    main()
