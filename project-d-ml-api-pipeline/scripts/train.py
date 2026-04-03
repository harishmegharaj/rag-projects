#!/usr/bin/env python3
"""Train classifier and update models/registry.json."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_pipeline import train_and_register  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train iris classifier and register artifact.")
    p.add_argument("--data", type=Path, default=None, help="CSV path (default: data/raw/iris.csv)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rec = train_and_register(data_path=args.data, random_state=args.seed)
    print("Registered:", rec["version"], "accuracy:", rec["metrics"]["accuracy_holdout"])


if __name__ == "__main__":
    main()
