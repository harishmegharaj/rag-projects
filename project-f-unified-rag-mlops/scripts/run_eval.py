#!/usr/bin/env python3
"""Run offline eval from a JSONL gold set; emit JSON report for dashboards or CI."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_suite import load_jsonl, run_case, summarize


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate RAG pipeline on a JSONL file (id, question, expected_keywords optional)."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/eval/gold_qa.example.jsonl",
        help="Path to JSONL gold set",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the full JSON report",
    )
    args = p.parse_args()
    cases = load_jsonl(args.input)
    results = [run_case(c) for c in cases]
    report = {"summary": summarize(results), "cases": results}
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
