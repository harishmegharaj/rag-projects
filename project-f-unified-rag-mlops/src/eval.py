"""Smoke eval — extend with labeled JSONL via scripts/run_eval.py."""
from .config import project_root
from .eval_suite import load_jsonl, run_case, summarize


def main():
    path = project_root() / "data" / "eval" / "gold_qa.example.jsonl"
    if not path.is_file():
        print("No gold file at", path, "- add data/eval/*.jsonl or run scripts/run_eval.py --help")
        return
    cases = load_jsonl(path)
    results = [run_case(c) for c in cases]
    print("summary:", summarize(results))
    for r in results:
        print(r["keyword_overlap"], r["answer_preview"][:400])


if __name__ == "__main__":
    main()
