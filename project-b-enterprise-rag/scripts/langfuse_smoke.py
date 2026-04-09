#!/usr/bin/env python
"""Send three compatibility-safe smoke traces to the local Langfuse instance."""
from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env (if present) before reading env vars
# ---------------------------------------------------------------------------
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file, override=False)  # shell exports take priority
        print(f"[smoke] Loaded {_env_file}")
    except ImportError:
        pass  # python-dotenv not installed; fall back to shell env

# ---------------------------------------------------------------------------
# Env check
# ---------------------------------------------------------------------------
pub = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
sec = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
host = os.getenv("LANGFUSE_HOST", "http://localhost:3000").strip()

if not pub or not sec:
    print(
        "[smoke] ERROR: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set.\n"
        f"       Open {host} → Settings → API keys, then:\n"
        "         export LANGFUSE_PUBLIC_KEY=pk-lf-...\n"
        "         export LANGFUSE_SECRET_KEY=sk-lf-..."
    )
    sys.exit(1)

try:
    import langfuse
    from langfuse import Langfuse
except ImportError:
    print("[smoke] ERROR: langfuse not installed. Run: pip install langfuse")
    sys.exit(1)

print(f"[smoke] Connecting to {host} (langfuse v{langfuse.__version__})")

lf = Langfuse(public_key=pub, secret_key=sec, host=host)

# ---------------------------------------------------------------------------
# Connection check
# ---------------------------------------------------------------------------
try:
    ok = lf.auth_check()
    print(f"[smoke] auth_check: {ok}")
except Exception as exc:
    print(f"[smoke] auth_check FAILED: {exc}")
    print(f"       Is Langfuse running at {host}?  docker compose -f docker-compose.langfuse.yml up -d")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Trace 1 — simple RAG pipeline simulation
# ---------------------------------------------------------------------------
trace_id_1 = str(uuid.uuid4())
print(f"\n[smoke] Sending trace 1  (id={trace_id_1})")

trace_1 = lf.trace(
    id=trace_id_1,
    name="rag-pipeline-smoke",
    input={"question": "What is Enterprise RAG?"},
    output={"answer": "Enterprise RAG combines hybrid retrieval, reranking, and LLM generation for robust QA."},
    metadata={"source": "langfuse_smoke.py", "run": 1},
)
time.sleep(0.05)

guard_1 = lf.span(
    trace_id=trace_id_1,
    name="guardrails",
    input={"query": "What is Enterprise RAG?"},
    output={"ok": True, "redacted_query": "What is Enterprise RAG?"},
)
time.sleep(0.02)

retrieval_1 = lf.span(
    trace_id=trace_id_1,
    name="hybrid-retrieval",
    input={"query": "What is Enterprise RAG?"},
    output={"chunks_returned": 3, "sources": ["doc1.pdf", "doc2.md", "doc3.pdf"]},
)
time.sleep(0.04)

rerank_1 = lf.span(
    trace_id=trace_id_1,
    name="rerank",
    input={"top_k": 5},
    output={"reranked_top_n": 3},
)
time.sleep(0.02)

generation_1 = lf.generation(
    trace_id=trace_id_1,
    name="llm-generation",
    input={"prompt": "Summarize the retrieved context about Enterprise RAG."},
    output="Enterprise RAG combines hybrid retrieval, reranking, and LLM generation for robust QA.",
    model="gpt-4o-mini",
    model_parameters={"temperature": 0.2},
    usage_details={"input": 120, "output": 30, "total": 150},
)

print(f"[smoke] Trace 1 sent  → trace_id={trace_id_1}")

# ---------------------------------------------------------------------------
# Trace 2 — blocked outcome
# ---------------------------------------------------------------------------
trace_id_2 = str(uuid.uuid4())
print(f"\n[smoke] Sending trace 2 (error/blocked)  (id={trace_id_2})")

trace_2 = lf.trace(
    id=trace_id_2,
    name="rag-pipeline-smoke",
    input={"question": "Tell me how to do something harmful"},
    output={"blocked": True, "reason": "policy_violation"},
    metadata={"source": "langfuse_smoke.py", "run": 2},
)
guard_2 = lf.span(
    trace_id=trace_id_2,
    name="guardrails",
    input={"query": "Tell me how to do something harmful"},
    output={"ok": False, "reason": "policy_violation"},
    level="WARNING",
)

print(f"[smoke] Trace 2 sent  → trace_id={trace_id_2}")

# ---------------------------------------------------------------------------
# Trace 3 — no_context outcome
# ---------------------------------------------------------------------------
trace_id_3 = str(uuid.uuid4())
print(f"\n[smoke] Sending trace 3 (no_context)  (id={trace_id_3})")

trace_3 = lf.trace(
    id=trace_id_3,
    name="rag-pipeline-smoke",
    input={"question": "What is the capital of Mars?"},
    output={"no_context": True, "answer": "No relevant documents found."},
    metadata={"source": "langfuse_smoke.py", "run": 3},
)
guard_3 = lf.span(
    trace_id=trace_id_3,
    name="guardrails",
    input={"query": "What is the capital of Mars?"},
    output={"ok": True},
)
retrieval_3 = lf.span(
    trace_id=trace_id_3,
    name="hybrid-retrieval",
    input={"query": "What is the capital of Mars?"},
    output={"chunks_returned": 0},
)

print(f"[smoke] Trace 3 sent  → trace_id={trace_id_3}")

# ---------------------------------------------------------------------------
# Flush and summarise
# ---------------------------------------------------------------------------
print("\n[smoke] Flushing to Langfuse …")
lf.flush()
print("[smoke] Done!\n")
print("Open Langfuse and click Traces to see the 3 smoke traces:")
print(f"  {host}/traces")
print("Filter by source=langfuse_smoke.py or search for the trace IDs printed above.")
lf.shutdown()
