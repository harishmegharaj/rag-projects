"""Retrieve → rerank → LLM with citations."""
import json
import logging
from pathlib import Path
from typing import Any, List

from .guardrails import redact_pii
from .prompt_orchestration import run_langchain_pipeline

logger = logging.getLogger(__name__)

def _result(
    *,
    answer: str,
    retrieved: List[dict],
    blocked: bool = False,
    no_context: bool = False,
    error: bool = False,
    error_code: str | None = None,
    error_detail: str | None = None,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "retrieved": retrieved,
        "blocked": blocked,
        "no_context": no_context,
        "error": error,
        "error_code": error_code,
        "error_detail": error_detail,
    }
def run_pipeline(
    question: str,
    chroma_dir: Path,
    bm25_dir: Path,
    bm25_k: int = 20,
    vector_k: int = 20,
    fusion_top_n: int = 15,
    rerank_top_n: int = 5,
    callbacks: list | None = None,
) -> dict[str, Any]:
    logger.info("retrieval.query=%s", redact_pii(question)[:200])
    out = run_langchain_pipeline(
        question=question,
        chroma_dir=chroma_dir,
        bm25_dir=bm25_dir,
        bm25_k=bm25_k,
        vector_k=vector_k,
        fusion_top_n=fusion_top_n,
        rerank_top_n=rerank_top_n,
        callbacks=callbacks,
    )
    answer = out["answer"]
    top = out["retrieved"]

    logger.info(
        "rag.response_preview=%s",
        redact_pii(answer)[:300],
    )
    return _result(
        answer=answer,
        retrieved=top,
        blocked=out.get("blocked", False),
        no_context=out.get("no_context", False),
        error=out.get("error", False),
        error_code=out.get("error_code"),
        error_detail=out.get("error_detail"),
    )


def debug_json(result: dict) -> str:
    slim = {
        "blocked": result.get("blocked"),
        "no_context": result.get("no_context"),
        "error": result.get("error"),
        "error_code": result.get("error_code"),
        "answer": result.get("answer"),
        "retrieved": [
            {"metadata": h.get("metadata"), "rerank_score": h.get("rerank_score"), "preview": h["text"][:180]}
            for h in result.get("retrieved", [])
        ],
    }
    return json.dumps(slim, indent=2)
