"""LangChain prompt orchestration for enterprise RAG query flow."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI

from . import config
from .guardrails import check_query
from .hybrid_retrieve import hybrid_retrieve
from .rerank import rerank

NO_CONTEXT_ANSWER = (
    "No passages were retrieved from the indexed documents for this question. "
    "The topic may not be covered in your corpus, or the index may be empty—try rephrasing, "
    "adding relevant documents, or running a full reindex."
)


class PipelineState(TypedDict, total=False):
    question: str
    redacted_question: str
    chroma_dir: Path
    bm25_dir: Path
    bm25_k: int
    vector_k: int
    fusion_top_n: int
    rerank_top_n: int
    blocked: bool
    block_reason: str
    no_context: bool
    error: bool
    error_code: str | None
    error_detail: str | None
    candidates: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    context: str
    answer: str


def format_context(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for i, h in enumerate(chunks, start=1):
        meta = h.get("metadata") or {}
        src = meta.get("source", "unknown")
        page = meta.get("page")
        page_s = f" p.{page}" if page is not None and page >= 0 else ""
        blocks.append(f"[{i}] (source: {src}{page_s})\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def _guardrails_step(state: PipelineState) -> PipelineState:
    question = state["question"]
    g = check_query(question)
    next_state: PipelineState = dict(state)
    if not g.ok:
        next_state.update(
            blocked=True,
            block_reason=g.reason or "policy_violation",
            answer=f"I cannot process this request ({g.reason})",
            retrieved=[],
        )
        return next_state
    q = g.redacted_query or question
    next_state.update(redacted_question=q)
    return next_state


def _retrieve_step(state: PipelineState) -> PipelineState:
    if state.get("blocked"):
        return state
    q = state["redacted_question"]
    try:
        candidates = hybrid_retrieve(
            q,
            chroma_dir=state["chroma_dir"],
            bm25_dir=state["bm25_dir"],
            bm25_k=state["bm25_k"],
            vector_k=state["vector_k"],
            fusion_top_n=state["fusion_top_n"],
        )
    except FileNotFoundError as e:
        out = dict(state)
        out.update(
            error=True,
            error_code="index_missing",
            error_detail=str(e),
            answer="The search index is missing or incomplete. Build indexes (e.g. run build_index or POST /v1/index/rebuild) and try again.",
            retrieved=[],
        )
        return out
    except Exception as e:  # pragma: no cover - defensive for runtime issues
        out = dict(state)
        out.update(
            error=True,
            error_code="retrieval_failed",
            error_detail=str(e),
            answer="Search failed due to an unexpected error. Check logs, verify indexes, and try again.",
            retrieved=[],
        )
        return out
    out = dict(state)
    out["candidates"] = candidates
    return out


def _rerank_step(state: PipelineState) -> PipelineState:
    if state.get("blocked") or state.get("error"):
        return state
    q = state["redacted_question"]
    top = rerank(q, state.get("candidates", []), top_n=state["rerank_top_n"])
    out = dict(state)
    out["retrieved"] = top
    if not top:
        out.update(no_context=True, answer=NO_CONTEXT_ANSWER, context="")
        return out
    out["context"] = format_context(top)
    return out


def _should_short_circuit(state: PipelineState) -> bool:
    return bool(state.get("blocked") or state.get("error") or state.get("no_context"))


def _short_circuit_answer(state: PipelineState) -> PipelineState:
    return state


def _generate_answer(state: PipelineState) -> PipelineState:
    q = state["redacted_question"]
    context = state["context"]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful assistant. Answer only using the provided context. "
                "If the context is insufficient to answer, say clearly that you do not have enough information in the sources. "
                "Do not invent facts. Cite sources using [1], [2] notation.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        answer = (
            "[Stub — set OPENAI_API_KEY in .env]\n\n"
            f"After rerank, top chunk preview:\n{state['retrieved'][0]['text'][:400]}"
        )
        out = dict(state)
        out["answer"] = answer
        return out

    try:
        model = ChatOpenAI(
            model=config.openai_chat_model(),
            temperature=0.2,
            api_key=api_key,
        )
        chain = prompt | model | StrOutputParser()
        answer = chain.invoke({"context": context, "question": q})
    except Exception as e:  # pragma: no cover - maps runtime provider failures
        msg = str(e).lower()
        if "rate limit" in msg:
            code = "llm_rate_limit"
            answer = "The language model service is temporarily busy (rate limited). Please try again shortly."
        elif "timeout" in msg or "connection" in msg or "network" in msg:
            code = "llm_network"
            answer = "Could not reach the language model service (network error or timeout). Check connectivity and try again."
        else:
            code = "llm_upstream"
            answer = "The language model service returned an error and could not complete the request. Please try again later."
        out = dict(state)
        out.update(error=True, error_code=code, error_detail=str(e), answer=answer)
        return out

    out = dict(state)
    out["answer"] = answer or ""
    return out


def build_chain() -> Runnable[PipelineState, PipelineState]:
    return (
        RunnableLambda(_guardrails_step)
        | RunnableLambda(_retrieve_step)
        | RunnableLambda(_rerank_step)
        | RunnableBranch(
            (_should_short_circuit, RunnableLambda(_short_circuit_answer)),
            RunnableLambda(_generate_answer),
        )
    )


def run_langchain_pipeline(
    question: str,
    chroma_dir: Path,
    bm25_dir: Path,
    bm25_k: int,
    vector_k: int,
    fusion_top_n: int,
    rerank_top_n: int,
    callbacks: list | None = None,
) -> dict[str, Any]:
    chain = build_chain()
    state: PipelineState = {
        "question": question,
        "chroma_dir": chroma_dir,
        "bm25_dir": bm25_dir,
        "bm25_k": bm25_k,
        "vector_k": vector_k,
        "fusion_top_n": fusion_top_n,
        "rerank_top_n": rerank_top_n,
        "blocked": False,
        "no_context": False,
        "error": False,
        "error_code": None,
        "error_detail": None,
        "retrieved": [],
    }
    out = chain.invoke(state, config={"callbacks": callbacks or []})
    if out.get("redacted_question"):
        # Keep retrieval query logging in one place for consistency with existing output.
        pass
    return {
        "answer": out.get("answer", ""),
        "retrieved": out.get("retrieved", []),
        "blocked": out.get("blocked", False),
        "no_context": out.get("no_context", False),
        "error": out.get("error", False),
        "error_code": out.get("error_code"),
        "error_detail": out.get("error_detail"),
    }


