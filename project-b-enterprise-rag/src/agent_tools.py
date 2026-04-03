"""LangChain tools for the optional ReAct agent: corpus search, safe arithmetic, allowlisted HTTP GET."""

from __future__ import annotations

import ast
import operator
import os
from pathlib import Path
from urllib.parse import urlparse

import httpx
from langchain_core.tools import tool

from .hybrid_retrieve import hybrid_retrieve
from .prompt_orchestration import format_context
from .rerank import rerank

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _eval_math(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_math(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_math(node.left), _eval_math(node.right))
    raise ValueError("Only numbers and + - * / ** with parentheses are allowed.")


def _http_allowlist() -> set[str]:
    raw = os.getenv("AGENT_HTTP_ALLOWLIST", "127.0.0.1,localhost")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def build_agent_tools(
    chroma_dir: Path,
    bm25_dir: Path,
    *,
    bm25_k: int = 20,
    vector_k: int = 20,
    fusion_top_n: int = 15,
    rerank_top_n: int = 5,
) -> list:
    """Build tool callables bound to index paths (closure over chroma/bm25 dirs)."""

    @tool
    def search_documents(query: str) -> str:
        """Search the indexed document corpus (hybrid lexical + dense retrieval, then reranking).

        Use for questions about policies, products, or facts likely contained in uploaded/indexed files.
        Pass a short, focused natural-language query (not a full conversation).
        """
        q = (query or "").strip()
        if not q:
            return "Empty query."
        try:
            candidates = hybrid_retrieve(
                q,
                chroma_dir=chroma_dir,
                bm25_dir=bm25_dir,
                bm25_k=bm25_k,
                vector_k=vector_k,
                fusion_top_n=fusion_top_n,
            )
        except FileNotFoundError as e:
            return f"Index missing: {e}. Build indexes (e.g. scripts/build_index.py) and retry."
        except Exception as e:  # noqa: BLE001 — surface errors to the model as text
            return f"Retrieval failed: {e}"
        if not candidates:
            return "No passages retrieved."
        top = rerank(q, candidates, top_n=rerank_top_n)
        return format_context(top)

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a safe arithmetic expression: numbers, + - * / **, parentheses.

        Example: (2 + 3) * 4
        """
        expr = (expression or "").strip()
        if not expr:
            return "Empty expression."
        try:
            tree = ast.parse(expr, mode="eval")
            value = _eval_math(tree)
        except ZeroDivisionError:
            return "Error: division by zero."
        except (ValueError, SyntaxError, TypeError) as e:
            return f"Error: {e}"
        if abs(value - round(value)) < 1e-12:
            return str(int(round(value)))
        return f"{value:.12g}"

    @tool
    def fetch_internal_http(url: str) -> str:
        """HTTP GET a URL allowed by AGENT_HTTP_ALLOWLIST (comma-separated hostnames).

        Default allowlist: 127.0.0.1,localhost. Use for small internal JSON/health endpoints, not large downloads.
        """
        u = (url or "").strip()
        if not u:
            return "Empty URL."
        parsed = urlparse(u)
        if parsed.scheme not in ("http", "https"):
            return "Only http and https are allowed."
        host = (parsed.hostname or "").lower()
        if host not in _http_allowlist():
            return (
                f"Host {host!r} is not allowlisted. "
                f"Set AGENT_HTTP_ALLOWLIST (e.g. '127.0.0.1,localhost,api.internal')."
            )
        try:
            r = httpx.get(u, timeout=10.0, follow_redirects=False)
            body = (r.text or "")[:8000]
            return f"HTTP {r.status_code}\n\n{body}"
        except Exception as e:
            return f"Request failed: {e}"

    return [search_documents, calculator, fetch_internal_http]
