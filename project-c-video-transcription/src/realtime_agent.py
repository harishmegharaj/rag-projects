"""Realtime memory + relation graph + lightweight agent tools."""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]{1,}")
ENTITY_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")

SYNONYM_TO_CANONICAL = {
    "pricing": "cost",
    "price": "cost",
    "expensive": "cost",
    "cheap": "cost",
    "budget": "cost",
    "onboarding": "adoption",
    "adoption": "adoption",
    "rollout": "deployment",
    "deployment": "deployment",
    "integrate": "integration",
    "integration": "integration",
    "secure": "security",
    "security": "security",
    "renewal": "retention",
    "retention": "retention",
}


@dataclass
class MessageNode:
    message_id: str
    session_id: str
    role: str
    text: str
    ts: float
    embedding: list[float]
    entities: set[str]


@dataclass
class RelationEdge:
    src: str
    dst: str
    weight: float
    reason: str


@dataclass
class SessionState:
    messages: list[MessageNode] = field(default_factory=list)
    edges: list[RelationEdge] = field(default_factory=list)
    subscribers: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)


class InMemorySemanticStore:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.sessions: dict[str, SessionState] = {}

    def get_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState()
        return self.sessions[session_id]

    def _canonicalize(self, token: str) -> str:
        t = token.lower()
        return SYNONYM_TO_CANONICAL.get(t, t)

    def _tokenize(self, text: str) -> list[str]:
        return [self._canonicalize(m.group(0)) for m in TOKEN_RE.finditer(text)]

    def _hash_index(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % self.dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        return idx, sign

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = self._tokenize(text)
        if not tokens:
            return vec
        for tok in tokens:
            idx, sign = self._hash_index(tok)
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    @staticmethod
    def extract_entities(text: str) -> set[str]:
        raw = {m.group(1).strip() for m in ENTITY_RE.finditer(text)}
        return {x for x in raw if len(x) >= 3}

    def add_message(self, session_id: str, role: str, text: str) -> MessageNode:
        msg = MessageNode(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            text=text,
            ts=time.time(),
            embedding=self.embed(text),
            entities=self.extract_entities(text),
        )
        st = self.get_session(session_id)
        st.messages.append(msg)

        for prev in st.messages[-51:-1]:
            sim = self.cosine(msg.embedding, prev.embedding)
            shared = msg.entities.intersection(prev.entities)
            if sim >= 0.55:
                st.edges.append(RelationEdge(src=msg.message_id, dst=prev.message_id, weight=sim, reason="semantic"))
            elif shared:
                st.edges.append(
                    RelationEdge(
                        src=msg.message_id,
                        dst=prev.message_id,
                        weight=0.45,
                        reason=f"shared_entity:{','.join(sorted(shared)[:3])}",
                    )
                )
        return msg

    def top_related_messages(self, session_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        st = self.get_session(session_id)
        q_vec = self.embed(query)
        scored: list[tuple[float, MessageNode]] = []
        for m in st.messages:
            sim = self.cosine(q_vec, m.embedding)
            scored.append((sim, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for sim, m in scored[:top_k]:
            out.append(
                {
                    "message_id": m.message_id,
                    "role": m.role,
                    "text": m.text,
                    "score": round(sim, 4),
                    "ts": datetime.fromtimestamp(m.ts, timezone.utc).isoformat(),
                }
            )
        return out

    def relation_summary(self, session_id: str, max_edges: int = 8) -> list[dict[str, Any]]:
        st = self.get_session(session_id)
        edges = sorted(st.edges, key=lambda e: e.weight, reverse=True)[:max_edges]
        return [
            {
                "from": e.src,
                "to": e.dst,
                "weight": round(e.weight, 4),
                "reason": e.reason,
            }
            for e in edges
        ]

    def rolling_topics(self, session_id: str, window: int = 50, top_n: int = 8) -> list[str]:
        st = self.get_session(session_id)
        recent = st.messages[-window:]
        freq: dict[str, int] = {}
        for m in recent:
            for tok in self._tokenize(m.text):
                if len(tok) < 3:
                    continue
                freq[tok] = freq.get(tok, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in ranked[:top_n]]


class RealtimeAgent:
    def __init__(self, store: InMemorySemanticStore) -> None:
        self.store = store

    def plan(self, user_text: str) -> list[str]:
        tools = ["semantic_search", "relation_graph", "topic_window"]
        if "plan" in user_text.lower() or "next step" in user_text.lower():
            tools.append("action_plan")
        return tools

    def execute_tools(self, session_id: str, user_text: str, tools: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for t in tools:
            if t == "semantic_search":
                out[t] = self.store.top_related_messages(session_id, user_text, top_k=5)
            elif t == "relation_graph":
                out[t] = self.store.relation_summary(session_id, max_edges=8)
            elif t == "topic_window":
                out[t] = self.store.rolling_topics(session_id, window=50, top_n=8)
            elif t == "action_plan":
                topics = self.store.rolling_topics(session_id, window=50, top_n=6)
                out[t] = [
                    "Prioritize top two recurring concerns from last 50 messages.",
                    "Map each concern to one measurable KPI and owner.",
                    f"Prepare next-call agenda using themes: {', '.join(topics) if topics else 'discovery, validation, close'}.",
                ]
        return out

    def respond(self, session_id: str, user_text: str) -> dict[str, Any]:
        self.store.add_message(session_id, role="user", text=user_text)
        plan = self.plan(user_text)
        observations = self.execute_tools(session_id, user_text, plan)

        topics = observations.get("topic_window", [])
        related = observations.get("semantic_search", [])
        top_related = related[0]["text"] if related else ""

        answer_lines = [
            "Realtime analysis complete.",
            f"Dominant themes (last 50 messages): {', '.join(topics) if topics else 'insufficient history'}.",
            f"Closest related message: {top_related[:140] if top_related else 'none yet'}",
            "I can now generate a structured action plan if you ask for 'next steps'.",
        ]
        assistant_text = "\n".join(answer_lines)
        self.store.add_message(session_id, role="assistant", text=assistant_text)

        return {
            "answer": assistant_text,
            "tools_used": plan,
            "observations": observations,
        }


class RealtimeHub:
    def __init__(self, store: InMemorySemanticStore) -> None:
        self.store = store

    async def publish(self, session_id: str, event: dict[str, Any]) -> None:
        st = self.store.get_session(session_id)
        dead: list[asyncio.Queue[dict[str, Any]]] = []
        for q in st.subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        if dead:
            st.subscribers = [q for q in st.subscribers if q not in dead]

    def subscribe(self, session_id: str, maxsize: int = 200) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        st = self.store.get_session(session_id)
        st.subscribers.append(q)
        return q

    def unsubscribe(self, session_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        st = self.store.get_session(session_id)
        st.subscribers = [x for x in st.subscribers if x is not q]


realtime_store = InMemorySemanticStore(dim=256)
realtime_agent = RealtimeAgent(realtime_store)
realtime_hub = RealtimeHub(realtime_store)
