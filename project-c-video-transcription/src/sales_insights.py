"""Sales-focused transcript analysis: keywords, signals, plan, and model guidance."""
from __future__ import annotations

import json
import logging
import re
from collections import Counter

from .config import openai_api_key, sales_reasoning_default_model, sales_reasoning_temperature

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]*")

_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

_SALES_TERMS = {
    "roi",
    "revenue",
    "growth",
    "pipeline",
    "qualified",
    "conversion",
    "close",
    "deal",
    "renewal",
    "upsell",
    "cross-sell",
    "acv",
    "churn",
    "retention",
    "pricing",
    "budget",
    "cost",
    "value",
    "pain",
    "problem",
    "challenge",
    "objection",
    "security",
    "compliance",
    "integration",
    "implementation",
    "timeline",
    "poc",
    "trial",
    "demo",
    "decision",
    "stakeholder",
    "procurement",
}

_SIGNAL_PATTERNS = {
    "pain_points": [
        r"\b(problem|pain point|challenge|struggle|bottleneck|manual|slow|error-prone)\b",
        r"\b(expensive|costly|wasting time|too long|slow onboarding|long onboarding)\b",
    ],
    "business_goals": [
        r"\b(grow|increase|improve|optimi[sz]e|reduce|save|scale|accelerate|faster|lower)\b",
        r"\b(revenue|margin|retention|renewal|conversion|efficiency)\b",
    ],
    "objections": [
        r"\b(price|pricing|budget|cost|expensive)\b",
        r"\b(security|compliance|legal|risk|privacy)\b",
        r"\b(integration|migration|timeline|resources|headcount)\b",
    ],
    "buying_signals": [
        r"\b(next step|follow up|send (a )?proposal|pilot|poc|trial|demo)\b",
        r"\b(procurement|contract|sign off|decision maker|stakeholder)\b",
    ],
}


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _keyword_score(term: str, freq: int) -> float:
    boost = 1.0
    if any(piece in _SALES_TERMS for piece in term.split()):
        boost += 0.6
    if len(term) >= 12:
        boost += 0.1
    return float(freq) * boost


def extract_sales_keywords(text: str, top_k: int = 15) -> list[dict]:
    if not text.strip():
        return []

    tokens = _tokenize(text)
    unigram = Counter(t for t in tokens if len(t) > 2 and t not in _STOPWORDS)

    bigram = Counter()
    trigram = Counter()
    for i in range(len(tokens) - 1):
        t1, t2 = tokens[i], tokens[i + 1]
        if t1 in _STOPWORDS or t2 in _STOPWORDS:
            continue
        bigram[f"{t1} {t2}"] += 1
    for i in range(len(tokens) - 2):
        t1, t2, t3 = tokens[i], tokens[i + 1], tokens[i + 2]
        if t1 in _STOPWORDS or t2 in _STOPWORDS or t3 in _STOPWORDS:
            continue
        trigram[f"{t1} {t2} {t3}"] += 1

    candidates: Counter[str] = Counter()
    for term, freq in unigram.items():
        if freq >= 2 or term in _SALES_TERMS:
            candidates[term] = freq
    for term, freq in bigram.items():
        if freq >= 2 or any(piece in _SALES_TERMS for piece in term.split()):
            candidates[term] = max(candidates[term], freq)
    for term, freq in trigram.items():
        if freq >= 2:
            candidates[term] = max(candidates[term], freq)

    ranked = sorted(
        (
            {
                "term": term,
                "frequency": freq,
                "score": round(_keyword_score(term, freq), 3),
            }
            for term, freq in candidates.items()
        ),
        key=lambda x: (x["score"], x["frequency"], len(x["term"])),
        reverse=True,
    )
    return ranked[: max(1, min(top_k, 50))]


def detect_sales_signals(text: str) -> dict:
    lower = text.lower()
    out: dict[str, list[str]] = {}
    for bucket, patterns in _SIGNAL_PATTERNS.items():
        hits: list[str] = []
        for pat in patterns:
            for match in re.finditer(pat, lower):
                phrase = match.group(0)
                if phrase not in hits:
                    hits.append(phrase)
        out[bucket] = hits
    return out


def _top_terms_for_plan(keywords: list[dict], n: int = 6) -> list[str]:
    return [k["term"] for k in keywords[:n] if isinstance(k, dict) and k.get("term")]


def build_sales_pitch_plan(transcript_text: str, keywords: list[dict], signals: dict) -> list[dict]:
    top_terms = _top_terms_for_plan(keywords)
    pains = signals.get("pain_points", [])[:3]
    goals = signals.get("business_goals", [])[:3]
    objections = signals.get("objections", [])[:3]
    buying = signals.get("buying_signals", [])[:3]

    return [
        {
            "phase": "1. Discovery Summary",
            "objective": "Align sales narrative to buyer pain and business priorities.",
            "actions": [
                f"Open with pain statements heard in call: {', '.join(pains) if pains else 'no explicit pains captured'}.",
                f"Tie outcomes to goals: {', '.join(goals) if goals else 'confirm target KPI and timeline'}.",
                "Confirm buying committee and success criteria before proposing solution depth.",
            ],
        },
        {
            "phase": "2. Value Positioning",
            "objective": "Translate product capabilities into measurable business value.",
            "actions": [
                f"Position around transcript keywords: {', '.join(top_terms) if top_terms else 'core product value and ROI'}.",
                "For each key capability, map current-state friction -> future-state metric improvement.",
                "Quantify expected ROI with baseline assumptions and 30/60/90-day milestones.",
            ],
        },
        {
            "phase": "3. Objection Handling",
            "objective": "Address risks early with evidence and implementation clarity.",
            "actions": [
                f"Prepare objection responses for: {', '.join(objections) if objections else 'budget, security, and integration'}.",
                "Share deployment architecture, security controls, and integration path with estimated effort.",
                "Offer a low-risk evaluation path (pilot or phased rollout) with clear exit criteria.",
            ],
        },
        {
            "phase": "4. Close Plan",
            "objective": "Convert momentum into committed next actions.",
            "actions": [
                f"Anchor next meeting around buying signals: {', '.join(buying) if buying else 'proposal review and decision timeline'}.",
                "Send recap within 24 hours with owners, dates, and open questions.",
                "Book executive/technical follow-up to remove final blockers and define close date.",
            ],
        },
    ]


def recommend_models_for_sales_use_case() -> list[dict]:
    return [
        {
            "stage": "Transcription (speaker-aware calls)",
            "recommended_model": "deepgram:nova-2",
            "why": "Strong diarization and turn-level timestamps, ideal for multi-speaker sales meetings.",
        },
        {
            "stage": "Transcription (highest neutral baseline)",
            "recommended_model": "openai:whisper-1",
            "why": "Reliable speech-to-text quality across accents and noisy business calls.",
        },
        {
            "stage": "Transcript reasoning + plan generation",
            "recommended_model": "gpt-4.1-mini",
            "why": "Good quality/cost balance for keyword extraction, call summarization, and sales action plans.",
        },
        {
            "stage": "Transcript reasoning (best quality)",
            "recommended_model": "gpt-4.1",
            "why": "Best option when nuanced objection handling and executive-ready plans are required.",
        },
        {
            "stage": "Fully local/private deployment",
            "recommended_model": "llama-3.1-8b-instruct (or larger)",
            "why": "Viable on-prem option when strict data residency blocks cloud-hosted LLMs.",
        },
    ]


def analyze_transcript_for_sales(transcript_text: str, top_k: int = 15) -> dict:
    keywords = extract_sales_keywords(transcript_text, top_k=top_k)
    signals = detect_sales_signals(transcript_text)
    plan = build_sales_pitch_plan(transcript_text, keywords, signals)
    return {
        "summary": {
            "transcript_chars": len(transcript_text),
            "keyword_count": len(keywords),
        },
        "keywords": keywords,
        "signals": signals,
        "sales_plan": plan,
        "recommended_models": recommend_models_for_sales_use_case(),
        "reasoning": {
            "mode": "heuristic",
            "model": None,
        },
    }


def _llm_messages(transcript_text: str, top_k: int) -> list[dict]:
    schema_hint = {
        "summary": {
            "transcript_chars": "int",
            "keyword_count": "int",
            "key_takeaway": "string",
        },
        "keywords": [
            {
                "term": "string",
                "frequency": "int",
                "score": "float",
            }
        ],
        "signals": {
            "pain_points": ["string"],
            "business_goals": ["string"],
            "objections": ["string"],
            "buying_signals": ["string"],
        },
        "sales_plan": [
            {
                "phase": "string",
                "objective": "string",
                "actions": ["string", "string", "string"],
            }
        ],
    }
    system = (
        "You are a B2B sales strategist and NLP analyst. "
        "Extract practical sales insights from meeting transcripts. "
        "Return strict JSON only."
    )
    user = (
        "Analyze the transcript and return JSON with this exact top-level structure: "
        f"{json.dumps(schema_hint)}. "
        f"Return up to {top_k} keywords sorted by importance. "
        "Keep actions concrete and near-term.\n\n"
        f"Transcript:\n{transcript_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _normalize_llm_output(payload: dict, transcript_text: str, top_k: int) -> dict:
    summary = payload.get("summary") if isinstance(payload, dict) else {}
    if not isinstance(summary, dict):
        summary = {}

    raw_keywords = payload.get("keywords") if isinstance(payload, dict) else []
    keywords: list[dict] = []
    if isinstance(raw_keywords, list):
        for item in raw_keywords[: max(1, min(top_k, 50))]:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            if not term:
                continue
            try:
                frequency = int(item.get("frequency", 1))
            except (TypeError, ValueError):
                frequency = 1
            try:
                score = float(item.get("score", 1.0))
            except (TypeError, ValueError):
                score = 1.0
            keywords.append(
                {
                    "term": term,
                    "frequency": max(1, frequency),
                    "score": round(max(0.0, score), 3),
                }
            )

    signals = payload.get("signals") if isinstance(payload, dict) else {}
    if not isinstance(signals, dict):
        signals = {}
    norm_signals: dict[str, list[str]] = {
        "pain_points": [str(x) for x in signals.get("pain_points", []) if str(x).strip()],
        "business_goals": [str(x) for x in signals.get("business_goals", []) if str(x).strip()],
        "objections": [str(x) for x in signals.get("objections", []) if str(x).strip()],
        "buying_signals": [str(x) for x in signals.get("buying_signals", []) if str(x).strip()],
    }

    raw_plan = payload.get("sales_plan") if isinstance(payload, dict) else []
    plan: list[dict] = []
    if isinstance(raw_plan, list):
        for item in raw_plan:
            if not isinstance(item, dict):
                continue
            phase = str(item.get("phase", "")).strip()
            objective = str(item.get("objective", "")).strip()
            actions_raw = item.get("actions", [])
            actions = [str(x).strip() for x in actions_raw if str(x).strip()] if isinstance(actions_raw, list) else []
            if phase and objective and actions:
                plan.append(
                    {
                        "phase": phase,
                        "objective": objective,
                        "actions": actions[:5],
                    }
                )

    key_takeaway = str(summary.get("key_takeaway", "")).strip()
    out = {
        "summary": {
            "transcript_chars": len(transcript_text),
            "keyword_count": len(keywords),
            "key_takeaway": key_takeaway,
        },
        "keywords": keywords,
        "signals": norm_signals,
        "sales_plan": plan,
        "recommended_models": recommend_models_for_sales_use_case(),
    }

    if not out["keywords"] or not out["sales_plan"]:
        fallback = analyze_transcript_for_sales(transcript_text, top_k=top_k)
        fallback["reasoning"] = {
            "mode": "heuristic_fallback",
            "model": None,
        }
        return fallback

    return out


def analyze_transcript_for_sales_llm(
    transcript_text: str,
    top_k: int = 15,
    reasoning_model: str | None = None,
) -> dict:
    if not transcript_text.strip():
        return analyze_transcript_for_sales(transcript_text, top_k=top_k)

    key = openai_api_key()
    if not key:
        raise ValueError("OPENAI_API_KEY is required when use_llm=true")

    model = (reasoning_model or sales_reasoning_default_model()).strip()
    if not model:
        model = "gpt-4.1-mini"

    from openai import OpenAI

    client = OpenAI(api_key=key)
    completion = client.chat.completions.create(
        model=model,
        temperature=sales_reasoning_temperature(),
        response_format={"type": "json_object"},
        messages=_llm_messages(transcript_text, top_k),
    )
    content = (completion.choices[0].message.content or "{}").strip()

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("LLM output was not valid JSON, using heuristic fallback: %s", e)
        fallback = analyze_transcript_for_sales(transcript_text, top_k=top_k)
        fallback["reasoning"] = {
            "mode": "heuristic_fallback",
            "model": model,
            "error": "invalid_json_from_llm",
        }
        return fallback

    out = _normalize_llm_output(payload, transcript_text, top_k)
    out["reasoning"] = {
        "mode": "llm",
        "model": model,
    }
    return out
