"""Extract plain text from supported document types."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


SUPPORTED_SUFFIXES = frozenset({".pdf", ".md", ".markdown", ".txt"})


def extract_text_from_pdf(data: bytes) -> str:
    from io import BytesIO

    reader = PdfReader(BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts).strip()


def extract_text(data: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(data)
    if suffix in {".md", ".markdown", ".txt"}:
        return data.decode("utf-8", errors="replace").strip()
    raise ValueError(f"Unsupported file type: {suffix}. Use one of {sorted(SUPPORTED_SUFFIXES)}")
