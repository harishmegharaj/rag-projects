"""
Load documents from data/raw, chunk with metadata (source, page where applicable).
"""
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_pdf_pages(path: Path) -> List[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: List[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_pdf(path: Path, chunk_size: int = 800, chunk_overlap: int = 120) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    docs: List[dict] = []
    source = path.name
    for page_num, text in load_pdf_pages(path):
        for chunk in splitter.create_documents([text], metadatas=[{"source": source, "page": page_num}]):
            docs.append(
                {
                    "text": chunk.page_content,
                    "metadata": {**chunk.metadata, "source": source, "page": page_num},
                }
            )
    return docs


def load_markdown(path: Path, chunk_size: int = 800, chunk_overlap: int = 120) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text = path.read_text(encoding="utf-8", errors="replace")
    source = path.name
    docs: List[dict] = []
    for chunk in splitter.create_documents([text], metadatas=[{"source": source, "page": None}]):
        docs.append({"text": chunk.page_content, "metadata": {**chunk.metadata, "source": source}})
    return docs


def ingest_directory(raw_dir: Path) -> List[dict]:
    all_chunks: List[dict] = []
    for path in sorted(raw_dir.glob("**/*")):
        if path.suffix.lower() == ".pdf":
            all_chunks.extend(chunk_pdf(path))
        elif path.suffix.lower() in {".md", ".markdown"}:
            all_chunks.extend(load_markdown(path))
    return all_chunks
