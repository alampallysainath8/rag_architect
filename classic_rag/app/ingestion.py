"""
Ingestion module — loads a document (PDF/TXT) and splits it into chunks.
"""

import os
import re
from typing import List, Dict
from pypdf import PdfReader

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Text Extraction ──────────────────────────────────────────────────────────

def extract_pages_from_pdf(file_path: str) -> List[Dict]:
    """Extract text from each page of a PDF. Returns [{page: int, text: str}, ...]."""
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def extract_pages_from_txt(file_path: str) -> List[Dict]:
    """Read plain-text file as a single page."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [{"page": 1, "text": f.read()}]


def extract_pages(file_path: str) -> List[Dict]:
    """Extract pages from a file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_pages_from_pdf(file_path)
    elif ext in (".txt", ".md"):
        return extract_pages_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Chunking ─────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise whitespace."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Chunk text from pages while tracking which page(s) each chunk came from.
    Returns [{"chunk_text": str, "pages": [int, ...]}, ...].
    """
    # Build a flat character stream and a parallel array mapping each char → page number
    full_text = ""
    char_to_page: List[int] = []
    for p in pages:
        cleaned = clean_text(p["text"])
        if cleaned:
            if full_text:
                full_text += " "
                char_to_page.append(p["page"])
            full_text += cleaned
            char_to_page.extend([p["page"]] * len(cleaned))

    chunks: List[Dict] = []
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk = full_text[start:end].strip()
        if chunk:
            # Pages spanned by this chunk
            page_set = sorted(set(char_to_page[start:end]))
            chunks.append({"chunk_text": chunk, "pages": page_set})
        start += chunk_size - overlap
    return chunks


# ── Public API ───────────────────────────────────────────────────────────────

def ingest_document(file_path: str) -> List[Dict]:
    """
    Full ingestion pipeline for a single document.
    Returns a list of dicts with keys: id, chunk_text, source, pages.
    """
    file_name = os.path.basename(file_path)
    pages = extract_pages(file_path)
    chunks = chunk_pages(pages)

    records = []
    for idx, chunk in enumerate(chunks):
        page_str = ",".join(str(p) for p in chunk["pages"])
        records.append(
            {
                "id": f"{file_name}::chunk-{idx}",
                "chunk_text": chunk["chunk_text"],
                "source": file_name,
                "pages": page_str,
            }
        )

    print(f"✅ Ingested '{file_name}' → {len(records)} chunks")
    return records


if __name__ == "__main__":
    import sys

    print("=== Ingestion Test ===")
    test_path = sys.argv[1] if len(sys.argv) > 1 else "docs/Apple_Q24.pdf"
    print(f"Testing with: {test_path}")

    records = ingest_document(test_path)
    print(f"Total chunks: {len(records)}")
    print(f"\nFirst chunk preview:")
    print(f"  ID   : {records[0]['id']}")
    print(f"  Source: {records[0]['source']}")
    print(f"  Text : {records[0]['chunk_text'][:200]}...")
    print(f"\nLast chunk preview:")
    print(f"  ID   : {records[-1]['id']}")
    print(f"  Text : {records[-1]['chunk_text'][:200]}...")
    print("✅ Ingestion test passed!")