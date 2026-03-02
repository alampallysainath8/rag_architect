"""
Chunking Strategy 1 — Parent-Child (Hierarchical) Chunking
===========================================================

Decouples the unit of retrieval from the unit of context:
  • Small child chunks  → indexed in Pinecone for precise retrieval
  • Large parent chunks → stored in Redis, passed to the LLM for rich context

Flow:
  1. Split document into large parent chunks (default 1 500 chars, 100 overlap).
  2. For every parent, split further into child chunks (default 300 chars, 50 overlap).
  3. Each child carries a `parent_id` in its metadata.
  4. At retrieval time the matching child's `parent_id` is used to fetch the full
     parent text from Redis — the LLM always sees the wider context window.

Metadata on each child record (Pinecone):
  {
    "level":        "child",
    "parent_id":    "<doc_id>::parent_<n>",
    "doc_id":       "<doc_id>",
    "source":       "<filename>",
    "chunk_index":  <int>,
  }

Metadata on each parent record (Redis key: parent:<parent_id>):
  {
    "parent_id":    "<doc_id>::parent_<n>",
    "doc_id":       "<doc_id>",
    "source":       "<filename>",
    "text":         "<full parent text>",
  }
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
_PC_CFG = cfg["chunking"]["parent_child"]
DEFAULT_PARENT_SIZE: int = _PC_CFG["parent_chunk_size"]
DEFAULT_PARENT_OVERLAP: int = _PC_CFG["parent_overlap"]
DEFAULT_CHILD_SIZE: int = _PC_CFG["child_chunk_size"]
DEFAULT_CHILD_OVERLAP: int = _PC_CFG["child_overlap"]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ParentChunk:
    parent_id: str          # e.g. "report.pdf::parent_3"
    doc_id: str             # e.g. "report.pdf"
    source: str             # original filename
    text: str
    index: int              # 0-based position within document

    def to_redis_dict(self) -> dict:
        return {
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "text": self.text,
            "index": str(self.index),
        }


@dataclass
class ChildChunk:
    chunk_id: str           # e.g. "report.pdf::parent_3::child_1"
    parent_id: str
    doc_id: str
    source: str
    text: str
    chunk_index: int        # global child index within document
    chunk_type: str = "text"  # "text" | "table" | "image"

    def to_pinecone_record(self) -> dict:
        """Return a record suitable for Pinecone upsert_records()."""
        return {
            "_id": self.chunk_id,
            "chunk_text": self.text,
            "source": self.source,
            "doc_id": self.doc_id,
            "parent_id": self.parent_id,
            "level": "child",
            "chunk_type": self.chunk_type,
            "chunk_index": str(self.chunk_index),
        }


# ── Core function ─────────────────────────────────────────────────────────────

def parent_child_chunk(
    text: str,
    doc_id: str,
    source: str,
    parent_chunk_size: int = DEFAULT_PARENT_SIZE,
    parent_overlap: int = DEFAULT_PARENT_OVERLAP,
    child_chunk_size: int = DEFAULT_CHILD_SIZE,
    child_overlap: int = DEFAULT_CHILD_OVERLAP,
) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    """
    Split *text* into a two-level parent/child hierarchy.

    Args:
        text:              Full document text (may be Markdown or plain text).
        doc_id:            Unique identifier for this document (e.g. filename).
        source:            Human-readable source label (e.g. filename).
        parent_chunk_size: Max characters per parent chunk.
        parent_overlap:    Overlap between consecutive parent chunks.
        child_chunk_size:  Max characters per child chunk.
        child_overlap:     Overlap between consecutive child chunks.

    Returns:
        Tuple (parent_chunks, child_chunks).
    """
    # Step 1 — parent chunks
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    parent_texts = parent_splitter.split_text(text)

    parents: List[ParentChunk] = []
    for i, pt in enumerate(parent_texts):
        parents.append(
            ParentChunk(
                parent_id=f"{doc_id}::parent_{i}",
                doc_id=doc_id,
                source=source,
                text=pt,
                index=i,
            )
        )

    # Step 2 — child chunks per parent
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    children: List[ChildChunk] = []
    global_child_idx = 0
    for parent in parents:
        child_texts = child_splitter.split_text(parent.text)
        for local_idx, ct in enumerate(child_texts):
            children.append(
                ChildChunk(
                    chunk_id=f"{parent.parent_id}::child_{local_idx}",
                    parent_id=parent.parent_id,
                    doc_id=doc_id,
                    source=source,
                    text=ct,
                    chunk_index=global_child_idx,
                )
            )
            global_child_idx += 1

    logger.info(
        f"Parent-Child chunking: doc='{doc_id}' → "
        f"{len(parents)} parents, {len(children)} children"
    )
    return parents, children
