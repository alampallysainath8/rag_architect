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
from dataclasses import dataclass
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
MIN_CHILD_CONTENT_LENGTH: int = 50  # Minimum meaningful content length for a child chunk


# ── Markdown Preprocessing ────────────────────────────────────────────────────

def _normalize_markdown(text: str) -> str:
    """
    Normalize markdown to prevent headers from being isolated during splitting.
    
    Strategy:
      1. Attach headers to the following paragraph by removing empty lines after headers
      2. Preserve double newlines between paragraphs (not after headers)
      3. Keep markdown structure intact
    
    Example:
        Before:
            ## Topic 1
            
            Some text here
            
        After:
            ## Topic 1
            Some text here
            
    This ensures RecursiveCharacterTextSplitter keeps headers with their content.
    """
    lines = text.split('\n')
    normalized = []
    i = 0
    
    while i < len(lines):
        current_line = lines[i]
        
        # Check if current line is a markdown header
        if re.match(r'^#{1,6}\s+', current_line.strip()):
            normalized.append(current_line)
            i += 1
            
            # Skip empty lines immediately after a header
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            # If there's content following the header, keep it attached
            # Don't add newline between header and first content line
            continue
        else:
            normalized.append(current_line)
            i += 1
    
    return '\n'.join(normalized)


# ── Validation Helpers ────────────────────────────────────────────────────────

def _is_valid_child_chunk(text: str, min_length: int = MIN_CHILD_CONTENT_LENGTH) -> bool:
    """
    Check if a chunk has meaningful content (not just headings or whitespace).
    
    Returns False if:
      - Only markdown headings (e.g., "## Introduction")
      - Multiple headers with no content
      - Too short after stripping whitespace
      - Only punctuation/special characters
      - Only page numbers or navigation text
      - Only whitespace and special characters
    """
    stripped = text.strip()
    
    # Skip empty or very short chunks
    if len(stripped) < min_length:
        return False
    
    # Get all lines
    lines = [line.strip() for line in stripped.split('\n') if line.strip()]
    
    if not lines:
        return False
    
    # Skip chunks that are only markdown headings (single or multiple)
    # Check if ALL lines are headers
    if all(re.match(r'^#{1,6}\s+', line) for line in lines):
        return False
    
    # Skip single-line heading-only chunks
    if len(lines) == 1 and re.match(r'^#{1,6}\s+', lines[0]):
        return False
    
    # Skip chunks that are only page numbers or navigation
    if re.fullmatch(r'^(page\s*)?\d+$', stripped, re.IGNORECASE):
        return False
    
    # Filter out heading lines to check actual content
    non_heading_lines = [
        line for line in lines 
        if not re.match(r'^#{1,6}\s+', line)
    ]
    
    # Must have at least some non-heading content
    if not non_heading_lines:
        return False
    
    # Check if remaining content has sufficient length and alphanumeric content
    content_without_headings = ' '.join(non_heading_lines)
    content_clean = content_without_headings.strip()
    
    if len(content_clean) < min_length:
        return False
    
    # Must have at least some alphanumeric characters (not just punctuation)
    if not re.search(r'[a-zA-Z0-9]', content_clean):
        return False
    
    return True


def _is_valid_parent_chunk(text: str, min_length: int = 100) -> bool:
    """
    Validate that a parent chunk has meaningful content.
    Less strict than child validation since parents should be larger.
    """
    stripped = text.strip()
    
    if len(stripped) < min_length:
        return False
    
    # Must have some alphanumeric content
    if not re.search(r'[a-zA-Z0-9]', stripped):
        return False
    
    return True


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
        """Return a record suitable for Pinecone upsert_records().

        ``text`` is consumed by Pinecone's field_map for embedding.
        ``chunk_text`` is the same value stored as a plain metadata field
        so rank_fields and fields queries can reference it by name.
        """
        return {
            "_id":        self.chunk_id,
            "text":       self.text,        # embedding source (field_map)
            "chunk_text": self.text,        # reranking + metadata retrieval
            "source":     self.source,
            "doc_id":     self.doc_id,
            "parent_id":  self.parent_id,
            "level":      "child",
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
    # Pipeline Step 1: Normalize markdown to attach headers to content
    logger.debug(f"Normalizing markdown for '{doc_id}' ({len(text)} chars)")
    normalized_text = _normalize_markdown(text)
    logger.debug(f"Normalized to {len(normalized_text)} chars")
    
    # Pipeline Step 2: Parent chunking with optimal separators
    # Custom separator order prioritizes keeping headers with content:
    # - "\n## " catches markdown headers before they get isolated
    # - "\n\n" preserves paragraph boundaries
    # - Rest follow natural text boundaries
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
    )
    parent_texts = parent_splitter.split_text(normalized_text)

    parents: List[ParentChunk] = []
    parent_idx = 0
    for pt in parent_texts:
        # Validate parent chunks too
        if not _is_valid_parent_chunk(pt):
            logger.debug(f"Skipped invalid parent chunk (too short or meaningless): '{pt[:50]}...'")
            continue
        
        parents.append(
            ParentChunk(
                parent_id=f"{doc_id}::parent_{parent_idx}",
                doc_id=doc_id,
                source=source,
                text=pt,
                index=parent_idx,
            )
        )
        parent_idx += 1

    if not parents:
        logger.warning(f"No valid parent chunks generated for doc '{doc_id}'")
        return [], []

    # Pipeline Step 3: Child chunking with header-aware separators
    # Use same separator strategy to keep headers attached at child level
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
    )

    children: List[ChildChunk] = []
    global_child_idx = 0
    skipped_count = 0
    
    # Pipeline Step 4: Generate and validate child chunks
    for parent in parents:
        child_texts = child_splitter.split_text(parent.text)
        
        for local_idx, ct in enumerate(child_texts):
            # Pipeline Step 5: Filter useless chunks
            if not _is_valid_child_chunk(ct):
                skipped_count += 1
                logger.debug(
                    f"Skipped invalid child chunk (header-only/short): '{ct[:50]}...'"
                )
                continue
            
            children.append(
                ChildChunk(
                    chunk_id=f"{parent.parent_id}::child_{global_child_idx}",
                    parent_id=parent.parent_id,
                    doc_id=doc_id,
                    source=source,
                    text=ct,
                    chunk_index=global_child_idx,
                )
            )
            global_child_idx += 1

    if not children:
        logger.warning(
            f"No valid child chunks generated for doc '{doc_id}'. "
            f"All {len(parents)} parent chunks produced only heading-only or short children."
        )

    # Pipeline complete
    logger.info(
        f"Parent-Child pipeline complete for '{doc_id}':\n"
        f"  → Normalized: {len(text)} → {len(normalized_text)} chars\n"
        f"  → Parents: {len(parents)} (valid)\n"
        f"  → Children: {len(children)} (valid, skipped {skipped_count} invalid)\n"
        f"  → Ready for Pinecone (children) + Redis (parents)"
    )
    return parents, children
