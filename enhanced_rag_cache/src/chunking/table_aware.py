"""
Table-Aware Markdown Chunker
============================

Chunks Markdown while keeping two special block types intact:
  • Tables         — the full table (with any directly-preceding headers) is
                     emitted as a single chunk, never split mid-row.
  • Image blocks   — description / extracted-table / JSON triples produced by
                     image_enricher are each kept as one atomic chunk.

Regular text is split using MarkdownHeaderTextSplitter (respects H1-H4
boundaries) with a RecursiveCharacterTextSplitter fallback for oversized
sections.

Page-number-only lines (e.g. '#### **3**' or '#### 12') are filtered out
before splitting and from chunk metadata.

Pinecone record fields per chunk:
  {
    "_id":         "<doc_id>::tbl_<n>  |  img_<n>  |  txt_<n>",
    "chunk_text":  "<content>",
    "source":      "<filename>",
    "doc_id":      "<doc_id>",
    "chunk_type":  "table" | "image" | "text",
    "h1".."h4":    "<heading>" (text chunks only, when present),
    "chunk_index": "<int>",
    # tables also carry:
    "table_rows":   "<int>",
    "table_columns": "<int>",
    # image chunks also carry:
    "image_number": "<int>",
  }
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

_PAGE_NUMBER_RE = re.compile(r"^#{1,6}\s*[*_\s]*\d{1,4}[*_\s]*$")
_TABLE_LINE_RE = re.compile(r"^\|.+\|")
_IMAGE_BLOCK_RE = re.compile(
    r"\*\*\[Image \d+ Description\]:\*\*.*?(?=\*\*\[Image \d+|$)",
    re.S,
)


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class TableAwareChunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    chunk_type: str          # "table" | "image" | "text"
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_pinecone_record(self) -> dict:
        record: dict[str, Any] = {
            "_id": self.chunk_id,
            "chunk_text": self.text,
            "source": self.source,
            "doc_id": self.doc_id,
            "chunk_type": self.chunk_type,
            "chunk_index": str(self.chunk_index),
            "strategy": "table_aware",
        }
        record.update(self.metadata)
        return record


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_page_number_line(line: str) -> bool:
    return bool(_PAGE_NUMBER_RE.match(line.strip()))


def _strip_page_number_lines(md: str) -> str:
    """Remove heading lines that are pure page numbers before splitting."""
    return "\n".join(
        ln for ln in md.splitlines() if not _is_page_number_line(ln)
    )


def _clean_heading_meta(meta: dict) -> dict:
    """
    Keep only hN keys that are meaningful (not numeric / page-number values).
    Also drop 'Header N' style keys from MarkdownHeaderTextSplitter output.
    """
    out = {}
    for k, v in meta.items():
        # MarkdownHeaderTextSplitter uses "Header 1", "Header 2", … — normalise
        norm_k = k
        hm = re.match(r"Header\s+(\d)", k, re.I)
        if hm:
            norm_k = f"h{hm.group(1)}"

        # skip if value looks like a page number or very short number
        if re.fullmatch(r"[\s\*_]*\d{1,4}[\s\*_]*", str(v)):
            continue
        out[norm_k] = v
    return out


# ── Special-block extraction ──────────────────────────────────────────────────

def extract_special_blocks(
    markdown: str,
) -> tuple[str, list[dict]]:
    """
    Pull tables and image-description blocks out of *markdown*, replace each
    with a short placeholder token, and return:
      (cleaned_markdown, [special_block_dict, …])

    Each special_block_dict:
      { "type": "table"|"image", "placeholder": str, "content": str,
        **type-specific keys }
    """
    special_blocks: list[dict] = []
    cleaned = markdown

    # ── 0. Raw base64 images (not yet enriched by image_enricher) ────────
    # This handles the case where GROQ_API_KEY is absent: base64 blobs are
    # still in the Markdown.  We pull them out so they don't corrupt chunking.
    _RAW_IMG_RE = re.compile(
        r"!\[\]\(data:image/png;base64,[A-Za-z0-9+/=\n\r]+\)"
    )
    for raw_m in list(_RAW_IMG_RE.finditer(cleaned)):
        img_n = len(special_blocks) + 1
        token = f"__IMG_BLOCK_{len(special_blocks)}__"
        special_blocks.append({
            "type": "image",
            "placeholder": token,
            "content": f"[Image {img_n}: embedded image — no description available (vision model not configured)]",
            "image_number": img_n,
        })
        cleaned = cleaned.replace(raw_m.group(0), f"\n\n{token}\n\n", 1)
        logger.debug(f"extract_special_blocks: raw base64 image {img_n} → {token}")

    # ── 1. Image blocks (produced by image_enricher) ──────────────────────
    # Pattern: **[Image N Description]:** … up to next image block or table or heading
    img_pattern = re.compile(
        r"(\*\*\[Image \d+ Description\]:\*\*"
        r".*?)"
        r"(?=\n\*\*\[Image \d+|\n#{1,6} |\n\||\Z)",
        re.S,
    )
    for m in img_pattern.finditer(cleaned):
        content = m.group(0).strip()
        token = f"__IMG_BLOCK_{len(special_blocks)}__"
        img_num_m = re.search(r"\[Image (\d+)", content)
        special_blocks.append({
            "type": "image",
            "placeholder": token,
            "content": content,
            "image_number": int(img_num_m.group(1)) if img_num_m else None,
        })
        cleaned = cleaned.replace(m.group(0), f"\n\n{token}\n\n", 1)

    # ── 2. Tables ─────────────────────────────────────────────────────────
    # Walk line-by-line; when a table starts, look back for headers to attach.
    lines = cleaned.splitlines()
    new_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _TABLE_LINE_RE.match(line.strip()):
            # Capture headers immediately before the table (up to 3 heading lines)
            h_lines: list[str] = []
            back = len(new_lines) - 1
            while back >= 0 and len(h_lines) < 3:
                prev = new_lines[back].strip()
                if prev.startswith("#"):
                    h_lines.insert(0, new_lines[back])
                    back -= 1
                elif not prev:
                    back -= 1
                else:
                    break
            # Remove those header lines from new_lines (they travel with the table)
            for _ in h_lines:
                if new_lines:
                    new_lines.pop()
                if new_lines and not new_lines[-1].strip():
                    new_lines.pop()

            # Collect all consecutive table lines
            tbl_lines: list[str] = []
            while i < len(lines) and _TABLE_LINE_RE.match(lines[i].strip()):
                tbl_lines.append(lines[i])
                i += 1

            full_table = "\n".join(
                h_lines + ([""] if h_lines else []) + tbl_lines
            ).strip()
            token = f"__TABLE_{len(special_blocks)}__"

            # Count cols/rows (first row / exclude separator row)
            col_count = len([c for c in tbl_lines[0].split("|") if c.strip()]) if tbl_lines else 0
            row_count = len([l for l in tbl_lines if not re.match(r"^\|[\s\-:]+\|", l)])

            special_blocks.append({
                "type": "table",
                "placeholder": token,
                "content": full_table,
                "table_columns": col_count,
                "table_rows": row_count,
            })
            new_lines.append(token)
            continue

        new_lines.append(line)
        i += 1

    cleaned = "\n".join(new_lines)
    return cleaned, special_blocks


# ── Main chunker ──────────────────────────────────────────────────────────────

def table_aware_chunk(
    markdown_text: str,
    doc_id: str,
    source: str,
    max_section_size: int = 1500,
    recursive_chunk_size: int = 1000,
    recursive_overlap: int = 150,
) -> List[TableAwareChunk]:
    """
    Chunk *markdown_text* into TableAwareChunk objects.

    Args:
        markdown_text:        Enriched Markdown (from pdf_to_markdown + image_enricher).
        doc_id:               Unique document identifier.
        source:               Human-readable source label.
        max_section_size:     Max chars per text section before recursive splitting.
        recursive_chunk_size: Target size for sub-splits.
        recursive_overlap:    Overlap for sub-splits.

    Returns:
        Ordered list of TableAwareChunk objects.
    """
    # ── Pre-process ───────────────────────────────────────────────────────
    clean_md = _strip_page_number_lines(markdown_text)
    clean_md, special_blocks = extract_special_blocks(clean_md)
    placeholder_map = {b["placeholder"]: b for b in special_blocks}

    # ── Header split on remaining text ───────────────────────────────────
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    try:
        splits = header_splitter.split_text(clean_md)
    except Exception as e:
        logger.warning(f"Header splitting failed ({e}), using full text as one section.")
        splits = [type("_Doc", (), {"page_content": clean_md, "metadata": {}})()]

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=recursive_chunk_size,
        chunk_overlap=recursive_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: List[TableAwareChunk] = []
    global_idx = 0

    for split in splits:
        if hasattr(split, "page_content"):
            content = split.page_content
            split_meta = _clean_heading_meta(getattr(split, "metadata", {}))
        elif isinstance(split, dict):
            content = split.get("content", split.get("page_content", ""))
            split_meta = _clean_heading_meta(split.get("metadata", {}))
        else:
            content = str(split)
            split_meta = {}

        # Remove page number lines that slipped through
        content_lines = [
            ln for ln in content.splitlines() if not _is_page_number_line(ln)
        ]
        content = "\n".join(content_lines).strip()
        if not content:
            continue

        # ── Check if this split is (or contains) a special block ──────────
        matched_special = False
        for token, block in placeholder_map.items():
            if token in content:
                # Replace token with actual content for output
                actual = block["content"]
                btype = block["type"]
                meta: dict[str, Any] = {"char_count": len(actual)}
                if btype == "table":
                    meta["table_rows"] = block["table_rows"]
                    meta["table_columns"] = block["table_columns"]
                elif btype == "image":
                    if block["image_number"] is not None:
                        meta["image_number"] = block["image_number"]

                chunks.append(TableAwareChunk(
                    chunk_id=f"{doc_id}::{btype}_{global_idx}",
                    doc_id=doc_id,
                    source=source,
                    text=actual,
                    chunk_type=btype,
                    chunk_index=global_idx,
                    metadata=meta,
                ))
                global_idx += 1
                matched_special = True
                # Remove token from content; if anything remains process as text
                content = content.replace(token, "").strip()

        if not content:
            continue

        # ── Regular text chunk ────────────────────────────────────────────
        # Skip if the remaining content is only page numbers / short numbers
        if re.fullmatch(r"[\s\d*_#]+", content):
            continue

        if len(content) > max_section_size:
            sub_texts = recursive_splitter.split_text(content)
        else:
            sub_texts = [content]

        for sub_i, sub in enumerate(sub_texts):
            sub = sub.strip()
            if not sub or re.fullmatch(r"[\s\d*_#]+", sub):
                continue

            headers_found = re.findall(r"^#{1,6}\s+(.+)$", sub, re.MULTILINE)
            text_meta: dict[str, Any] = {
                "char_count": len(sub),
                **split_meta,
            }
            if headers_found:
                text_meta["primary_header"] = headers_found[0]

            chunks.append(TableAwareChunk(
                chunk_id=f"{doc_id}::txt_{global_idx}",
                doc_id=doc_id,
                source=source,
                text=sub,
                chunk_type="text",
                chunk_index=global_idx,
                metadata=text_meta,
            ))
            global_idx += 1

    logger.info(
        f"table_aware_chunk: doc='{doc_id}' → "
        f"{len(chunks)} chunks  "
        f"(text={sum(1 for c in chunks if c.chunk_type=='text')}, "
        f"table={sum(1 for c in chunks if c.chunk_type=='table')}, "
        f"image={sum(1 for c in chunks if c.chunk_type=='image')})"
    )
    return chunks
