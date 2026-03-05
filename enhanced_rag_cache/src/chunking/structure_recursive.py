"""
Structure-Aware Recursive Chunker
==================================
Pipeline: PDF → Markdown → extract tables/images → split by headings → chunks.

Each chunk has:
  chunk_type   : "text" | "table" | "image"
  heading_meta : h1–h4 hierarchy at the chunk’s location
  metadata     : char_count + table_rows/cols or image_number
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
_SR_CFG = cfg["chunking"]["structure_recursive"]
DEFAULT_MAX_SECTION_SIZE: int = _SR_CFG["max_section_size"]
DEFAULT_RECURSIVE_SIZE: int   = _SR_CFG["recursive_chunk_size"]
DEFAULT_RECURSIVE_OVERLAP: int = _SR_CFG["recursive_overlap"]
_RAW_HEADERS: list = _SR_CFG["headers_to_split"]
DEFAULT_HEADERS_TO_SPLIT: list[tuple[str, str]] = [(h[0], h[1]) for h in _RAW_HEADERS]

# ── Compiled regexes ───────────────────────────────────────────────────────────
_PAGE_NUM_RE          = re.compile(r"^#{1,6}\s*[*_\s]*\d{1,4}[*_\s]*$")
_TABLE_LINE_RE        = re.compile(r"^\|.+\|")
_PLAIN_TBL_CAPTION_RE = re.compile(r"^Table\s+\d+\s*:", re.M)
_PLACEHOLDER_RE       = re.compile(r"(__(?:IMG_BLOCK|TABLE)_\d+__)")


def _is_plain_table_row(line: str) -> bool:
    """
    Return True when a line looks like a row inside a borderless PDF table.
    Covers: numeric data rows, 3+ **bold** column headers, italic/bold-italic
    section labels, and short all-abbreviation header rows.
    """
    s = line.strip()
    if not s or s.startswith("#") or s == "-----":
        return False
    if re.search(r"\b\d[\d,.]*\b", s):                          # numbers → data row
        return True
    if len(re.findall(r"\*\*[^*\n]+\*\*", s)) >= 3:             # 3+ bold → header row
        return True
    if re.fullmatch(r"_[A-Za-z][^_\n]+_", s):                  # _italic group_
        return True
    if re.fullmatch(r"\*\*_[^*\n]+_\*\*", s):                   # **_bold-italic_**
        return True
    tokens = s.split()
    if len(tokens) >= 4 and all(len(t) <= 6 for t in tokens):  # abbrev header row
        return True
    return False


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class StructChunk:
    chunk_id:     str
    doc_id:       str
    source:       str
    text:         str
    chunk_type:   str                   # "text" | "table" | "image"
    heading_meta: dict[str, str]        # e.g. {"h3": "Introduction"}
    chunk_index:  int
    sub_index:    int = 0
    metadata:     dict[str, Any] = field(default_factory=dict)

    def to_pinecone_record(self) -> dict:
        record: dict[str, Any] = {
            "_id":          self.chunk_id,
            "text":         self.text,
            "source":       self.source,
            "doc_id":       self.doc_id,
            "strategy":     "structure_recursive",
            "chunk_type":   self.chunk_type,
            "chunk_index":  str(self.chunk_index),
            "sub_index":    str(self.sub_index),
        }
        record.update(self.heading_meta)
        record.update(self.metadata)
        return record


# ── Internal helpers ───────────────────────────────────────────────────────────

def _strip_page_numbers(md: str) -> str:
    """Remove heading lines that are pure page numbers (e.g. ### 3)."""
    return "\n".join(ln for ln in md.splitlines() if not _PAGE_NUM_RE.match(ln.strip()))


def _clean_heading_meta(meta: dict) -> dict:
    """
    Normalise MarkdownHeaderTextSplitter metadata keys ("Header 1" → "h1")
    and drop values that look like bare page numbers.
    """
    out: dict[str, str] = {}
    for k, v in meta.items():
        norm = k
        m = re.match(r"Header\s+(\d)", k, re.I)
        if m:
            norm = f"h{m.group(1)}"
        if re.fullmatch(r"[\s*_]*\d{1,4}[\s*_]*", str(v)):
            continue
        out[norm] = v
    return out


# ── Special-block extraction ───────────────────────────────────────────────────

def extract_special_blocks(
    markdown: str,
    image_blocks: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Extract pipe tables and plain-text tables from *markdown*, replacing each
    with a placeholder token.

    Images are handled upstream by enrich_markdown_images() which has already
    replaced base64 blobs with ``__IMG_BLOCK_N__`` tokens.  Pass the returned
    image_blocks list here so they are included in the placeholder map.

    Returns: (cleaned_markdown, all_block_dicts)
    """
    blocks: list[dict] = list(image_blocks) if image_blocks else []
    cleaned = markdown

    # 1. Markdown pipe tables — walk line-by-line, pull up preceding headings
    lines     = cleaned.splitlines()
    new_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _TABLE_LINE_RE.match(line.strip()):
            # Grab up to 3 heading lines that immediately precede the table
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
            # Remove those header lines from output (they travel with the table)
            for _ in h_lines:
                if new_lines:
                    new_lines.pop()
                if new_lines and not new_lines[-1].strip():
                    new_lines.pop()
            # Collect all consecutive table lines
            tbl: list[str] = []
            while i < len(lines) and _TABLE_LINE_RE.match(lines[i].strip()):
                tbl.append(lines[i])
                i += 1
            full_table = "\n".join(h_lines + ([""] if h_lines else []) + tbl).strip()
            token = f"__TABLE_{len(blocks)}__"
            col_count = len([c for c in tbl[0].split("|") if c.strip()]) if tbl else 0
            row_count = len([l for l in tbl if not re.match(r"^\|[\s\-:]+\|", l)])
            blocks.append({
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

    # 2. Plain-text tables — borderless PDF tables ("Table N:" caption + heuristic rows)
    lines2: list[str] = cleaned.splitlines()
    new_lines2: list[str] = []
    j = 0
    while j < len(lines2):
        ln = lines2[j]
        if _PLAIN_TBL_CAPTION_RE.match(ln.strip()):
            # Scan ahead (up to 15 lines) past multi-line captions for first table row
            look, found = j + 1, False
            while look < len(lines2) and look < j + 15:
                if _is_plain_table_row(lines2[look]):
                    found = True
                    break
                if lines2[look].strip().startswith("#"):
                    break
                look += 1

            if found:
                # Collect caption lines until first blank or immediate table row
                caption: list[str] = [ln]
                j += 1
                while j < len(lines2):
                    cur = lines2[j]
                    if not cur.strip():          # blank → end of caption
                        j += 1; break
                    if _is_plain_table_row(cur): # immediate row → end of caption
                        break
                    caption.append(cur)
                    j += 1

                # Collect body rows; tolerate up to 3 consecutive blank lines
                # (large tables span multiple sub-sections separated by blanks)
                body: list[str] = []
                blanks = 0
                while j < len(lines2):
                    cur = lines2[j]
                    if not cur.strip():
                        blanks += 1
                        if blanks > 3:
                            break
                        # Keep blank only if a table row follows
                        nxt = j + 1
                        while nxt < len(lines2) and not lines2[nxt].strip():
                            nxt += 1
                        if nxt < len(lines2) and _is_plain_table_row(lines2[nxt]):
                            body.append(cur); j += 1
                        else:
                            break
                    elif cur.strip().startswith("#"):
                        break
                    elif _is_plain_table_row(cur):
                        blanks = 0
                        body.append(cur); j += 1
                    else:
                        # Single non-table line tolerated if sandwiched by table rows
                        nxt = j + 1
                        while nxt < len(lines2) and not lines2[nxt].strip():
                            nxt += 1
                        if nxt < len(lines2) and _is_plain_table_row(lines2[nxt]):
                            body.append(cur); j += 1
                        else:
                            break

                if body:
                    token = f"__TABLE_{len(blocks)}__"
                    full  = "\n".join(caption + [""] + body).strip()
                    data_rows = [l for l in body if re.search(r"\b\d[\d,.]*\b", l)]
                    hdr = next((l for l in body if "**" in l), "")
                    cols = max(1, len(re.findall(r"\*\*[^*]+\*\*", hdr)) or
                               len((body[0] if body else "").split()))
                    blocks.append({"type": "table", "placeholder": token,
                                   "content": full, "table_columns": cols,
                                   "table_rows": len(data_rows)})
                    new_lines2.append(token)
                    continue
                else:
                    new_lines2.extend(caption)
                    continue

        new_lines2.append(ln)
        j += 1

    return "\n".join(new_lines2), blocks


# ── Main chunker ───────────────────────────────────────────────────────────────

def structure_recursive_chunk(
    markdown_text: str,
    doc_id: str,
    source: str,
    max_section_size: int = DEFAULT_MAX_SECTION_SIZE,
    recursive_chunk_size: int = DEFAULT_RECURSIVE_SIZE,
    recursive_overlap: int = DEFAULT_RECURSIVE_OVERLAP,
    headers_to_split: list[tuple[str, str]] = DEFAULT_HEADERS_TO_SPLIT,
    image_blocks: list[dict] | None = None,
) -> List[StructChunk]:
    """
    Unified structure-recursive + table/image-aware chunker.

    Args:
        markdown_text:        Markdown with __IMG_BLOCK_N__ tokens already in
                              place (output of enrich_markdown_images).
        doc_id:               Unique document identifier.
        source:               Human-readable source label (filename).
        image_blocks:         Block dicts from enrich_markdown_images — merged
                              into the placeholder map alongside table blocks.

    Returns:
        Ordered list of StructChunk objects ready for Pinecone upsert.
    """
    md = _strip_page_numbers(markdown_text)
    md, special_blocks = extract_special_blocks(md, image_blocks=image_blocks)
    placeholder_map = {b["placeholder"]: b for b in special_blocks}

    # Step 2: split by Markdown headings
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False,
    )
    try:
        header_docs = header_splitter.split_text(md)
    except Exception as exc:
        logger.warning(f"Header splitting failed for '{doc_id}' ({exc}) — using full text.")
        header_docs = []

    if not header_docs:
        header_docs = [type("_D", (), {"page_content": md, "metadata": {}})()]

    # Step 3: recursive splitter (used only when a text part is too large)
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=recursive_chunk_size,
        chunk_overlap=recursive_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[StructChunk] = []
    chunk_index = 0

    for doc in header_docs:
        section_text: str = (
            doc.page_content.strip() if hasattr(doc, "page_content") else str(doc).strip()
        )
        heading_meta: dict[str, str] = _clean_heading_meta(
            getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
        )
        if not section_text:
            continue

        # Step 4: within the section, split on placeholder tokens so that every
        # table / image block becomes a dedicated chunk.
        parts = _PLACEHOLDER_RE.split(section_text)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # ── Special block (table or image) ──────────────────────────
            if part in placeholder_map:
                block   = placeholder_map[part]
                btype   = block["type"]
                content = block["content"]
                meta: dict[str, Any] = {"char_count": len(content)}
                if btype == "table":
                    meta["table_rows"]    = block["table_rows"]
                    meta["table_columns"] = block["table_columns"]
                elif btype == "image" and block.get("image_number") is not None:
                    meta["image_number"] = block["image_number"]
                chunks.append(StructChunk(
                    chunk_id    = f"{doc_id}::struct_{chunk_index}",
                    doc_id      = doc_id,
                    source      = source,
                    text        = content,
                    chunk_type  = btype,
                    heading_meta= heading_meta,
                    chunk_index = chunk_index,
                    sub_index   = 0,
                    metadata    = meta,
                ))
                chunk_index += 1
                continue

            # ── Text — skip if it's just numbers/whitespace ──────────────
            if re.fullmatch(r"[\s\d*_#\-]+", part):
                continue

            # ── Text — recursively split if oversized ────────────────────
            sub_texts: list[str] = (
                rec_splitter.split_text(part) if len(part) > max_section_size else [part]
            )
            for sub_idx, sub in enumerate(sub_texts):
                sub = sub.strip()
                if not sub or re.fullmatch(r"[\s\d*_#\-]+", sub):
                    continue
                text_meta: dict[str, Any] = {"char_count": len(sub)}
                primary = re.findall(r"^#{1,6}\s+(.+)$", sub, re.MULTILINE)
                if primary:
                    text_meta["primary_header"] = primary[0]
                chunks.append(StructChunk(
                    chunk_id    = (
                        f"{doc_id}::struct_{chunk_index}_sub_{sub_idx}"
                        if sub_idx else f"{doc_id}::struct_{chunk_index}"
                    ),
                    doc_id      = doc_id,
                    source      = source,
                    text        = sub,
                    chunk_type  = "text",
                    heading_meta= heading_meta,
                    chunk_index = chunk_index,
                    sub_index   = sub_idx,
                    metadata    = text_meta,
                ))
                chunk_index += 1

    logger.info(
        f"structure_recursive_chunk: doc='{doc_id}' → {len(chunks)} chunks "
        f"(text={sum(1 for c in chunks if c.chunk_type=='text')}, "
        f"table={sum(1 for c in chunks if c.chunk_type=='table')}, "
        f"image={sum(1 for c in chunks if c.chunk_type=='image')})"
    )
    return chunks
