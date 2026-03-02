"""
Ingestion Pipeline — document loading, chunking, and Pinecone indexing.

Two strategies:
  • "structure_recursive" — PDF→Markdown (table_strategy="lines") → header
    splitting + recursive fallback.  Bordered tables are detected by pymupdf4llm;
    borderless plain-text tables are detected in the chunker via caption heuristic.
  • "parent_child"        — hierarchical parent/child chunking with
    table/image block preservation via placeholder tokens.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Literal

from src.caching.parent_cache import store_parents
from src.chunking.parent_child import parent_child_chunk, ParentChunk, ChildChunk
from src.chunking.structure_recursive import structure_recursive_chunk, StructChunk, extract_special_blocks
from src.retrieval.pinecone_manager import upsert_records
from src.utils.pdf_to_markdown import file_to_text
from src.utils.image_enricher import enrich_markdown_images
from src.utils.logger import get_logger
from src.utils.exceptions import IngestionException

logger = get_logger(__name__)

ChunkStrategy = Literal["parent_child", "structure_recursive"]


def ingest_document(
    file_path: str,
    strategy: ChunkStrategy = "parent_child",
    namespace: str | None = None,
) -> Dict[str, Any]:
    """
    Full ingestion pipeline for a single document.

    PDF files are automatically enriched:
      - Converted to Markdown with base64-embedded images.
      - Images described by the Groq vision model (description + table + JSON).
      - Tables and image blocks protected from splitting via placeholders.

    Args:
        file_path: Absolute path to a .pdf, .txt, or .md file.
        strategy:  Chunking strategy —
                     "parent_child"        hierarchical parent-child
                     "structure_recursive" header + recursive splitting
        namespace: Pinecone namespace for upsert (None → use config default).

    Returns:
        Summary dict with doc_id, source, strategy, counts, breakdown, etc.

    Raises:
        FileNotFoundError:  if the file does not exist.
        ValueError:         if an unknown strategy is supplied.
        IngestionException: for any pipeline error during chunking/upsert.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if strategy is None:
        strategy = "parent_child"

    doc_id = path.name
    source = path.name
    is_pdf = path.suffix.lower() == ".pdf"

    logger.info(f"Starting ingestion: doc='{doc_id}', strategy='{strategy}'")

    kwargs: Dict[str, Any] = {}
    if namespace:
        kwargs["namespace"] = namespace

    try:
        # ── Step 1: Load & enrich ─────────────────────────────────────────
        image_blocks: list[dict] = []
        if is_pdf:
            logger.info(f"[PDF] Step 1/3 — Converting with embedded images: {doc_id}")
            raw_md = file_to_text(str(path), embed_images=True)
            logger.info(f"[PDF] Raw markdown: {len(raw_md):,} chars")

            logger.info(f"[PDF] Step 2/3 — Image enrichment via Groq vision")
            # Single pass: enrich images AND emit __IMG_BLOCK_N__ placeholders.
            # Returns (cleaned_md_with_tokens, image_block_dicts) so the
            # chunker never needs to re-scan for enriched image text.
            enriched_md, image_blocks = enrich_markdown_images(raw_md)
            logger.info(f"[PDF] Enriched markdown: {len(enriched_md):,} chars, {len(image_blocks)} image(s)")
        else:
            enriched_md = file_to_text(str(path))
            logger.info(f"Loaded '{doc_id}': {len(enriched_md):,} chars")

        # ── Step 3: Chunk + upsert ─────────────────────────────────────────
        if strategy == "structure_recursive":
            return _ingest_structure_recursive(
                enriched_md, doc_id, source, image_blocks=image_blocks, **kwargs
            )

        # For parent_child: extract placeholders first, then restore after chunking
        logger.info(f"Extracting special blocks (tables, images) for '{doc_id}'")
        cleaned_md, special_blocks = extract_special_blocks(
            enriched_md, image_blocks=image_blocks or None
        )
        placeholder_map = {b["placeholder"]: b for b in special_blocks}
        logger.info(
            f"Special blocks extracted: {len(special_blocks)} "
            f"({sum(1 for b in special_blocks if b['type']=='table')} tables, "
            f"{sum(1 for b in special_blocks if b['type']=='image')} images)"
        )

        if strategy == "parent_child":
            return _ingest_parent_child(
                cleaned_md, doc_id, source, placeholder_map, **kwargs
            )
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Choose 'parent_child' or 'structure_recursive'."
            )

    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        logger.error(f"Ingestion failed for '{doc_id}': {exc}")
        raise IngestionException(
            f"Ingestion failed for '{doc_id}' using strategy '{strategy}': {exc}",
            exc,
        ) from exc


# ── Placeholder helpers ───────────────────────────────────────────────────────

def _restore_text(text: str, placeholder_map: dict) -> tuple[str, str]:
    """
    If *text* contains a placeholder token, replace it with the original block
    content and return (restored_text, block_type).
    If no placeholder is found, return (text, 'text') unchanged.
    """
    for token, block in placeholder_map.items():
        if token in text:
            restored = text.replace(token, block["content"]).strip()
            return restored, block["type"]
    return text, "text"


def _restore_chunks(chunks: list, placeholder_map: dict) -> None:
    """
    Mutate each chunk's `text` attribute in-place:
      • Replace any placeholder token with the original block content.
      • Set `chunk_type` to "table" or "image" when the chunk is a special block.
    """
    for chunk in chunks:
        restored, btype = _restore_text(chunk.text, placeholder_map)
        chunk.text = restored
        chunk.chunk_type = btype


# ── Private strategy helpers ──────────────────────────────────────────────────

def _ingest_parent_child(
    cleaned_md: str,
    doc_id: str,
    source: str,
    placeholder_map: dict,
    namespace: str | None = None,
) -> Dict[str, Any]:
    """
    Parent-Child chunking with table / image block preservation.

    The cleaned Markdown (placeholders replacing special blocks) is split into
    parent/child pairs.  Placeholders are then restored in every chunk and
    chunk_type is tagged accordingly.
    """
    parents, children = parent_child_chunk(cleaned_md, doc_id=doc_id, source=source)

    # Restore special blocks in parent texts (stored in Redis for context retrieval)
    for parent in parents:
        parent.text, _ = _restore_text(parent.text, placeholder_map)

    # Restore special blocks in child texts and tag types
    _restore_chunks(children, placeholder_map)

    parent_dicts = [p.to_redis_dict() for p in parents]
    store_parents(parent_dicts)
    logger.info(f"Stored {len(parents)} parent chunks in Redis for '{doc_id}'")

    records = [c.to_pinecone_record() for c in children]
    upserted = upsert_records(
        records, **({} if namespace is None else {"namespace": namespace})
    )

    n_text  = sum(1 for c in children if c.chunk_type == "text")
    n_table = sum(1 for c in children if c.chunk_type == "table")
    n_image = sum(1 for c in children if c.chunk_type == "image")

    logger.info(
        f"Ingestion complete (parent_child): doc='{doc_id}' "
        f"parents={len(parents)}, children={len(children)} "
        f"(text={n_text}, table={n_table}, image={n_image}), upserted={upserted}"
    )
    return {
        "doc_id": doc_id,
        "source": source,
        "strategy": "parent_child",
        "parent_count": len(parents),
        "chunk_count": len(children),
        "upserted": upserted,
        "breakdown": {"text": n_text, "table": n_table, "image": n_image},
    }


def _ingest_structure_recursive(
    enriched_md: str,
    doc_id: str,
    source: str,
    image_blocks: list[dict] | None = None,
    namespace: str | None = None,
) -> Dict[str, Any]:
    """
    Structure-Recursive chunking with integrated table/image handling.

    Args:
        enriched_md:   Markdown with __IMG_BLOCK_N__ tokens (from enrich_markdown_images).
        image_blocks:  Block dicts from enrich_markdown_images — merged into the
                       placeholder map alongside table blocks.
    """
    chunks = structure_recursive_chunk(
        enriched_md,
        doc_id=doc_id,
        source=source,
        image_blocks=image_blocks or None,
    )

    records = [c.to_pinecone_record() for c in chunks]
    upserted = upsert_records(
        records, **({} if namespace is None else {"namespace": namespace})
    )

    n_text  = sum(1 for c in chunks if c.chunk_type == "text")
    n_table = sum(1 for c in chunks if c.chunk_type == "table")
    n_image = sum(1 for c in chunks if c.chunk_type == "image")

    logger.info(
        f"Ingestion complete (structure_recursive): doc='{doc_id}' "
        f"chunks={len(chunks)} "
        f"(text={n_text}, table={n_table}, image={n_image}), upserted={upserted}"
    )
    return {
        "doc_id": doc_id,
        "source": source,
        "strategy": "structure_recursive",
        "parent_count": 0,
        "chunk_count": len(chunks),
        "upserted": upserted,
        "breakdown": {"text": n_text, "table": n_table, "image": n_image},
    }

