"""
Parent-Aware Retrieval Processor
=================================

Merges child chunks that share the same parent before context is sent to the
LLM, eliminating two inefficiencies:

  1. Duplicate Redis lookups  -- N children with the same parent_id triggered
                                 N Redis GETs. Now: 1 GET per unique parent.
  2. Duplicate LLM context    -- the same parent text appeared multiple times
                                 in the prompt. Now: one entry per parent.

Pipeline position
-----------------
  Child hits (Pinecone)
    -> group_children_by_parent()    deduplicate parent IDs, preserve order
    -> fetch_parent_chunks()         one Redis GET per unique parent
    -> merge_children_to_parents()   assemble final context list for the LLM

Public API
----------
  group_children_by_parent(children)               -> List[str]
  fetch_parent_chunks(parent_ids, parent_cache)     -> List[Dict]
  merge_children_to_parents(children, parent_cache) -> List[Dict]
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.caching.parent_cache import get_parent
from src.utils.logger import get_logger

logger = get_logger(__name__)

ChunkHit = Dict[str, Any]


def group_children_by_parent(children: List[ChunkHit]) -> List[str]:
    """
    Extract unique parent IDs from a list of child hit dicts.

    Preserves first-occurrence order so the highest-ranked parent appears
    first. Children with no parent_id (other strategies) are skipped.

    Args:
        children: Hit dicts from Pinecone or the Tier-3 retrieval cache.

    Returns:
        Ordered list of unique, non-empty parent ID strings.

    Example::

        children = [
            {"parent_id": "doc::parent_0", ...},
            {"parent_id": "doc::parent_0", ...},  # duplicate
            {"parent_id": "doc::parent_1", ...},
        ]
        group_children_by_parent(children)
        # -> ["doc::parent_0", "doc::parent_1"]
    """
    seen: set[str] = set()
    unique: List[str] = []
    for child in children:
        pid: str = child.get("parent_id", "") or ""
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(pid)
    logger.debug(
        "group_children_by_parent: %d children -> %d unique parent IDs",
        len(children), len(unique),
    )
    return unique


def fetch_parent_chunks(
    parent_ids: List[str],
    parent_cache: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve parent chunk dicts from Redis for the given deduplicated IDs.

    Each ID is fetched exactly once. Missing entries (cache miss / expired TTL)
    are skipped with a warning so the pipeline degrades gracefully.

    Args:
        parent_ids:   Ordered list of unique parent ID strings.
        parent_cache: Optional callable override for Redis lookup (testing).

    Returns:
        Ordered list of parent chunk dicts::

            {"parent_id": str, "text": str, "source": str, "doc_id": str}
    """
    lookup: Callable[[str], Optional[Dict[str, Any]]] = (
        parent_cache if callable(parent_cache) else get_parent
    )
    results: List[Dict[str, Any]] = []
    for pid in parent_ids:
        parent = lookup(pid)
        if parent is None:
            logger.warning(
                "fetch_parent_chunks: parent '%s' not found in Redis -- skipping.", pid
            )
            continue
        results.append(parent)
    logger.debug(
        "fetch_parent_chunks: requested %d, retrieved %d parent chunks.",
        len(parent_ids), len(results),
    )
    return results


def merge_children_to_parents(
    children: List[ChunkHit],
    parent_cache: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> List[ChunkHit]:
    """
    Convert retrieved child hits into a deduplicated parent context list.

    Steps
    -----
    1. Partition hits into parent-child hits (have parent_id) and plain
       hits (no parent_id -- other strategies).
    2. Deduplicate parent IDs via group_children_by_parent().
    3. Batch-fetch from Redis via fetch_parent_chunks()
       -- one GET per unique parent.
    4. Build one context hit per fetched parent, carrying the metadata of the
       highest-ranked child that pointed to it (score, source, headings).
    5. Plain hits pass through unchanged.

    Args:
        children:     Hit dicts from Pinecone retrieval or Tier-3 cache.
        parent_cache: Optional callable override for Redis lookup (testing).

    Returns:
        Context hit dicts compatible with generate_answer / format_sources:

        - context_text    -- full parent text (or child text for plain hits)
        - chunk_text      -- same as context_text for consistency
        - _merged_parent  -- True flag for debugging/tracing
        - All original metadata from the representative child
    """
    parent_hits = [c for c in children if c.get("parent_id")]
    plain_hits  = [c for c in children if not c.get("parent_id")]

    unique_pids = group_children_by_parent(parent_hits)
    parent_data = fetch_parent_chunks(unique_pids, parent_cache=parent_cache)

    # Map each parent_id -> first (highest-ranked) child that referenced it
    first_child: Dict[str, ChunkHit] = {}
    for child in parent_hits:
        pid = child.get("parent_id", "") or ""
        if pid and pid not in first_child:
            first_child[pid] = child

    merged: List[ChunkHit] = []

    for pd in parent_data:
        pid = pd["parent_id"]
        rep = first_child.get(pid, {})
        ctx = dict(rep)
        ctx["context_text"]   = pd["text"]
        ctx["chunk_text"]     = pd["text"]
        ctx["_merged_parent"] = True
        merged.append(ctx)

    for hit in plain_hits:
        ctx = dict(hit)
        ctx.setdefault("context_text", hit.get("chunk_text", ""))
        merged.append(ctx)

    logger.info(
        "merge_children_to_parents: %d children -> %d context chunks "
        "(%d unique parents + %d plain hits).",
        len(children), len(merged), len(parent_data), len(plain_hits),
    )
    return merged
