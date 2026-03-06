"""
Reranker — Pinecone-hosted BGE reranker (bge-reranker-v2-m3).

Combines vector search + reranking in a single Pinecone call for efficiency.
After reranking, child hits are passed through merge_children_to_parents()
which deduplicates by parent_id, performs one Redis GET per unique parent,
and returns one context entry per parent — eliminating duplicate LLM tokens.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from pinecone import Pinecone

from src.retrieval.parent_merger import merge_children_to_parents
from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PCFG = cfg["pinecone"]
_RCFG = cfg["retrieval"]

_pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))


def rerank(
    query: str,
    top_k: int = _RCFG["top_k"],
    top_n: int = _RCFG["rerank_top_n"],
    namespace: str = _PCFG["namespace"],
) -> List[Dict[str, Any]]:
    """
    Run Pinecone search + BGE reranking in one API call.

    For each result that is a parent-child *child* chunk, the parent text is
    fetched from Redis and attached as `context_text`. The LLM generator then
    uses `context_text` (parent) instead of `chunk_text` (child).

    Returns:
        List of reranked hit dicts (top_n), each with optional `context_text`.
    """
    index = _pc.Index(_PCFG["index_name"])
    try:
        results = index.search(
            namespace=namespace,
            query={
                "top_k": top_k,
                "inputs": {"text": query},
            },
            rerank={
                "model": _PCFG["rerank_model"],
                "top_n": top_n,
                "rank_fields": ["chunk_text"],
            },
            fields=["chunk_text", "source", "doc_id", "parent_id", "level",
                    "chunk_index", "strategy", "h1", "h2", "h3", "h4"],
        )
    except Exception as exc:
        logger.error(f"Pinecone rerank failed: {exc}")
        raise

    # ── Collect raw child hits from Pinecone ────────────────────────────
    raw_hits: List[Dict[str, Any]] = []
    for item in results.get("result", {}).get("hits", []):
        fields: dict = item.get("fields", {})
        raw_hits.append({
            "id":         item.get("_id", ""),
            "score":      item.get("_score", 0.0),
            "chunk_text": fields.get("chunk_text", ""),
            "source":     fields.get("source", ""),
            "doc_id":     fields.get("doc_id", ""),
            "parent_id":  fields.get("parent_id", ""),
            "level":      fields.get("level", ""),
            "strategy":   fields.get("strategy", ""),
            "h1":         fields.get("h1", ""),
            "h2":         fields.get("h2", ""),
            "h3":         fields.get("h3", ""),
            "h4":         fields.get("h4", ""),
        })

    logger.info(
        "rerank: %d raw hits from Pinecone for query='%s'",
        len(raw_hits), query[:60],
    )

    # ── Merge children -> unique parents (1 Redis GET per unique parent) ─
    hits = merge_children_to_parents(raw_hits)

    logger.info(
        "rerank: %d context chunks after parent merge for query='%s'",
        len(hits), query[:60],
    )
    return hits


def rerank_preloaded(
    query: str,
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Re-run parent context injection on pre-loaded chunks
    (used when chunks come from the Tier-3 retrieval cache).

    Args:
        query:  User query (unused but kept for API symmetry).
        chunks: List of hit dicts from the retrieval cache.

    Returns:
        Same list with `context_text` populated from Redis when available.
    """
    logger.debug(
        "rerank_preloaded: %d cached chunks for query='%s'",
        len(chunks), (query or "")[:60],
    )
    return merge_children_to_parents(chunks)
