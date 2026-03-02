"""
Reranker — Pinecone-hosted BGE reranker (bge-reranker-v2-m3).

Combines vector search + reranking in a single Pinecone call for efficiency.
After reranking, parent context is injected for any child chunks from the
parent-child strategy so the LLM always sees the full parent text.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from pinecone import Pinecone

from src.caching.parent_cache import get_parent
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

    hits: List[Dict[str, Any]] = []
    for item in results.get("result", {}).get("hits", []):
        fields: dict = item.get("fields", {})
        hit: Dict[str, Any] = {
            "id": item.get("_id", ""),
            "score": item.get("_score", 0.0),
            "chunk_text": fields.get("chunk_text", ""),
            "source": fields.get("source", ""),
            "doc_id": fields.get("doc_id", ""),
            "parent_id": fields.get("parent_id", ""),
            "level": fields.get("level", ""),
            "strategy": fields.get("strategy", ""),
            "h1": fields.get("h1", ""),
            "h2": fields.get("h2", ""),
            "h3": fields.get("h3", ""),
            "h4": fields.get("h4", ""),
            "context_text": fields.get("chunk_text", ""),  # default: same as chunk
        }

        # ── Parent-Child: replace context_text with full parent ───────────
        parent_id = hit["parent_id"]
        if parent_id:
            parent = get_parent(parent_id)
            if parent:
                hit["context_text"] = parent["text"]
                logger.debug(
                    f"Parent context injected for chunk '{hit['id']}' "
                    f"(parent_id='{parent_id}', "
                    f"parent_len={len(parent['text'])})"
                )
            else:
                logger.debug(
                    f"Parent '{parent_id}' not found in Redis cache — "
                    f"using child chunk text as context."
                )

        hits.append(hit)

    logger.info(
        f"Reranked: {len(hits)} hits returned for query='{query[:60]}'"
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
    enriched = []
    for hit in chunks:
        hit = dict(hit)   # shallow copy
        parent_id = hit.get("parent_id", "")
        if parent_id:
            parent = get_parent(parent_id)
            if parent:
                hit["context_text"] = parent["text"]
        enriched.append(hit)
    return enriched
