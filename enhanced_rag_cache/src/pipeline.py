"""
RAG Query Pipeline — wires cache → retrieval → reranking → generation.

This is the core query runner used by the API.

Query flow:
  1. Check Tier-1 Exact Cache      → HIT: return immediately
  2. Check Tier-2 Semantic Cache   → HIT: return immediately
  3. Check Tier-3 Retrieval Cache  → HIT: skip Pinecone, run LLM only
  4. FULL MISS: Pinecone search + rerank + LLM generation
  5. Write-through: store in all three tiers for future queries

Returns a response dict with answer, sources, and cache metadata.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.caching.cache_manager import CacheManager
from src.generation.generator import format_sources, generate_answer
from src.retrieval.reranker import rerank, rerank_preloaded
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cache = CacheManager()


def run_query(
    query: str,
    use_reranker: bool = True,
    debug: bool = False,
    source_filter: str = "",
) -> Dict[str, Any]:
    """
    Execute the full RAG query pipeline with three-tier caching.

    Args:
        query:         User question string.
        use_reranker:  If True, use Pinecone's BGE reranker (recommended).
                       If False, use raw vector search results.
        debug:         If True, include internal pipeline details in response.
        source_filter: Optional document source name to scope cache lookups
                       (matches the ingested document filename). Leave empty
                       to search across all documents.

    Returns:
        Response dict with at minimum:
          {
            "query":            str,
            "answer":           str,
            "sources":          list,
            "cache_hit":        bool,
            "cache_tier":       int | None,
            "cache_tier_name":  str | None,
            "total_latency_ms": float,
          }
    """
    t_start = time.perf_counter()
    pipeline_steps: List[str] = []

    # ── Phase 1: Cache look-up ────────────────────────────────────────────
    cache_result = _cache.lookup(query, source_filter=source_filter)

    # ── Tier 1 or Tier 2 hit → return cached answer directly ─────────────
    if cache_result["cache_hit"] and cache_result["cache_tier"] in (1, 2):
        pipeline_steps.append(f"cache_hit_tier{cache_result['cache_tier']}")
        return _build_response(
            query=query,
            answer=cache_result["answer"],
            sources=cache_result["sources"],
            cache_result=cache_result,
            pipeline_steps=pipeline_steps,
            total_ms=_ms(t_start),
            debug=debug,
        )

    chunks: List[Dict[str, Any]] = []
    tier3_hit = False

    # ── Tier 3 hit → reuse cached chunks, regenerate answer ──────────────
    if cache_result["cache_hit"] and cache_result["cache_tier"] == 3:
        pipeline_steps.append("cache_hit_tier3_retrieval")
        cached_chunks = cache_result["cached_chunks"]
        # Re-inject parent context (Redis look-up) for parent-child chunks
        chunks = rerank_preloaded(query, cached_chunks)
        tier3_hit = True
        logger.info(
            f"Tier-3 hit: generating answer from {len(chunks)} cached chunks."
        )

    # ── Full miss → Pinecone retrieval + optional reranking ───────────────
    if not chunks:
        pipeline_steps.append("pinecone_retrieval")
        if use_reranker:
            pipeline_steps.append("reranking")
            from src.utils.config_loader import cfg
            chunks = rerank(
                query,
                top_k=cfg["retrieval"]["top_k"],
                top_n=cfg["retrieval"]["rerank_top_n"],
            )
        else:
            from src.retrieval.retriever import search
            raw_hits = search(query)
            # Still inject parent context even without formal reranking
            chunks = rerank_preloaded(query, raw_hits)

    # ── Phase 2: LLM Generation ───────────────────────────────────────────
    pipeline_steps.append("generation")
    answer = generate_answer(query, chunks)
    sources = format_sources(chunks)

    # ── Phase 3: Write-through cache ────────────────────────────────────
    pipeline_steps.append("cache_write")
    _cache.store(query, answer, sources, chunks, source_filter=source_filter, lookup_result=cache_result)

    return _build_response(
        query=query,
        answer=answer,
        sources=sources,
        cache_result=cache_result,
        pipeline_steps=pipeline_steps,
        total_ms=_ms(t_start),
        debug=debug,
        chunks=chunks if debug else None,
        tier3_hit=tier3_hit,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_response(
    query: str,
    answer: str,
    sources: list,
    cache_result: Dict[str, Any],
    pipeline_steps: List[str],
    total_ms: float,
    debug: bool = False,
    chunks: Optional[List[Dict]] = None,
    tier3_hit: bool = False,
) -> Dict[str, Any]:
    resp: Dict[str, Any] = {
        "query": query,
        "answer": answer,
        "sources": sources,
        "cache_hit": cache_result["cache_hit"],
        "cache_tier": cache_result["cache_tier"],
        "cache_tier_name": cache_result["cache_tier_name"],
        "cache_similarity": cache_result.get("cache_similarity"),
        "tier3_regen": tier3_hit,
        "total_latency_ms": total_ms,
    }
    if debug:
        resp["pipeline_steps"] = pipeline_steps
        resp["raw_chunks"] = chunks or []
    return resp


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
