"""
Cache Manager — Three-Tier Cache Orchestrator
==============================================

Provides a single entry point for the full cache lookup and write-through
logic across all three tiers, delegating to whichever backend is configured
(Redis or SQLite) via the get_cache_backend() factory.

Query flow:
  Tier 1 → Exact Cache     (SHA-256 of normalized query)
  Tier 2 → Semantic Cache  (cosine similarity ≥ threshold on query embedding)
  Tier 3 → Retrieval Cache (lower similarity → reuse chunks, still call LLM)
  FULL MISS → run Pinecone + LLM, then write all three tiers

Usage:
  from src.caching.cache_manager import CacheManager
  cm = CacheManager()

  result = cm.lookup(query)
  if result["cache_hit"]:
      return result          # answer (and optionally chunks) already cached

  # ... run pipeline ...
  cm.store(query, answer, sources, chunks)

Return structure of lookup():
  {
    "cache_hit":        bool,
    "cache_tier":       int | None,   # 1, 2, 3, or None
    "cache_tier_name":  str | None,   # "exact", "semantic", "retrieval", None
    "cache_similarity": float | None,
    "answer":           str | None,
    "sources":          list | None,
    "cached_chunks":    list | None,  # only set for Tier-3 hits
  }
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from src.caching import get_cache_backend
from src.utils.config_loader import cfg
from src.utils.embeddings import embed_query, normalize_query, hash_query
from src.utils.logger import get_logger

logger = get_logger(__name__)

_EXACT_TTL     = cfg["cache"]["exact"]["ttl_seconds"]
_SEMANTIC_TTL  = cfg["cache"]["semantic"]["ttl_seconds"]
_SEMANTIC_THR  = cfg["cache"]["semantic"]["similarity_threshold"]
_RETRIEVAL_TTL = cfg["cache"]["retrieval"]["ttl_seconds"]
_RETRIEVAL_THR = cfg["cache"]["retrieval"]["similarity_threshold"]


class CacheManager:
    """
    Thin orchestrator around the configured cache backend.
    Encapsulates normalize → hash → embed → get/set logic for all three tiers.
    Stateless — safe to instantiate once at module level.
    """

    # ── Lookup ──────────────────────────────────────────────────────────────

    def lookup(
        self,
        query: str,
        source_filter: str = "",
    ) -> Dict[str, Any]:
        """
        Run the three-tier lookup for *query*.

        Args:
            query:         Raw user query string.
            source_filter: Optional document source to scope Tier 2 & 3 lookups.

        Returns:
            Result dict (see module docstring).
        """
        backend = get_cache_backend()

        # ── Tier 1: Exact ──────────────────────────────────────────────────
        normalized = normalize_query(query)
        if source_filter:
            normalized = f"{normalized}|source={source_filter}"
        query_hash = hash_query(normalized)

        exact_hit = backend.get_exact(query_hash)
        if exact_hit:
            logger.info(f"[Cache HIT Tier 1 exact] {query[:60]!r}")
            return {
                "cache_hit":        True,
                "cache_tier":       1,
                "cache_tier_name":  "exact",
                "cache_similarity": 1.0,
                "query_hash":       query_hash,
                "embedding":        None,
                "answer":           exact_hit["answer"],
                "sources":          _parse_sources(exact_hit["sources_json"]),
                "cached_chunks":    None,
            }

        # ── Tier 2: Semantic ───────────────────────────────────────────────
        embedding = embed_query(query)

        semantic_hit = backend.get_semantic(embedding, _SEMANTIC_THR, source_filter)
        if semantic_hit:
            logger.info(
                f"[Cache HIT Tier 2 semantic sim={semantic_hit['similarity']:.3f}] {query[:60]!r}"
            )
            return {
                "cache_hit":        True,
                "cache_tier":       2,
                "cache_tier_name":  "semantic",
                "cache_similarity": semantic_hit["similarity"],
                "query_hash":       query_hash,
                "embedding":        embedding,
                "answer":           semantic_hit["answer"],
                "sources":          _parse_sources(semantic_hit["sources_json"]),
                "cached_chunks":    None,
            }

        # ── Tier 3: Retrieval ──────────────────────────────────────────────
        retrieval_hit = backend.get_retrieval(embedding, _RETRIEVAL_THR, source_filter)
        if retrieval_hit:
            logger.info(
                f"[Cache HIT Tier 3 retrieval sim={retrieval_hit['similarity']:.3f}] {query[:60]!r}"
            )
            return {
                "cache_hit":        True,
                "cache_tier":       3,
                "cache_tier_name":  "retrieval",
                "cache_similarity": retrieval_hit["similarity"],
                "query_hash":       query_hash,
                "embedding":        embedding,
                "answer":           None,        # caller must call generator
                "sources":          None,
                "cached_chunks":    json.loads(retrieval_hit["chunks_json"]),
            }

        # ── Total miss ─────────────────────────────────────────────────────
        logger.info(f"[Cache MISS] {query[:60]!r}")
        return {
            "cache_hit":        False,
            "cache_tier":       None,
            "cache_tier_name":  None,
            "cache_similarity": None,
            "query_hash":       query_hash,
            "embedding":        embedding,
            "answer":           None,
            "sources":          None,
            "cached_chunks":    None,
        }

    # ── Write-through ────────────────────────────────────────────────────────

    def store(
        self,
        query: str,
        answer: str,
        sources: List[Any],
        chunks: List[Dict[str, Any]],
        source_filter: str = "",
        lookup_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write a completed pipeline result into all three cache tiers.

        Args:
            query:          Raw user query string.
            answer:         Generated LLM answer.
            sources:        Source dicts shown to the user.
            chunks:         Raw retrieved/reranked chunks (for Tier 3).
            source_filter:  Optional document source scope.
            lookup_result:  Dict returned by lookup() — avoids re-computing
                            query_hash and embedding if already available.
        """
        backend = get_cache_backend()
        doc_version = backend.get_doc_version()

        # Reuse pre-computed hash/embedding from lookup if available
        if lookup_result:
            query_hash = lookup_result.get("query_hash") or _make_hash(query, source_filter)
            embedding  = lookup_result.get("embedding")  or embed_query(query)
        else:
            query_hash = _make_hash(query, source_filter)
            embedding  = embed_query(query)

        sources_json = json.dumps(
            [s if isinstance(s, dict) else vars(s) for s in sources]
        )
        chunks_json = json.dumps(chunks)

        backend.set_exact(
            query_hash, query, answer, sources_json, doc_version, _EXACT_TTL
        )
        backend.set_semantic(
            query, embedding, answer, sources_json, doc_version, _SEMANTIC_TTL, source_filter
        )
        backend.set_retrieval(
            query, embedding, chunks_json, doc_version, _RETRIEVAL_TTL, source_filter
        )

    # ── Admin helpers ────────────────────────────────────────────────────────

    def flush_all(self) -> Dict[str, int]:
        """Wipe all three cache tiers. Returns counts deleted per tier."""
        return get_cache_backend().clear_all()

    def stats(self) -> Dict[str, Any]:
        """Aggregate stats from the backend, shaped for the API response."""
        raw = get_cache_backend().get_stats()
        return {
            "tier1_exact":     raw.get("exact",     {}),
            "tier2_semantic":  raw.get("semantic",  {}),
            "tier3_retrieval": raw.get("retrieval", {}),
            "doc_version":     raw.get("doc_version", 0),
            "backend":         raw.get("backend", "unknown"),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_hash(query: str, source_filter: str = "") -> str:
    normalized = normalize_query(query)
    if source_filter:
        normalized = f"{normalized}|source={source_filter}"
    return hash_query(normalized)


def _parse_sources(sources_json: str) -> List[Dict[str, Any]]:
    """Deserialise cached sources JSON back to a list of dicts."""
    try:
        return json.loads(sources_json)
    except (json.JSONDecodeError, TypeError):
        return []


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
