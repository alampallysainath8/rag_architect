"""
Caching Layer Tests  (2 test cases)
=====================================

Case 1 — Tier-1 Exact Cache: hit, miss, and normalisation
  Mocks the Redis client. Verifies set→get round-trip and that a
  differently-phrased query misses.

Case 2 — Tier-3 Retrieval Cache: chunk list round-trip through Redis
  Mocks Redis + embed_text. Verifies chunks are stored and recovered
  accurately when similarity exceeds the threshold.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

# ── Case 1: Exact Cache ───────────────────────────────────────────────────────

def test_exact_cache_hit_miss_and_normalisation():
    """
    set() stores JSON; get() on same (normalised) query returns payload.
    A different query returns None.
    """
    store: dict = {}

    mock_redis = MagicMock()
    mock_redis.get.side_effect    = lambda k: store.get(k)
    mock_redis.set.side_effect    = lambda k, v, ex=None: store.update({k: v})
    mock_redis.delete.side_effect = lambda k: store.pop(k, None)

    with patch("src.caching.exact_cache.get_client", return_value=mock_redis):
        from src.caching import exact_cache

        payload = {"answer": "Revenue was $500M.", "sources": [{"source": "report.pdf"}]}
        query   = "What was the revenue?"

        # --- set ---
        exact_cache.set(query, payload.copy())
        assert len(store) == 1, "Expected exactly one Redis key after set()"

        # --- hit on exact (normalised) match ---
        result = exact_cache.get(query)
        assert result is not None, "Expected a cache hit for the same query"
        assert result["answer"] == payload["answer"]

        # --- hit is case/punctuation insensitive ---
        result2 = exact_cache.get("what was the revenue?")
        assert result2 is not None, "Normalised query should also hit"

        # --- miss on different query ---
        miss = exact_cache.get("What are the operating costs?")
        assert miss is None, "Different query must return None"


# ── Case 2: Retrieval Cache chunk round-trip ──────────────────────────────────

def test_retrieval_cache_stores_and_retrieves_chunks():
    """
    set() serialises chunk list to Redis; get() with a similar query
    (sim >= threshold via mocked embeddings) returns the same chunks.
    """
    store: dict = {}
    hash_store: dict = {}

    mock_redis = MagicMock()
    mock_redis.get.side_effect    = lambda k: store.get(k)
    mock_redis.set.side_effect    = lambda k, v, ex=None: store.update({k: v})
    mock_redis.hset.side_effect   = lambda k, eid, v: hash_store.update({eid: v})
    mock_redis.hgetall.side_effect= lambda k: hash_store.copy()
    mock_redis.hlen.side_effect   = lambda k: len(hash_store)
    mock_redis.hdel.side_effect   = lambda k, eid: hash_store.pop(eid, None)
    mock_redis.expire.return_value = True

    # Fixed embedding — cosine_similarity(v, v) == 1.0 > 0.80 threshold
    fixed_emb = [0.1] * 1536

    sample_chunks = [
        {"id": "doc::parent_0::child_0", "chunk_text": "Revenue was $500M.", "score": 0.92},
        {"id": "doc::parent_0::child_1", "chunk_text": "Costs were stable.",  "score": 0.87},
    ]

    with (
        patch("src.caching.retrieval_cache.get_client", return_value=mock_redis),
        patch("src.caching.retrieval_cache.embed_text",  return_value=fixed_emb),
    ):
        from src.caching import retrieval_cache

        retrieval_cache.set("What was the revenue?", sample_chunks)

        # Verify Redis has the embedding and chunk keys
        chunk_keys = [k for k in store if "chunks" in k]
        assert chunk_keys, "Expected a chunks key in Redis store"

        retrieved = retrieval_cache.get("What was the revenue?")
        assert retrieved is not None, "Expected a cache hit for similar query"
        assert len(retrieved) == len(sample_chunks), (
            f"Expected {len(sample_chunks)} chunks, got {len(retrieved)}"
        )
        assert retrieved[0]["chunk_text"] == sample_chunks[0]["chunk_text"]
