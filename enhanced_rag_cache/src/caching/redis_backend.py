"""
Redis Cache Backend
===================

Production-grade, Redis-backed implementation of the CacheBackend interface.

Key design:
  - All data stored as JSON blobs (bytes, decode_responses=False).
  - Embeddings serialised as hex-encoded numpy float32 bytes.
  - doc_version invalidation: stale entries are deleted on read.
  - hit_count/last_hit_at updated on every cache hit.
  - source_filter scoping for Tier 2 & Tier 3.
  - Document hash deduplication via Redis Sets & JSON blobs.

Redis key scheme:
  cache:exact:<sha256>          → JSON blob (Tier 1)
  cache:semantic:<uuid>         → JSON blob (Tier 2)
  cache:semantic:index          → Redis Set  (Tier 2 UUIDs)
  cache:retrieval:<uuid>        → JSON blob (Tier 3)
  cache:retrieval:index         → Redis Set  (Tier 3 UUIDs)
  cache:metadata:doc_version    → integer string
  doc:hash:<file_hash>          → JSON blob
  doc:hash:index                → Redis Set
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Optional

import redis

from src.caching.base import CacheBackend
from src.caching.redis_client import get_redis_client
from src.utils.embeddings import (
    cosine_similarity,
    bytes_to_embedding,
    embedding_to_bytes,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Key Prefixes ───────────────────────────────────────────────────────────────
_PFX_EXACT = "cache:exact"
_PFX_SEM   = "cache:semantic"
_PFX_RET   = "cache:retrieval"
_PFX_META  = "cache:metadata"
_PFX_DOC   = "doc:hash"


class RedisCacheBackend(CacheBackend):
    """
    Redis-backed cache implementing all three tiers plus document versioning
    and hash-based deduplication.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis_url = redis_url
        self._r: redis.Redis = get_redis_client()  # type: ignore[assignment]
        if self._r is None:
            raise ConnectionError(
                f"Could not connect to Redis at {redis_url}. "
                "Check that Redis is running or switch to the SQLite backend."
            )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get(self, key: str) -> Optional[dict]:
        """GET a key and JSON-decode. Returns None if missing."""
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def _setex(self, key: str, ttl: int, data: dict) -> None:
        """JSON-encode data and SET with a TTL."""
        self._r.setex(key, ttl, json.dumps(data).encode())

    def _set(self, key: str, data: dict) -> None:
        """JSON-encode and SET without TTL."""
        self._r.set(key, json.dumps(data).encode())

    # ── Tier 1: Exact Cache ────────────────────────────────────────────────

    def get_exact(self, query_hash: str) -> Optional[dict]:
        key = f"{_PFX_EXACT}:{query_hash}"

        entry = self._get(key)
        if entry is None:
            return None

        # Doc-version staleness check
        current_version = self.get_doc_version()
        if entry.get("doc_version", 0) < current_version:
            self._r.delete(key)
            logger.debug(f"Exact cache stale (doc_version) for key {key!r}")
            return None

        # Update hit stats (re-store with remaining TTL)
        ttl_remaining = self._r.ttl(key)
        entry["hit_count"] = entry.get("hit_count", 0) + 1
        entry["last_hit_at"] = time.time()
        if ttl_remaining and ttl_remaining > 0:
            self._r.setex(key, ttl_remaining, json.dumps(entry).encode())

        return {
            "question":    entry["question"],
            "answer":      entry["answer"],
            "sources_json": entry["sources_json"],
        }

    def set_exact(
        self,
        query_hash: str,
        question: str,
        answer: str,
        sources_json: str,
        doc_version: int,
        ttl_seconds: int,
    ) -> None:
        key = f"{_PFX_EXACT}:{query_hash}"
        data = {
            "question":    question,
            "answer":      answer,
            "sources_json": sources_json,
            "doc_version": doc_version,
            "created_at":  time.time(),
            "hit_count":   0,
            "last_hit_at": None,
        }
        self._setex(key, ttl_seconds, data)

    # ── Tier 2: Semantic Cache ─────────────────────────────────────────────

    def get_semantic(
        self,
        embedding: list[float],
        threshold: float,
        source_filter: str = "",
    ) -> Optional[dict]:
        index_key = f"{_PFX_SEM}:index"
        entry_ids = self._r.smembers(index_key)

        current_version = self.get_doc_version()
        best_match: Optional[dict] = None
        best_id: Optional[bytes] = None
        best_similarity = 0.0

        for raw_id in entry_ids:
            entry_key = f"{_PFX_SEM}:{raw_id.decode()}"
            entry = self._get(entry_key)
            if entry is None:
                # Key expired but still in set — will be cleaned by cleanup_expired
                continue

            # Filter by source
            if entry.get("source_filter", "") != source_filter:
                continue

            # Doc-version staleness
            if entry.get("doc_version", 0) < current_version:
                self._r.delete(entry_key)
                self._r.srem(index_key, raw_id)
                continue

            # Compute similarity
            cached_emb = bytes_to_embedding(bytes.fromhex(entry["embedding_hex"]))
            sim = cosine_similarity(embedding, cached_emb)

            if sim >= threshold and sim > best_similarity:
                best_similarity = sim
                best_match = entry
                best_id = raw_id

        if best_match is None or best_id is None:
            return None

        # Update hit stats
        entry_key = f"{_PFX_SEM}:{best_id.decode()}"
        ttl_remaining = self._r.ttl(entry_key)
        best_match["hit_count"] = best_match.get("hit_count", 0) + 1
        best_match["last_hit_at"] = time.time()
        if ttl_remaining and ttl_remaining > 0:
            self._r.setex(entry_key, ttl_remaining, json.dumps(best_match).encode())

        return {
            "question":     best_match["question"],
            "answer":       best_match["answer"],
            "sources_json": best_match["sources_json"],
            "similarity":   best_similarity,
        }

    def set_semantic(
        self,
        question: str,
        embedding: list[float],
        answer: str,
        sources_json: str,
        doc_version: int,
        ttl_seconds: int,
        source_filter: str = "",
    ) -> None:
        entry_id = str(uuid.uuid4())
        entry_key = f"{_PFX_SEM}:{entry_id}"
        index_key = f"{_PFX_SEM}:index"

        data = {
            "question":      question,
            "embedding_hex": embedding_to_bytes(embedding).hex(),
            "answer":        answer,
            "sources_json":  sources_json,
            "source_filter": source_filter,
            "doc_version":   doc_version,
            "created_at":    time.time(),
            "hit_count":     0,
            "last_hit_at":   None,
        }
        self._setex(entry_key, ttl_seconds, data)
        self._r.sadd(index_key, entry_id)

    # ── Tier 3: Retrieval Cache ────────────────────────────────────────────

    def get_retrieval(
        self,
        embedding: list[float],
        threshold: float,
        source_filter: str = "",
    ) -> Optional[dict]:
        index_key = f"{_PFX_RET}:index"
        entry_ids = self._r.smembers(index_key)

        current_version = self.get_doc_version()
        best_match: Optional[dict] = None
        best_id: Optional[bytes] = None
        best_similarity = 0.0

        for raw_id in entry_ids:
            entry_key = f"{_PFX_RET}:{raw_id.decode()}"
            entry = self._get(entry_key)
            if entry is None:
                continue

            if entry.get("source_filter", "") != source_filter:
                continue

            if entry.get("doc_version", 0) < current_version:
                self._r.delete(entry_key)
                self._r.srem(index_key, raw_id)
                continue

            cached_emb = bytes_to_embedding(bytes.fromhex(entry["embedding_hex"]))
            sim = cosine_similarity(embedding, cached_emb)

            if sim >= threshold and sim > best_similarity:
                best_similarity = sim
                best_match = entry
                best_id = raw_id

        if best_match is None or best_id is None:
            return None

        # Update hit stats
        entry_key = f"{_PFX_RET}:{best_id.decode()}"
        ttl_remaining = self._r.ttl(entry_key)
        best_match["hit_count"] = best_match.get("hit_count", 0) + 1
        best_match["last_hit_at"] = time.time()
        if ttl_remaining and ttl_remaining > 0:
            self._r.setex(entry_key, ttl_remaining, json.dumps(best_match).encode())

        return {
            "question":   best_match["question"],
            "chunks_json": best_match["chunks_json"],
            "similarity":  best_similarity,
        }

    def set_retrieval(
        self,
        question: str,
        embedding: list[float],
        chunks_json: str,
        doc_version: int,
        ttl_seconds: int,
        source_filter: str = "",
    ) -> None:
        entry_id = str(uuid.uuid4())
        entry_key = f"{_PFX_RET}:{entry_id}"
        index_key = f"{_PFX_RET}:index"

        data = {
            "question":      question,
            "embedding_hex": embedding_to_bytes(embedding).hex(),
            "chunks_json":   chunks_json,
            "source_filter": source_filter,
            "doc_version":   doc_version,
            "created_at":    time.time(),
            "hit_count":     0,
            "last_hit_at":   None,
        }
        self._setex(entry_key, ttl_seconds, data)
        self._r.sadd(index_key, entry_id)

    # ── Document Versioning ────────────────────────────────────────────────

    def get_doc_version(self) -> int:
        key = f"{_PFX_META}:doc_version"
        raw = self._r.get(key)
        return int(raw) if raw else 0

    def bump_doc_version(self) -> int:
        key = f"{_PFX_META}:doc_version"
        new_version = self._r.incr(key)
        logger.info(f"Doc version bumped to {new_version}")
        return int(new_version)

    # ── Cache Management ───────────────────────────────────────────────────

    def clear_all(self) -> dict:
        exact_count = 0
        for key in self._r.scan_iter(f"{_PFX_EXACT}:*"):
            self._r.delete(key)
            exact_count += 1

        sem_ids = self._r.smembers(f"{_PFX_SEM}:index")
        for raw_id in sem_ids:
            self._r.delete(f"{_PFX_SEM}:{raw_id.decode()}")
        self._r.delete(f"{_PFX_SEM}:index")

        ret_ids = self._r.smembers(f"{_PFX_RET}:index")
        for raw_id in ret_ids:
            self._r.delete(f"{_PFX_RET}:{raw_id.decode()}")
        self._r.delete(f"{_PFX_RET}:index")

        return {
            "exact":     exact_count,
            "semantic":  len(sem_ids),
            "retrieval": len(ret_ids),
        }

    def get_stats(self) -> dict:
        # Exact: SCAN for keys
        exact_entries = 0
        exact_hits = 0
        for key in self._r.scan_iter(f"{_PFX_EXACT}:*"):
            exact_entries += 1
            entry = self._get(key.decode() if isinstance(key, bytes) else key)
            if entry:
                exact_hits += entry.get("hit_count", 0)

        # Semantic
        sem_ids = self._r.smembers(f"{_PFX_SEM}:index")
        sem_entries = 0
        sem_hits = 0
        for raw_id in sem_ids:
            entry = self._get(f"{_PFX_SEM}:{raw_id.decode()}")
            if entry:
                sem_entries += 1
                sem_hits += entry.get("hit_count", 0)

        # Retrieval
        ret_ids = self._r.smembers(f"{_PFX_RET}:index")
        ret_entries = 0
        ret_hits = 0
        for raw_id in ret_ids:
            entry = self._get(f"{_PFX_RET}:{raw_id.decode()}")
            if entry:
                ret_entries += 1
                ret_hits += entry.get("hit_count", 0)

        return {
            "backend":     "redis",
            "redis_url":   self._redis_url,
            "doc_version": self.get_doc_version(),
            "exact":       {"entries": exact_entries, "total_hits": exact_hits},
            "semantic":    {"entries": sem_entries,   "total_hits": sem_hits},
            "retrieval":   {"entries": ret_entries,   "total_hits": ret_hits},
        }

    def cleanup_expired(self) -> int:
        """
        Remove index set members whose backing keys have already expired in Redis.
        Redis handles TTL-based expiry automatically; this just syncs the index sets.
        """
        removed = 0

        for tier in (_PFX_SEM, _PFX_RET):
            index_key = f"{tier}:index"
            stale = []
            for raw_id in self._r.smembers(index_key):
                if not self._r.exists(f"{tier}:{raw_id.decode()}"):
                    stale.append(raw_id)
            for raw_id in stale:
                self._r.srem(index_key, raw_id)
                removed += 1

        return removed

    # ── Document Hash Deduplication ────────────────────────────────────────

    def get_document_hash(self, file_hash: str) -> Optional[dict]:
        key = f"{_PFX_DOC}:{file_hash}"
        return self._get(key)

    def set_document_hash(self, file_hash: str, metadata: dict) -> None:
        key = f"{_PFX_DOC}:{file_hash}"
        index_key = f"{_PFX_DOC}:index"
        data = {
            "file_name":   metadata["file_name"],
            "file_size":   metadata["file_size"],
            "chunk_count": metadata["chunk_count"],
            "created_at":  time.time(),
        }
        self._set(key, data)
        self._r.sadd(index_key, file_hash)

    def remove_document_hash_by_name(self, file_name: str) -> bool:
        index_key = f"{_PFX_DOC}:index"
        hash_ids = self._r.smembers(index_key)
        for raw_hash in hash_ids:
            file_hash = raw_hash.decode()
            entry = self._get(f"{_PFX_DOC}:{file_hash}")
            if entry and entry.get("file_name") == file_name:
                self._r.delete(f"{_PFX_DOC}:{file_hash}")
                self._r.srem(index_key, raw_hash)
                return True
        return False

    def clear_document_hashes(self) -> int:
        index_key = f"{_PFX_DOC}:index"
        hash_ids = self._r.smembers(index_key)
        count = 0
        for raw_hash in hash_ids:
            self._r.delete(f"{_PFX_DOC}:{raw_hash.decode()}")
            count += 1
        self._r.delete(index_key)
        return count
