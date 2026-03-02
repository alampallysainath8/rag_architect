"""
Abstract Cache Backend
======================

Defines the CacheBackend interface that all cache implementations must satisfy.
Concrete backends (Redis, SQLite) inherit from this class.

Tiers:
  - Tier 1 (Exact):     exact SHA-256 match on normalized query
  - Tier 2 (Semantic):  cosine similarity on query embedding ≥ threshold
  - Tier 3 (Retrieval): cosine similarity ≥ lower threshold; reuse chunks, rerun LLM

Document versioning:
  - doc_version is a monotonic integer bumped every time a new document is ingested.
  - Cache entries stamped with an older doc_version are treated as misses and deleted.

Document hash deduplication:
  - A SHA-256 hash of each uploaded file is stored to prevent re-ingesting identical content.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class CacheBackend(ABC):
    """
    Abstract interface for all cache backends.

    All concrete implementations (RedisCacheBackend, SQLiteCacheBackend)
    must implement every abstract method defined here.
    """

    # ── Tier 1: Exact Cache ────────────────────────────────────────────────

    @abstractmethod
    def get_exact(self, query_hash: str) -> Optional[dict]:
        """
        Look up an exact cache entry by normalized query hash.

        Args:
            query_hash: SHA-256 hex digest of the normalized query.

        Returns:
            Dict with keys {question, answer, sources_json} on hit, else None.
        """

    @abstractmethod
    def set_exact(
        self,
        query_hash: str,
        question: str,
        answer: str,
        sources_json: str,
        doc_version: int,
        ttl_seconds: int,
    ) -> None:
        """
        Store an exact cache entry.

        Args:
            query_hash:   SHA-256 hex digest of normalize_query(question).
            question:     Original question text.
            answer:       Generated LLM answer.
            sources_json: JSON-serialised list of source chunks.
            doc_version:  Current document version from get_doc_version().
            ttl_seconds:  Time-to-live in seconds.
        """

    # ── Tier 2: Semantic Cache ─────────────────────────────────────────────

    @abstractmethod
    def get_semantic(
        self,
        embedding: list[float],
        threshold: float,
        source_filter: str = "",
    ) -> Optional[dict]:
        """
        Find the best-matching semantic cache entry.

        Args:
            embedding:     Query embedding vector.
            threshold:     Minimum cosine similarity required for a hit.
            source_filter: Optional document source to scope the search.

        Returns:
            Dict with keys {question, answer, sources_json, similarity} on hit,
            else None.
        """

    @abstractmethod
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
        """
        Store a semantic cache entry with the query embedding.

        Args:
            question:      Original question text.
            embedding:     Question embedding vector.
            answer:        Generated LLM answer.
            sources_json:  JSON-serialised sources.
            doc_version:   Current document version.
            ttl_seconds:   Time-to-live in seconds.
            source_filter: Optional document source scope.
        """

    # ── Tier 3: Retrieval Cache ────────────────────────────────────────────

    @abstractmethod
    def get_retrieval(
        self,
        embedding: list[float],
        threshold: float,
        source_filter: str = "",
    ) -> Optional[dict]:
        """
        Find the best-matching retrieval cache entry (stored chunks, no answer).

        Args:
            embedding:     Query embedding vector.
            threshold:     Minimum cosine similarity for a hit.
            source_filter: Optional document source scope.

        Returns:
            Dict with keys {question, chunks_json, similarity} on hit, else None.
        """

    @abstractmethod
    def set_retrieval(
        self,
        question: str,
        embedding: list[float],
        chunks_json: str,
        doc_version: int,
        ttl_seconds: int,
        source_filter: str = "",
    ) -> None:
        """
        Store a retrieval cache entry containing Pinecone chunks (no answer).

        Args:
            question:      Original question text.
            embedding:     Question embedding vector.
            chunks_json:   JSON-serialised list of retrieved chunks.
            doc_version:   Current document version.
            ttl_seconds:   Time-to-live in seconds.
            source_filter: Optional document source scope.
        """

    # ── Document Versioning ────────────────────────────────────────────────

    @abstractmethod
    def get_doc_version(self) -> int:
        """
        Return the current document version counter.

        Returns:
            Non-negative integer — starts at 0, incremented by bump_doc_version().
        """

    @abstractmethod
    def bump_doc_version(self) -> int:
        """
        Increment the document version counter and return the new value.

        Call this after every successful document ingest or delete to
        invalidate stale cache entries on next access.

        Returns:
            The new (incremented) document version integer.
        """

    # ── Cache Management ───────────────────────────────────────────────────

    @abstractmethod
    def clear_all(self) -> dict:
        """
        Delete all entries from all three cache tiers.

        Returns:
            Dict with keys {exact, semantic, retrieval} mapping to counts deleted.
        """

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Return per-tier statistics.

        Returns:
            Dict with at minimum: {backend, doc_version,
            exact: {entries, total_hits},
            semantic: {entries, total_hits},
            retrieval: {entries, total_hits}}.
        """

    @abstractmethod
    def cleanup_expired(self) -> int:
        """
        Purge expired entries from all tiers.

        Returns:
            Total number of entries deleted.
        """

    # ── Document Hash Deduplication ────────────────────────────────────────

    @abstractmethod
    def get_document_hash(self, file_hash: str) -> Optional[dict]:
        """
        Check if a file with this SHA-256 hash has already been ingested.

        Args:
            file_hash: SHA-256 hex digest of the file contents.

        Returns:
            Dict with keys {file_name, file_size, chunk_count, created_at}
            if the hash is known, else None.
        """

    @abstractmethod
    def set_document_hash(self, file_hash: str, metadata: dict) -> None:
        """
        Record that a file with the given SHA-256 hash has been ingested.

        Args:
            file_hash: SHA-256 hex digest of file contents.
            metadata:  Dict with keys {file_name, file_size, chunk_count}.
        """

    @abstractmethod
    def remove_document_hash_by_name(self, file_name: str) -> bool:
        """
        Remove the hash record for a document by its file name.

        Args:
            file_name: The file name as stored in metadata.

        Returns:
            True if the record existed and was removed, False otherwise.
        """

    @abstractmethod
    def clear_document_hashes(self) -> int:
        """
        Remove all document hash records.

        Returns:
            Number of records deleted.
        """
