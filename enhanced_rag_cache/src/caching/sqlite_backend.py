"""
SQLite Cache Backend
====================

Development/fallback implementation of the CacheBackend interface using
SQLite with WAL mode for concurrent reads.

Ideal for:
  - Local development without a running Redis instance.
  - Single-server deployments where Redis is not available.
  - Testing the caching logic without external dependencies.

Database schema:
  cache_metadata    — key/value pairs (stores doc_version counter)
  exact_cache       — Tier 1: exact query hash → answer
  semantic_cache    — Tier 2: query embeddings (BLOB) → answer
  retrieval_cache   — Tier 3: query embeddings (BLOB) → chunks JSON
  document_hashes   — deduplication by file SHA-256
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Optional

from src.caching.base import CacheBackend
from src.utils.embeddings import (
    cosine_similarity,
    bytes_to_embedding,
    embedding_to_bytes,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteCacheBackend(CacheBackend):
    """
    SQLite-backed cache implementing all three tiers plus document versioning
    and hash-based deduplication.
    """

    def __init__(self, db_path: str = "data/rag_cache.db") -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_tables()
        logger.info(f"SQLite cache backend initialised at {db_path!r}")

    # ── Connection helper ──────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Open a new SQLite connection with WAL mode and Row factory."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Schema ─────────────────────────────────────────────────────────────

    def _init_tables(self) -> None:
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            INSERT OR IGNORE INTO cache_metadata (key, value) VALUES ('doc_version', '0');

            CREATE TABLE IF NOT EXISTS exact_cache (
                query_hash    TEXT PRIMARY KEY,
                question_text TEXT NOT NULL,
                answer_text   TEXT NOT NULL,
                sources_json  TEXT NOT NULL,
                doc_version   INTEGER NOT NULL DEFAULT 0,
                created_at    REAL NOT NULL,
                ttl_seconds   INTEGER NOT NULL,
                hit_count     INTEGER NOT NULL DEFAULT 0,
                last_hit_at   REAL
            );

            CREATE TABLE IF NOT EXISTS semantic_cache (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                question_text      TEXT NOT NULL,
                question_embedding BLOB NOT NULL,
                answer_text        TEXT NOT NULL,
                sources_json       TEXT NOT NULL,
                source_filter      TEXT NOT NULL DEFAULT '',
                doc_version        INTEGER NOT NULL DEFAULT 0,
                created_at         REAL NOT NULL,
                ttl_seconds        INTEGER NOT NULL,
                hit_count          INTEGER NOT NULL DEFAULT 0,
                last_hit_at        REAL
            );

            CREATE TABLE IF NOT EXISTS retrieval_cache (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                question_text      TEXT NOT NULL,
                question_embedding BLOB NOT NULL,
                chunks_json        TEXT NOT NULL,
                source_filter      TEXT NOT NULL DEFAULT '',
                doc_version        INTEGER NOT NULL DEFAULT 0,
                created_at         REAL NOT NULL,
                ttl_seconds        INTEGER NOT NULL,
                hit_count          INTEGER NOT NULL DEFAULT 0,
                last_hit_at        REAL
            );

            CREATE TABLE IF NOT EXISTS document_hashes (
                file_hash   TEXT PRIMARY KEY,
                file_name   TEXT NOT NULL,
                file_size   INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at  REAL NOT NULL
            );
            """
        )
        conn.commit()
        conn.close()

    # ── Tier 1: Exact Cache ────────────────────────────────────────────────

    def get_exact(self, query_hash: str) -> Optional[dict]:
        conn = self._get_conn()
        now = time.time()
        doc_version = self._get_doc_version_raw(conn)

        row = conn.execute(
            "SELECT question_text, answer_text, sources_json, doc_version, "
            "created_at, ttl_seconds FROM exact_cache WHERE query_hash = ?",
            (query_hash,),
        ).fetchone()

        if row is None:
            conn.close()
            return None

        # TTL check
        if now - row["created_at"] > row["ttl_seconds"]:
            conn.execute("DELETE FROM exact_cache WHERE query_hash = ?", (query_hash,))
            conn.commit()
            conn.close()
            return None

        # Doc-version staleness check
        if row["doc_version"] < doc_version:
            conn.execute("DELETE FROM exact_cache WHERE query_hash = ?", (query_hash,))
            conn.commit()
            conn.close()
            return None

        # Cache hit — update stats
        conn.execute(
            "UPDATE exact_cache SET hit_count = hit_count + 1, last_hit_at = ? "
            "WHERE query_hash = ?",
            (now, query_hash),
        )
        conn.commit()
        conn.close()

        return {
            "question":    row["question_text"],
            "answer":      row["answer_text"],
            "sources_json": row["sources_json"],
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
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO exact_cache
               (query_hash, question_text, answer_text, sources_json,
                doc_version, created_at, ttl_seconds, hit_count, last_hit_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
            (query_hash, question, answer, sources_json, doc_version, time.time(), ttl_seconds),
        )
        conn.commit()
        conn.close()

    # ── Tier 2: Semantic Cache ─────────────────────────────────────────────

    def get_semantic(
        self,
        embedding: list[float],
        threshold: float,
        source_filter: str = "",
    ) -> Optional[dict]:
        conn = self._get_conn()
        now = time.time()
        doc_version = self._get_doc_version_raw(conn)

        rows = conn.execute(
            "SELECT id, question_text, question_embedding, answer_text, sources_json, "
            "source_filter, doc_version, created_at, ttl_seconds "
            "FROM semantic_cache WHERE source_filter = ?",
            (source_filter,),
        ).fetchall()
        conn.close()

        best_match = None
        best_similarity = 0.0

        for row in rows:
            if now - row["created_at"] > row["ttl_seconds"]:
                continue
            if row["doc_version"] < doc_version:
                continue

            cached_emb = bytes_to_embedding(row["question_embedding"])
            sim = cosine_similarity(embedding, cached_emb)

            if sim >= threshold and sim > best_similarity:
                best_similarity = sim
                best_match = row

        if best_match is None:
            return None

        # Update hit stats
        conn = self._get_conn()
        conn.execute(
            "UPDATE semantic_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?",
            (now, best_match["id"]),
        )
        conn.commit()
        conn.close()

        return {
            "question":     best_match["question_text"],
            "answer":       best_match["answer_text"],
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
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO semantic_cache
               (question_text, question_embedding, answer_text, sources_json,
                source_filter, doc_version, created_at, ttl_seconds, hit_count, last_hit_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
            (
                question,
                embedding_to_bytes(embedding),
                answer,
                sources_json,
                source_filter,
                doc_version,
                time.time(),
                ttl_seconds,
            ),
        )
        conn.commit()
        conn.close()

    # ── Tier 3: Retrieval Cache ────────────────────────────────────────────

    def get_retrieval(
        self,
        embedding: list[float],
        threshold: float,
        source_filter: str = "",
    ) -> Optional[dict]:
        conn = self._get_conn()
        now = time.time()
        doc_version = self._get_doc_version_raw(conn)

        rows = conn.execute(
            "SELECT id, question_text, question_embedding, chunks_json, "
            "source_filter, doc_version, created_at, ttl_seconds "
            "FROM retrieval_cache WHERE source_filter = ?",
            (source_filter,),
        ).fetchall()
        conn.close()

        best_match = None
        best_similarity = 0.0

        for row in rows:
            if now - row["created_at"] > row["ttl_seconds"]:
                continue
            if row["doc_version"] < doc_version:
                continue

            cached_emb = bytes_to_embedding(row["question_embedding"])
            sim = cosine_similarity(embedding, cached_emb)

            if sim >= threshold and sim > best_similarity:
                best_similarity = sim
                best_match = row

        if best_match is None:
            return None

        # Update hit stats
        conn = self._get_conn()
        conn.execute(
            "UPDATE retrieval_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?",
            (now, best_match["id"]),
        )
        conn.commit()
        conn.close()

        return {
            "question":   best_match["question_text"],
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
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO retrieval_cache
               (question_text, question_embedding, chunks_json,
                source_filter, doc_version, created_at, ttl_seconds, hit_count, last_hit_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
            (
                question,
                embedding_to_bytes(embedding),
                chunks_json,
                source_filter,
                doc_version,
                time.time(),
                ttl_seconds,
            ),
        )
        conn.commit()
        conn.close()

    # ── Document Versioning ────────────────────────────────────────────────

    def _get_doc_version_raw(self, conn: sqlite3.Connection) -> int:
        """Internal: get doc_version using an existing connection."""
        row = conn.execute(
            "SELECT value FROM cache_metadata WHERE key = 'doc_version'"
        ).fetchone()
        return int(row["value"]) if row else 0

    def get_doc_version(self) -> int:
        conn = self._get_conn()
        version = self._get_doc_version_raw(conn)
        conn.close()
        return version

    def bump_doc_version(self) -> int:
        conn = self._get_conn()
        current = self._get_doc_version_raw(conn)
        new_version = current + 1
        conn.execute(
            "UPDATE cache_metadata SET value = ? WHERE key = 'doc_version'",
            (str(new_version),),
        )
        conn.commit()
        conn.close()
        logger.info(f"Doc version bumped to {new_version}")
        return new_version

    # ── Cache Management ───────────────────────────────────────────────────

    def clear_all(self) -> dict:
        conn = self._get_conn()
        exact_count    = conn.execute("SELECT COUNT(*) FROM exact_cache").fetchone()[0]
        semantic_count = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        retrieval_count = conn.execute("SELECT COUNT(*) FROM retrieval_cache").fetchone()[0]

        conn.execute("DELETE FROM exact_cache")
        conn.execute("DELETE FROM semantic_cache")
        conn.execute("DELETE FROM retrieval_cache")
        conn.commit()
        conn.close()

        return {
            "exact":     exact_count,
            "semantic":  semantic_count,
            "retrieval": retrieval_count,
        }

    def get_stats(self) -> dict:
        conn = self._get_conn()

        def _tier_stats(table: str) -> dict:
            row = conn.execute(
                f"SELECT COUNT(*) AS cnt, COALESCE(SUM(hit_count), 0) AS hits FROM {table}"
            ).fetchone()
            return {"entries": row["cnt"], "total_hits": row["hits"]}

        stats = {
            "backend":     "sqlite",
            "db_path":     self._db_path,
            "doc_version": self._get_doc_version_raw(conn),
            "exact":       _tier_stats("exact_cache"),
            "semantic":    _tier_stats("semantic_cache"),
            "retrieval":   _tier_stats("retrieval_cache"),
        }
        conn.close()
        return stats

    def cleanup_expired(self) -> int:
        conn = self._get_conn()
        now = time.time()
        total = 0

        for table in ("exact_cache", "semantic_cache", "retrieval_cache"):
            cursor = conn.execute(
                f"DELETE FROM {table} WHERE (? - created_at) > ttl_seconds",
                (now,),
            )
            total += cursor.rowcount

        conn.commit()
        conn.close()
        return total

    # ── Document Hash Deduplication ────────────────────────────────────────

    def get_document_hash(self, file_hash: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT file_name, file_size, chunk_count, created_at "
            "FROM document_hashes WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "file_name":   row["file_name"],
            "file_size":   row["file_size"],
            "chunk_count": row["chunk_count"],
            "created_at":  row["created_at"],
        }

    def set_document_hash(self, file_hash: str, metadata: dict) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO document_hashes
               (file_hash, file_name, file_size, chunk_count, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                file_hash,
                metadata["file_name"],
                metadata["file_size"],
                metadata["chunk_count"],
                time.time(),
            ),
        )
        conn.commit()
        conn.close()

    def remove_document_hash_by_name(self, file_name: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM document_hashes WHERE file_name = ?", (file_name,)
        )
        conn.commit()
        removed = cursor.rowcount > 0
        conn.close()
        return removed

    def clear_document_hashes(self) -> int:
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM document_hashes").fetchone()[0]
        conn.execute("DELETE FROM document_hashes")
        conn.commit()
        conn.close()
        return count
