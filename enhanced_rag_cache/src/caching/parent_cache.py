"""
Parent Chunk Cache
==================

Stores large parent chunks in Redis so that after Pinecone returns a
child chunk match, the full parent text can be looked up locally without
an additional Pinecone query.

Keys:  parent:<parent_id>   → JSON-serialised ParentChunk data
TTL:   config.yaml → cache.parent.ttl_seconds (default 7 days — static per doc)
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Any

from src.caching.redis_client import get_redis_client
from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TTL: int = cfg["cache"]["parent"]["ttl_seconds"]
_PREFIX = "parent"


def _make_key(*parts: str) -> str:
    return ":".join([_PREFIX] + list(parts))


def store_parents(parents: List[Dict[str, Any]]) -> None:
    """
    Persist a list of parent chunk dicts into Redis.

    Each dict must contain at minimum:
      { "parent_id": str, "text": str, "source": str, "doc_id": str }

    Args:
        parents: List of parent chunk dicts (from ParentChunk.to_redis_dict()).
    """
    client = get_redis_client()
    if client is None:
        return

    pipe = client.pipeline()
    for p in parents:
        key = _make_key(p["parent_id"])
        pipe.set(key, json.dumps(p).encode(), ex=_TTL)
    try:
        pipe.execute()
        logger.debug(f"Stored {len(parents)} parent chunks in Redis.")
    except Exception as exc:
        logger.warning(f"Parent cache STORE error: {exc}")


def get_parent(parent_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a single parent chunk dict by its ID.

    Args:
        parent_id: Parent chunk identifier (e.g. "report.pdf::parent_3").

    Returns:
        Parent chunk dict or None if not found / expired.
    """
    client = get_redis_client()
    if client is None:
        return None

    key = _make_key(parent_id)
    try:
        raw = client.get(key)
        if raw:
            return json.loads(raw)
    except Exception as exc:
        logger.warning(f"Parent cache GET error for '{parent_id}': {exc}")
    return None


def flush_document(doc_id: str) -> int:
    """Delete all parent chunks for a specific document."""
    client = get_redis_client()
    if client is None:
        return 0
    pattern = _make_key(f"{doc_id}::*")
    try:
        keys = client.keys(pattern)
        if keys:
            return client.delete(*keys)
        return 0
    except Exception as exc:
        logger.warning(f"Parent cache flush error for doc '{doc_id}': {exc}")
        return 0


def flush_all() -> int:
    """Delete ALL parent cache entries."""
    client = get_redis_client()
    if client is None:
        return 0
    pattern = _make_key("*")
    try:
        keys = client.keys(pattern)
        if keys:
            return client.delete(*keys)
        return 0
    except Exception as exc:
        logger.warning(f"Parent cache FLUSH ALL error: {exc}")
        return 0


def stats() -> Dict[str, Any]:
    client = get_redis_client()
    if client is None:
        return {"name": "parent", "entries": 0, "redis": False}
    pattern = _make_key("*")
    try:
        count = len(client.keys(pattern))
        return {"name": "parent", "entries": count, "ttl_seconds": _TTL, "redis": True}
    except Exception as exc:
        logger.warning(f"Parent cache STATS error: {exc}")
        return {"name": "parent", "entries": 0, "redis": False}
