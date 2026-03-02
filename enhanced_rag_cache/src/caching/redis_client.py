"""
Shared Redis client factory.

Creates a Redis connection via URL (redis://host:port/db) with
decode_responses=False so that binary embeddings (stored as hex bytes)
can be round-tripped correctly.

Usage:
    from src.caching.redis_client import get_redis_client
    r = get_redis_client()   # may be None if Redis is unreachable
"""

from __future__ import annotations

import os
from typing import Optional

import redis as _redis_lib

from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_redis_client() -> Optional[_redis_lib.Redis]:
    """
    Connect to Redis using REDIS_URL (env var) or config.yaml redis.url.

    Returns:
        A connected redis.Redis instance, or None if the connection fails.
    """
    redis_url: str = os.getenv(
        "REDIS_URL",
        cfg.get("redis", {}).get("url", "redis://localhost:6379/0"),
    )
    try:
        client = _redis_lib.from_url(
            redis_url,
            decode_responses=False,          # raw bytes — required for hex-embedded embeddings
            socket_connect_timeout=3,
        )
        client.ping()
        logger.info(f"Redis connection established: {redis_url}")
        return client
    except Exception as exc:
        logger.warning(
            f"Redis unavailable ({exc}). "
            "If using Redis backend, queries will fail. "
            "Switch to 'sqlite' backend for offline use."
        )
        return None
