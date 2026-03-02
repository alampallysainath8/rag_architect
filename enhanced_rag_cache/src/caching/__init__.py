"""
Caching package — singleton factory for the cache backend.

Backends:
  "redis"  — RedisCacheBackend  (production)
  "sqlite" — SQLiteCacheBackend (dev / fallback)

The active backend is selected via the CACHE_BACKEND env var or
config.yaml cache.backend setting (default: "redis").

Usage:
    from src.caching import get_cache_backend
    cache = get_cache_backend()
    hit = cache.get_exact(query_hash)
"""

from __future__ import annotations

from src.caching.base import CacheBackend

_backend_instance: CacheBackend | None = None


def get_cache_backend() -> CacheBackend:
    """
    Return the singleton cache backend instance.

    Initialises the backend on first call based on the CACHE_BACKEND
    config value ("redis" or "sqlite").
    """
    global _backend_instance
    if _backend_instance is None:
        from src.utils.config_loader import cfg

        backend_name: str = cfg.get("cache", {}).get("backend", "redis").lower()

        if backend_name == "sqlite":
            from src.caching.sqlite_backend import SQLiteCacheBackend

            db_path: str = cfg.get("cache", {}).get("database_path", "data/rag_cache.db")
            _backend_instance = SQLiteCacheBackend(db_path=db_path)
        else:
            from src.caching.redis_backend import RedisCacheBackend

            redis_url: str = cfg.get("redis", {}).get("url", "redis://localhost:6379/0")
            _backend_instance = RedisCacheBackend(redis_url=redis_url)

    return _backend_instance


def reset_cache_backend() -> None:
    """
    Discard the singleton backend instance (useful in tests or after
    config changes that require a fresh connection).
    """
    global _backend_instance
    _backend_instance = None
