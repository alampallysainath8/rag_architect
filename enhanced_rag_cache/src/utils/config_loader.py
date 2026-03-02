"""
Config loader — reads config.yaml and merges with .env overrides.

Usage:
    from src.utils.config_loader import cfg
    threshold = cfg["cache"]["semantic"]["similarity_threshold"]
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def _load() -> dict[str, Any]:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Allow env overrides for the most common fields
    _apply_env_overrides(data)
    return data


def _apply_env_overrides(data: dict) -> None:
    """Overlay values from environment variables onto the config dict."""
    env_map = {
        "PINECONE_INDEX_NAME": ("pinecone", "index_name"),
        "PINECONE_CLOUD": ("pinecone", "cloud"),
        "PINECONE_REGION": ("pinecone", "region"),
        "REDIS_HOST": ("redis", "host"),
        "REDIS_PORT": ("redis", "port"),
        "REDIS_DB": ("redis", "db"),
        "REDIS_PASSWORD": ("redis", "password"),
        "REDIS_URL": ("redis", "url"),
        "CACHE_BACKEND": ("cache", "backend"),
        "DATABASE_PATH": ("cache", "database_path"),
        "CACHE_ENABLED": ("cache", "enabled"),
        "EXACT_CACHE_TTL": ("cache", "exact", "ttl_seconds"),
        "SEMANTIC_CACHE_TTL": ("cache", "semantic", "ttl_seconds"),
        "SEMANTIC_CACHE_THRESHOLD": ("cache", "semantic", "similarity_threshold"),
        "RETRIEVAL_CACHE_TTL": ("cache", "retrieval", "ttl_seconds"),
        "RETRIEVAL_CACHE_THRESHOLD": ("cache", "retrieval", "similarity_threshold"),
    }
    for env_key, path in env_map.items():
        val = os.getenv(env_key)
        if val is None:
            continue
        if len(path) == 2:
            section, key = path
            if section in data and isinstance(data[section], dict):
                # coerce numeric/bool as needed
                data[section][key] = _coerce(val, data[section].get(key))
        elif len(path) == 3:
            section, sub, key = path
            if section in data and sub in data[section]:
                data[section][sub][key] = _coerce(val, data[section][sub].get(key))


def _coerce(env_val: str, current):
    """Coerce env string to the same type as the existing config value."""
    if isinstance(current, bool):
        return env_val.lower() in ("1", "true", "yes")
    if isinstance(current, int):
        try:
            return int(env_val)
        except ValueError:
            return env_val
    if isinstance(current, float):
        try:
            return float(env_val)
        except ValueError:
            return env_val
    return env_val


# Singleton config object
cfg: dict[str, Any] = _load()
