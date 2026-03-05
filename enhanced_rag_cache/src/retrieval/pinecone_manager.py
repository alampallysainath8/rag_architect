"""
Pinecone Manager — index creation, upsert, and retrieval for the cache project.

Supports two ingestion modes:
  • parent_child  — only child chunks are upserted; parent chunks are stored in Redis.
  • structure     — all StructChunks are upserted directly.

Index uses Pinecone's integrated embedding (multilingual-e5-large) so raw
chunk_text is sent and embedding is done server-side.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from pinecone import Pinecone

from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PCFG = cfg["pinecone"]
_pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))


def _get_or_create_index():
    """Return a handle to the Pinecone index, creating it if absent."""
    name = _PCFG["index_name"]
    if not _pc.has_index(name):
        logger.info(f"Creating Pinecone index '{name}' with integrated embedding …")
        _pc.create_index_for_model(
            name=name,
            cloud=_PCFG["cloud"],
            region=_PCFG["region"],
            embed={
                "model": _PCFG["embed_model"],
                "field_map": {"text": "text"},
            },
        )
        logger.info("Waiting for index to become ready …")
        for _ in range(60):
            if _pc.describe_index(name).status.get("ready", False):
                break
            time.sleep(2)
        logger.info(f"Index '{name}' is ready.")
    return _pc.Index(name)


# Module-level index handle
_index = None


def _idx():
    global _index
    if _index is None:
        _index = _get_or_create_index()
    return _index


def upsert_records(
    records: List[Dict[str, Any]],
    namespace: str = _PCFG["namespace"],
    batch_size: int = _PCFG["batch_size"],
) -> int:
    """
    Upsert records into Pinecone.

    Each record must contain at minimum: _id, text, source.
    Additional metadata fields are stored as-is.

    Returns total number of records upserted.
    """
    index = _idx()
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        try:
            index.upsert_records(namespace, batch)
            total += len(batch)
            logger.debug(f"Upserted batch {i // batch_size + 1} ({len(batch)} records)")
        except Exception as exc:
            logger.error(f"Pinecone upsert error at batch {i}: {exc}")
            raise
    logger.info(f"Pinecone upsert complete: {total} records in namespace '{namespace}'")
    return total


def delete_document(doc_id: str, namespace: str = _PCFG["namespace"]) -> None:
    """Delete all vectors for a given doc_id using metadata filtering."""
    index = _idx()
    try:
        index.delete(filter={"doc_id": {"$eq": doc_id}}, namespace=namespace)
        logger.info(f"Deleted all vectors for doc_id='{doc_id}' from namespace '{namespace}'")
    except Exception as exc:
        logger.warning(f"Pinecone delete error for doc_id '{doc_id}': {exc}")
