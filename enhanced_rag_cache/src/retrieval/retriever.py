"""
Retriever — semantic vector search against Pinecone.

Uses Pinecone's integrated inference for query embedding (no local model needed).
Returns raw hit dicts including all metadata fields stored at upsert time.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from pinecone import Pinecone

from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PCFG = cfg["pinecone"]
_RCFG = cfg["retrieval"]

_pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))


def search(
    query: str,
    top_k: int = _RCFG["top_k"],
    namespace: str = _PCFG["namespace"],
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant chunks for *query* from Pinecone.

    Pinecone's integrated embedding is used, so the query text is sent as-is.

    Returns:
        List of hit dicts:
        {
          "id":         str,
          "score":      float,
          "chunk_text": str,
          "source":     str,
          "doc_id":     str,
          "parent_id":  str | "",   # present for parent-child child chunks
          "level":      str | "",   # "child" for parent-child chunks
          ...                       # any other metadata fields
        }
    """
    index = _pc.Index(_PCFG["index_name"])
    try:
        results = index.search(
            namespace=namespace,
            query={
                "top_k": top_k,
                "inputs": {"text": query},
            },
            fields=["chunk_text", "source", "doc_id", "parent_id", "level",
                    "chunk_index", "strategy", "h1", "h2", "h3", "h4"],
        )
    except Exception as exc:
        logger.error(f"Pinecone search failed: {exc}")
        raise

    hits: List[Dict[str, Any]] = []
    for item in results.get("result", {}).get("hits", []):
        fields: dict = item.get("fields", {})
        hits.append(
            {
                "id": item.get("_id", ""),
                "score": item.get("_score", 0.0),
                "chunk_text": fields.get("chunk_text", ""),
                "source": fields.get("source", ""),
                "doc_id": fields.get("doc_id", ""),
                "parent_id": fields.get("parent_id", ""),
                "level": fields.get("level", ""),
                "strategy": fields.get("strategy", ""),
                "h1": fields.get("h1", ""),
                "h2": fields.get("h2", ""),
                "h3": fields.get("h3", ""),
                "h4": fields.get("h4", ""),
            }
        )

    logger.info(f"Pinecone search: {len(hits)} hits for query='{query[:60]}'")
    return hits
