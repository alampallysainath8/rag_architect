"""
Embedding utility — wraps LangChain's OpenAIEmbeddings (text-embedding-3-small)
for use in the semantic and retrieval cache tiers.

NOTE: Pinecone's integrated embedding is used for upsert/search.
This module is ONLY for computing embeddings locally (cache comparison).

Functions:
  embed_query(text)            → list[float]   — single query embedding
  embed_batch(texts)           → list[list[float]]  — batched embeddings
  normalize_query(text)        → str           — lowercase/strip/collapse whitespace
  hash_query(normalized)       → str           — SHA-256 hex digest
  cosine_similarity(a, b)      → float         — cosine similarity in [-1, 1]
  embedding_to_bytes(vec)      → bytes         — numpy float32 packed bytes
  bytes_to_embedding(b)        → list[float]   — unpack bytes back to list
"""

import hashlib
import re
from typing import List

import numpy as np
from langchain_openai import OpenAIEmbeddings

from src.utils.config_loader import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

_EMBED_MODEL: str = cfg["openai"]["embedding_model"]

# Singleton LangChain embeddings client
_lc_embedder: OpenAIEmbeddings | None = None


def _get_embedder() -> OpenAIEmbeddings:
    global _lc_embedder
    if _lc_embedder is None:
        _lc_embedder = OpenAIEmbeddings(model=_EMBED_MODEL)
    return _lc_embedder


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_query(text: str) -> List[float]:
    """
    Compute an embedding vector for a single text string.

    Args:
        text: Input text to embed.

    Returns:
        List of floats (embedding vector).
    """
    text = text.strip().replace("\n", " ")
    return _get_embedder().embed_query(text)


def embed_batch(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """
    Embed a list of texts in batches.

    Args:
        texts: List of text strings.
        batch_size: How many texts to embed per API call.

    Returns:
        List of embedding vectors.
    """
    all_embeddings: List[List[float]] = []
    embedder = _get_embedder()
    for i in range(0, len(texts), batch_size):
        batch = [t.strip().replace("\n", " ") for t in texts[i : i + batch_size]]
        try:
            all_embeddings.extend(embedder.embed_documents(batch))
        except Exception as exc:
            logger.error(f"Batch embedding failed at offset {i}: {exc}")
            raise
    return all_embeddings


def normalize_query(text: str) -> str:
    """
    Normalize a query string for consistent cache key generation.

    Steps:
      1. Lowercase
      2. Strip leading/trailing whitespace
      3. Collapse internal whitespace to single spaces
      4. Remove trailing punctuation (., ?, !)

    Args:
        text: Raw query string.

    Returns:
        Normalized string.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".?!")
    return text


def hash_query(normalized_text: str) -> str:
    """
    Compute a SHA-256 hex digest of a normalized query string.

    Args:
        normalized_text: Output of normalize_query().

    Returns:
        64-character hex string used as the exact cache key.
    """
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Returns:
        Float in [-1, 1] — higher means more similar.
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def embedding_to_bytes(vec: List[float]) -> bytes:
    """
    Serialize an embedding vector to compact numpy float32 bytes.

    Args:
        vec: Embedding as a list of floats.

    Returns:
        Raw bytes (4 bytes per dimension).
    """
    return np.array(vec, dtype=np.float32).tobytes()


def bytes_to_embedding(b: bytes) -> List[float]:
    """
    Deserialize bytes (from embedding_to_bytes) back to a float list.

    Args:
        b: Raw bytes from embedding_to_bytes().

    Returns:
        List of floats.
    """
    return np.frombuffer(b, dtype=np.float32).tolist()
