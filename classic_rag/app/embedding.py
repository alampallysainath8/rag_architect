"""
Embedding module — upserts document chunks into a Pinecone index
using Pinecone's integrated inference (server-side embeddings).
"""

import time
from typing import List, Dict
from pinecone import Pinecone

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_EMBED_MODEL,
)

# ── Pinecone client (module-level singleton) ─────────────────────────────────
_pc = Pinecone(api_key=PINECONE_API_KEY)


def _get_or_create_index():
    """Return handle to the index, creating it with integrated embedding if needed."""
    if not _pc.has_index(PINECONE_INDEX_NAME):
        print(f"🔨 Creating Pinecone index '{PINECONE_INDEX_NAME}' with integrated embedding ...")
        _pc.create_index_for_model(
            name=PINECONE_INDEX_NAME,
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
            embed={
                "model": PINECONE_EMBED_MODEL,
                "field_map": {"text": "chunk_text"},
            },
        )
        # Wait for the index to be ready
        print("⏳ Waiting for index to be ready ...")
        while not _pc.describe_index(PINECONE_INDEX_NAME).status.get("ready", False):
            time.sleep(1)
        print("✅ Index created and ready!")
    return _pc.Index(PINECONE_INDEX_NAME)


def upsert_chunks(records: List[Dict], batch_size: int = 96) -> int:
    """
    Upsert chunk records into Pinecone.
    The index uses integrated embedding, so we upsert raw text
    and Pinecone handles the embedding automatically via upsert_records().

    Each record must have keys: id, chunk_text, source.
    Returns the total number of records upserted.
    """
    index = _get_or_create_index()
    total = 0

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]

        # With integrated embedding, use upsert_records() with raw text.
        # Pinecone embeds the "chunk_text" field automatically (via field_map).
        pinecone_records = []
        for rec in batch:
            pinecone_records.append(
                {
                    "_id": rec["id"],
                    "chunk_text": rec["chunk_text"],
                    "source": rec["source"],
                    "pages": rec.get("pages", ""),
                }
            )

        index.upsert_records(PINECONE_NAMESPACE, pinecone_records)
        total += len(pinecone_records)

    print(f"📦 Upserted {total} records into '{PINECONE_INDEX_NAME}'")
    return total


if __name__ == "__main__":
    print("=== Embedding Test ===")

    # Test with a small set of dummy chunks
    test_records = [
        {"id": "test::chunk-0", "chunk_text": "Apple reported strong Q4 2024 earnings.", "source": "test.pdf"},
        {"id": "test::chunk-1", "chunk_text": "Nike revenue grew in fiscal year 2025.", "source": "test.pdf"},
    ]

    print(f"Upserting {len(test_records)} test chunks...")
    count = upsert_chunks(test_records)
    print(f"Upserted: {count} records")
    print("✅ Embedding test passed!")