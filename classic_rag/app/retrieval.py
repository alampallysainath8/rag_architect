"""
Retrieval module — performs semantic search against the Pinecone index.
Uses integrated embedding, so we search with raw text (no manual embedding needed).
"""

from typing import List, Dict
from pinecone import Pinecone

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    TOP_K,
)

_pc = Pinecone(api_key=PINECONE_API_KEY)


def search(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Perform a semantic vector search in Pinecone using integrated embedding.
    Pinecone embeds the query text automatically.

    Returns a list of dicts with keys: id, score, chunk_text, source.
    """
    index = _pc.Index(PINECONE_INDEX_NAME)

    # With integrated embedding, use index.search() with text input
    results = index.search(
        namespace=PINECONE_NAMESPACE,
        query={
            "top_k": top_k,
            "inputs": {"text": query},
        },
        fields=["chunk_text", "source", "pages"],
    )

    hits = []
    for item in results.get("result", {}).get("hits", []):
        hits.append(
            {
                "id": item.get("_id", ""),
                "score": item.get("_score", 0.0),
                "chunk_text": item.get("fields", {}).get("chunk_text", ""),
                "source": item.get("fields", {}).get("source", ""),
                "pages": item.get("fields", {}).get("pages", ""),
            }
        )

    return hits


if __name__ == "__main__":
    import sys

    print("=== Retrieval Test ===")
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What was Apple's revenue?"
    print(f"Query: {query}\n")

    results = search(query, top_k=3)
    print(f"Found {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r['score']:.4f} | source={r['source']}")
        print(f"      {r['chunk_text'][:150]}...\n")
    print("✅ Retrieval test passed!")